"""The long-running index pipeline. Qt-agnostic: progress/log/cancel are plain callables,
so it is unit-testable headless and wrapped by a QThread worker in the UI.

Two granularities (``cfg.index_granularity``):
  * "folder": one representative image per leaf folder (near-duplicate scene folders).
  * "image":  every image is its own record (diverse, individually-labeled datasets);
              captions come from YOLO labels (class-id -> name) when present.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from imgsearch.backends.base import Captioner, Embedder
from imgsearch.config import AppConfig
from imgsearch.core.models import IndexProgress
from imgsearch.index import labels, pairing, repr_select, walker
from imgsearch.store.chroma_store import ChromaStore, folder_id

ProgressCb = Callable[[IndexProgress], None]
LogCb = Callable[[str], None]
CancelCb = Callable[[], bool]
ThumbCb = Callable[[str], None]

_FLUSH_EVERY = 128


def _noop(*_a, **_k) -> None:
    return None


def _never() -> bool:
    return False


def image_id(path: str) -> str:
    norm = str(Path(path)).replace("\\", "/").lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _mkey(path: str) -> str:
    try:
        return f"{os.path.getmtime(path):.3f}"
    except OSError:
        return "0"


def _image_mkey(image_path: str) -> str:
    """Skip key for per-image records: image mtime + label mtime, so editing the
    YOLO label file (re-annotation) also invalidates the record."""
    m = _mkey(image_path)
    lp = labels.label_path_for(image_path)
    return f"{m}|{_mkey(str(lp))}" if lp is not None else m


class Indexer:
    def __init__(
        self,
        embedder: Embedder,
        captioner: Optional[Captioner],
        store: ChromaStore,
        config: AppConfig,
    ) -> None:
        self.embedder = embedder
        self.captioner = captioner
        self.store = store
        self.cfg = config

    # ------------------------------------------------------------------ records
    def _embed_checked(self, image_path: str) -> np.ndarray:
        vec = self.embedder.embed_image([image_path])[0]
        if not np.all(np.isfinite(vec)):
            raise ValueError("non-finite embedding vector")
        return vec

    def _base_meta(self, folder: str, image_path: str, members: int, method: str, mkey: str) -> dict:
        return {
            "leaf_folder": folder,
            "representative_image": image_path,
            "member_count": int(members),
            "rep_method": method,
            "model_id": getattr(self.embedder, "name", "?"),
            "embed_dim": int(getattr(self.embedder, "dim", 0)),
            "granularity": self.cfg.index_granularity,
            "mtime": mkey,
        }

    def _folder_record(self, dirpath: str, images: list[str], mkey: str):
        sidecar = pairing.folder_sidecar_text(dirpath, images)
        rep = repr_select.choose_representative(images, self.embedder, self.cfg.max_members_for_repr)
        caption = ""
        if self.cfg.caption_enabled and self.captioner is not None:
            caption = self.captioner.caption(rep.image_path, context=sidecar)
        vec = self._embed_checked(rep.image_path)
        doc = (caption + "\n" + sidecar).strip() or rep.image_path
        meta = self._base_meta(dirpath, rep.image_path, rep.member_count, rep.method, mkey)
        meta.update(caption=caption, sidecar_text=sidecar[:400])
        return folder_id(dirpath), vec, doc, meta

    def _label_caption(self, image_path: str) -> str:
        """Caption derived ONLY from the label/sidecar text files (no model calls)."""
        lp = labels.label_path_for(image_path)
        if lp is None:
            return ""
        caption = labels.yolo_caption(lp, self.cfg.class_names)  # "" if not YOLO
        if not caption:
            caption = pairing.sidecar_for(image_path)  # real text sidecar (skips YOLO)
        return caption

    def _image_caption(self, image_path: str) -> str:
        caption = self._label_caption(image_path)
        if not caption and self.cfg.caption_enabled and self.captioner is not None:
            caption = self.captioner.caption(image_path, "")
        return caption

    def _image_record(self, dirpath: str, image_path: str, mkey: str):
        caption = self._image_caption(image_path)
        vec = self._embed_checked(image_path)
        doc = caption or os.path.basename(image_path)
        meta = self._base_meta(dirpath, image_path, 1, "image", mkey)
        meta.update(caption=caption, sidecar_text=caption[:400])
        return image_id(image_path), vec, doc, meta

    # ------------------------------------------------------------------ build
    def build(
        self,
        root: str,
        progress_cb: ProgressCb = _noop,
        log_cb: LogCb = _noop,
        should_cancel: CancelCb = _never,
        thumb_cb: ThumbCb = _noop,
        full_rebuild: bool = False,
    ) -> int:
        if not root or not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root!r}")

        if full_rebuild:
            log_cb("기존 인덱스를 초기화합니다…")
            self.store.clear()

        # --- scan ---
        progress_cb(IndexProgress(0, 0, "scan", root))
        folders: list[tuple[str, list[str]]] = []
        for dirpath, images in walker.iter_leaf_folders(root, self.cfg.image_exts):
            folders.append((dirpath, images))
            if len(folders) % 50 == 0:
                progress_cb(IndexProgress(len(folders), 0, "scan", dirpath))
            if should_cancel():
                progress_cb(IndexProgress(0, 0, "cancelled", ""))
                return 0

        granularity = self.cfg.index_granularity
        if granularity == "image":
            units = [(d, img) for d, imgs in folders for img in imgs]
        else:
            units = list(folders)
        total = len(units)
        log_cb(f"{len(folders)}개 폴더 / {total}개 {'이미지' if granularity == 'image' else '폴더'} 발견.")

        # rebuild if the existing collection used a different embedder/dim/granularity
        if not full_rebuild:
            sig = self.store.stored_signature()
            cur = (getattr(self.embedder, "name", "?"), int(getattr(self.embedder, "dim", 0)))
            if sig is not None:
                base_mismatch = (sig[0], sig[1]) != cur
                # sig[2] is None for indexes that predate granularity recording
                gran_mismatch = sig[2] is not None and sig[2] != granularity
                if base_mismatch or gran_mismatch:
                    log_cb(f"색인 설정 변경 감지 {sig} → {cur + (granularity,)}. 전체 재빌드로 전환합니다.")
                    self.store.clear()
                    full_rebuild = True

        existing = {} if full_rebuild else self.store.existing_mtimes()

        ids: list[str] = []
        embs: list = []
        docs: list[str] = []
        metas: list[dict] = []
        n_written = 0

        def flush() -> None:
            nonlocal ids, embs, docs, metas, n_written
            if not ids:
                return
            self.store.upsert(ids, np.vstack(embs), docs, metas)
            n_written += len(ids)
            ids, embs, docs, metas = [], [], [], []

        for i, unit in enumerate(units):
            if should_cancel():
                flush()
                progress_cb(IndexProgress(i, total, "cancelled", ""))
                log_cb(f"취소됨 — {n_written}개 저장 완료.")
                return n_written

            if granularity == "image":
                dirpath, image_path = unit
                rid, mkey = image_id(image_path), _image_mkey(image_path)
                folder_label = os.path.basename(image_path)
            else:
                dirpath, images = unit
                rid, mkey = folder_id(dirpath), _mkey(dirpath)
                folder_label = os.path.basename(dirpath.rstrip("/\\"))

            if not full_rebuild and existing.get(rid) == mkey:
                progress_cb(IndexProgress(i + 1, total, "index", folder_label))
                continue

            try:
                if granularity == "image":
                    rid, vec, doc, meta = self._image_record(dirpath, image_path, mkey)
                else:
                    rid, vec, doc, meta = self._folder_record(dirpath, images, mkey)
                thumb_cb(meta["representative_image"])
            except Exception as e:
                log_cb(f"건너뜀 (오류) {folder_label}: {type(e).__name__}: {e}")
                progress_cb(IndexProgress(i + 1, total, "index", folder_label))
                continue

            ids.append(rid)
            embs.append(vec)
            docs.append(doc)
            metas.append(meta)
            if len(ids) >= _FLUSH_EVERY:
                flush()
            progress_cb(IndexProgress(i + 1, total, "index", folder_label))

        flush()
        progress_cb(IndexProgress(total, total, "done", ""))
        log_cb(f"인덱싱 완료 — 총 {self.store.count()}개 레코드가 색인되었습니다.")
        return n_written

    # ------------------------------------------------------------------ refresh
    def refresh_captions(
        self,
        root: str,
        progress_cb: ProgressCb = _noop,
        log_cb: LogCb = _noop,
        should_cancel: CancelCb = _never,
    ) -> int:
        """Recompute LABEL/SIDECAR captions and update store metadata WITHOUT
        re-embedding and WITHOUT any model inference. Records with no label/sidecar
        text are left untouched (their stored — possibly VLM-generated — captions
        must survive a name edit). Image granularity only.
        """
        if self.cfg.index_granularity != "image":
            raise ValueError("캡션 새로고침은 이미지 단위(image) 인덱스에서만 지원됩니다")
        if not root or not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root!r}")
        existing = set(self.store.all_ids())
        if not existing:
            log_cb("인덱스가 비어 있어 갱신할 항목이 없습니다.")
            return 0
        # keep the STORED embedding signature — this run's embedder may be a
        # lightweight mock, and we must not corrupt model_id/embed_dim/granularity
        sig = self.store.stored_signature()
        # keep the STORED mtime keys: the embedding reflects the file at BUILD time,
        # so re-stamping the current mtime here would mask a needed re-embed later
        prev_mtimes = self.store.existing_mtimes()

        progress_cb(IndexProgress(0, 0, "scan", root))
        units = [
            (d, img)
            for d, imgs in walker.iter_leaf_folders(root, self.cfg.image_exts)
            for img in imgs
        ]
        total = len(units)

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        n_updated = 0

        def flush() -> None:
            nonlocal ids, docs, metas, n_updated
            if not ids:
                return
            self.store.update_meta(ids, docs, metas)
            n_updated += len(ids)
            ids, docs, metas = [], [], []

        for i, (dirpath, image_path) in enumerate(units):
            if should_cancel():
                flush()
                progress_cb(IndexProgress(i, total, "cancelled", ""))
                log_cb(f"취소됨 — {n_updated}개 갱신 완료.")
                return n_updated
            rid = image_id(image_path)
            if rid in existing:
                try:
                    caption = self._label_caption(image_path)  # never calls a model
                except Exception as e:
                    log_cb(f"건너뜀 (오류) {os.path.basename(image_path)}: {e}")
                    progress_cb(IndexProgress(i + 1, total, "caption", ""))
                    continue
                if not caption:
                    # no label/sidecar text: leave the stored caption untouched
                    progress_cb(IndexProgress(i + 1, total, "caption", ""))
                    continue
                mkey = prev_mtimes.get(rid, _image_mkey(image_path))
                meta = self._base_meta(dirpath, image_path, 1, "image", mkey)
                meta.update(caption=caption, sidecar_text=caption[:400])
                if sig is not None:
                    meta["model_id"], meta["embed_dim"] = sig[0], int(sig[1])
                    if sig[2] is not None:
                        meta["granularity"] = sig[2]
                ids.append(rid)
                docs.append(caption)
                metas.append(meta)
                if len(ids) >= _FLUSH_EVERY:
                    flush()
            progress_cb(IndexProgress(i + 1, total, "caption", os.path.basename(image_path)))

        flush()
        progress_cb(IndexProgress(total, total, "done", ""))
        log_cb(f"캡션 갱신 완료 — {n_updated}개 레코드에 새 클래스 이름이 반영되었습니다.")
        return n_updated
