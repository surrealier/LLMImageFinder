"""오래 걸리는 인덱싱 파이프라인. Qt 비의존(Qt-agnostic): 진행률/로그/취소가 모두
평범한 콜러블(callable)이라, UI 없이 헤드리스로 단위 테스트할 수 있고 UI에서는
QThread 워커로 감싸 사용한다.

두 가지 입자도(granularity, ``cfg.index_granularity``):
  * "folder": 리프 폴더마다 대표 이미지 1장(거의 중복인 장면 폴더에 적합).
  * "image":  모든 이미지를 각각 하나의 레코드로(다양하고 개별 라벨이 있는 데이터셋용).
              가능하면 YOLO 라벨(class-id -> 이름)에서 캡션을 만든다.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from imgsearch.backends.base import Captioner, Embedder
from imgsearch.config import AppConfig
from imgsearch.core.models import BuildReport, IndexProgress
from imgsearch.index import labels, pairing, repr_select, walker
from imgsearch.store.chroma_store import ChromaStore, folder_id

# 콜백 타입 별칭들. 함수 시그니처를 읽기 쉽게 하고, UI/테스트가 자유롭게 주입할 수 있게 한다.
ProgressCb = Callable[[IndexProgress], None]  # 진행 상황 보고
LogCb = Callable[[str], None]                 # 사람이 읽는 로그 한 줄
CancelCb = Callable[[], bool]                 # True 반환 시 취소 요청
ThumbCb = Callable[[str], None]               # 썸네일 선생성 힌트(대표 이미지 경로 전달)

# 스토어에 한 번에 모아서 기록(upsert)할 레코드 개수. 너무 잦은 쓰기를 피해 처리량을 높인다.
_FLUSH_EVERY = 128
# embed_image() 한 번 호출 전에 모을 이미지 수 — 실제 백엔드(jina-clip)는 이 묶음을
# 이미지별 호출이 아니라 단일 배치 GPU forward 로 처리해 훨씬 빠르다.
_EMBED_BATCH = 32


def _noop(*_a, **_k) -> None:
    """기본 콜백용 무동작 함수. 어떤 인자가 와도 무시하고 None 을 반환한다."""
    return None


def _never() -> bool:
    """기본 취소 콜백. 항상 False(=취소되지 않음)를 반환한다."""
    return False


def image_id(path: str) -> str:
    """이미지 경로로부터 안정적인 레코드 id(SHA1 hex)를 만든다.

    경로를 정규화한 뒤 해싱하므로, 같은 파일이면 OS/구분자/대소문자가 달라도 같은 id 가 나온다
    (증분 빌드에서 "이미 처리한 레코드"를 정확히 매칭하기 위함).
    """
    # 역슬래시를 슬래시로, 대문자를 소문자로 통일해 Windows/Unix 경로 표기 차이를 흡수한다.
    norm = str(Path(path)).replace("\\", "/").lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _mkey(path: str) -> str:
    """파일 수정 시각(mtime)을 소수점 3자리 문자열로 만든 "변경 감지 키".

    증분 빌드에서 저장된 키와 비교해 파일이 바뀌었는지 판단한다. 파일이 없으면 "0".
    """
    try:
        return f"{os.path.getmtime(path):.3f}"
    except OSError:
        return "0"


def _image_mkey(image_path: str) -> str:
    """이미지 단위 레코드의 변경 감지 키: 이미지 mtime + 라벨 mtime 을 결합한다.

    이렇게 두 mtime 을 합치면 YOLO 라벨 파일만 수정(재어노테이션)해도 키가 달라져
    해당 레코드가 무효화되고 다음 빌드에서 다시 처리된다.
    """
    m = _mkey(image_path)
    lp = labels.label_path_for(image_path)
    # 라벨이 있으면 "이미지키|라벨키", 없으면 이미지 키만 사용.
    return f"{m}|{_mkey(str(lp))}" if lp is not None else m


class Indexer:
    """임베더/캡셔너/스토어/설정을 묶어 인덱싱 파이프라인을 수행하는 클래스."""

    def __init__(
        self,
        embedder: Embedder,
        captioner: Optional[Captioner],
        store: ChromaStore,
        config: AppConfig,
    ) -> None:
        """의존성 주입: 임베더는 필수, 캡셔너는 선택(VLM 미사용 시 None)이다."""
        self.embedder = embedder      # 이미지 -> 벡터 임베딩
        self.captioner = captioner    # 이미지 -> 자연어 캡션(선택)
        self.store = store            # 벡터/메타 저장소(ChromaDB)
        self.cfg = config             # 앱 설정(입자도/확장자/클래스 이름 등)

    # ------------------------------------------------------------------ records
    def _embed_checked(self, image_path: str) -> np.ndarray:
        """이미지 1장을 임베딩하고 유한성(finite)을 검증한 벡터를 반환한다.

        NaN/Inf 가 섞인 벡터는 검색 시 거리 계산을 오염시키므로, 여기서 즉시 예외로
        걸러 호출부가 그 이미지를 건너뛰게 한다.
        """
        vec = self.embedder.embed_image([image_path])[0]
        if not np.all(np.isfinite(vec)):
            raise ValueError("non-finite embedding vector")
        return vec

    def _base_meta(self, folder: str, image_path: str, members: int, method: str, mkey: str) -> dict:
        """레코드에 공통으로 붙는 메타데이터 딕셔너리를 만든다.

        model_id/embed_dim/granularity 는 나중에 "인덱스 설정이 바뀌었는지"를 판별하는
        시그니처(signature)로 쓰이므로, 모든 레코드에 일관되게 기록해야 한다.
        """
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
        """폴더 단위 레코드 하나를 만든다: (id, 벡터, 문서텍스트, 메타).

        대표 이미지를 고른 뒤 그 1장만 임베딩한다(폴더 전체가 아니라). 캡션은 설정이
        켜져 있고 캡셔너가 있을 때만 생성하며, 사이드카 텍스트를 문맥으로 넘겨 품질을 높인다.
        """
        sidecar = pairing.folder_sidecar_text(dirpath, images)
        rep = repr_select.choose_representative(images, self.embedder, self.cfg.max_members_for_repr)
        caption = ""
        if self.cfg.caption_enabled and self.captioner is not None:
            caption = self.captioner.caption(rep.image_path, context=sidecar)
        vec = self._embed_checked(rep.image_path)
        # 검색용 문서 텍스트: 캡션+사이드카. 둘 다 비면 경로라도 넣어 빈 문서를 피한다.
        doc = (caption + "\n" + sidecar).strip() or rep.image_path
        meta = self._base_meta(dirpath, rep.image_path, rep.member_count, rep.method, mkey)
        # 사이드카 원문은 400자까지만 메타에 보관(저장 비용/메타 비대화 방지).
        meta.update(caption=caption, sidecar_text=sidecar[:400])
        return folder_id(dirpath), vec, doc, meta

    def _label_caption(self, image_path: str) -> str:
        """오직 라벨/사이드카 텍스트 파일에서만 캡션을 끌어낸다(모델 호출 없음).

        먼저 YOLO 라벨을 캡션으로 바꿔 보고, YOLO 가 아니거나 비면 일반 텍스트 사이드카를
        쓴다. 모델을 부르지 않으므로 클래스 이름 편집 후 "캡션만 새로고침"하는 데 쓰인다.
        """
        lp = labels.label_path_for(image_path)
        if lp is None:
            return ""
        caption = labels.yolo_caption(lp, self.cfg.class_names)  # YOLO 가 아니면 ""
        if not caption:
            caption = pairing.sidecar_for(image_path)  # 실제 텍스트 사이드카(YOLO 라벨은 건너뜀)
        return caption

    def _image_caption(self, image_path: str) -> str:
        """이미지 단위 레코드의 캡션을 결정한다.

        1순위는 라벨/사이드카 텍스트(_label_caption)이며, 그게 비고 VLM 캡션이 켜져 있을
        때만 모델을 호출한다(비싼 모델 호출을 최소화하는 폴백 순서).
        """
        caption = self._label_caption(image_path)
        if not caption and self.cfg.caption_enabled and self.captioner is not None:
            caption = self.captioner.caption(image_path, "")
        return caption

    def _image_record(self, dirpath: str, image_path: str, mkey: str):
        """이미지 단위 레코드 하나를 만든다: (id, 벡터, 문서텍스트, 메타).

        주: build() 의 메인 루프는 배치 임베딩 경로(embed_pending)를 쓰므로 이 헬퍼를
        직접 호출하지 않는다. 단건 생성이 필요한 경우를 위한 보조 메서드다.
        """
        caption = self._image_caption(image_path)
        vec = self._embed_checked(image_path)
        # 캡션이 없으면 파일명이라도 문서로 써서 어휘 검색의 단서를 남긴다.
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
    ) -> BuildReport:
        """데이터셋 ``root`` 를 인덱싱한다. 증분(기본) 또는 전체 재빌드.

        진행률/로그/취소/썸네일은 콜백으로 주입한다(UI/테스트 양쪽에서 재사용 가능).
        취소가 들어오면 그때까지 모은 배치를 flush 하고 BuildReport(cancelled=True)를
        반환한다. 처리/건너뜀/정리 통계가 담긴 BuildReport 를 돌려준다.
        """
        if not root or not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root!r}")

        if full_rebuild:
            # 전체 재빌드 요청 시 기존 컬렉션을 완전히 비우고 처음부터 다시 채운다.
            log_cb("Clearing the existing index…")
            self.store.clear()

        # --- 스캔 단계: 디스크를 훑어 작업 대상 폴더 목록을 먼저 확정한다 ---
        progress_cb(IndexProgress(0, 0, "scan", root))
        folders: list[tuple[str, list[str]]] = []
        for dirpath, images in walker.iter_leaf_folders(root, self.cfg.image_exts):
            folders.append((dirpath, images))
            # 50개마다 진행률을 보고해 스캔이 길어도 UI가 멈춰 보이지 않게 한다.
            if len(folders) % 50 == 0:
                progress_cb(IndexProgress(len(folders), 0, "scan", dirpath))
            if should_cancel():
                # 스캔 도중 취소: 아무것도 기록하지 않았으므로 빈 취소 리포트로 즉시 종료.
                progress_cb(IndexProgress(0, 0, "cancelled", ""))
                return BuildReport(cancelled=True)

        granularity = self.cfg.index_granularity
        if granularity == "image":
            # 이미지 단위: (폴더, 이미지) 쌍을 펼쳐 이미지 하나하나를 작업 단위로 삼는다.
            units = [(d, img) for d, imgs in folders for img in imgs]
        else:
            # 폴더 단위: 폴더 자체가 작업 단위.
            units = list(folders)
        total = len(units)
        report = BuildReport(total=total)
        log_cb(f"Found {len(folders)} folders / {total} {'images' if granularity == 'image' else 'folders'}.")

        # 기존 컬렉션이 다른 임베더/차원/입자도로 만들어졌다면 호환되지 않으므로 전체 재빌드로 전환.
        if not full_rebuild:
            sig = self.store.stored_signature()  # (model_id, embed_dim, granularity) 또는 None
            cur = (getattr(self.embedder, "name", "?"), int(getattr(self.embedder, "dim", 0)))
            if sig is not None:
                base_mismatch = (sig[0], sig[1]) != cur  # 모델/차원이 다르면 벡터 공간이 호환 안 됨
                # sig[2] 가 None 인 경우는 입자도 기록 기능 이전에 만들어진 구버전 인덱스다(불일치로 보지 않음).
                gran_mismatch = sig[2] is not None and sig[2] != granularity
                if base_mismatch or gran_mismatch:
                    log_cb(f"Index settings changed {sig} → {cur + (granularity,)}. Switching to a full rebuild.")
                    self.store.clear()
                    full_rebuild = True

        # 전체 재빌드면 기존 mtime 맵이 필요 없다. 증분이면 "id -> 저장된 mtime키"를 받아 와 비교에 쓴다.
        existing = {} if full_rebuild else self.store.existing_mtimes()

        # 디스크에서 사라진 파일에 대응하는 레코드를 제거(prune)한다. 그래야 증분 빌드가
        # 진짜 "동기화"가 되어, 결과 화면에 죽은 썸네일이나 깨진 '폴더 열기'가 남지 않는다.
        if existing:
            # 현재 디스크에 실제 존재하는 작업 단위들의 id 집합을 만든다.
            if granularity == "image":
                current_ids = {image_id(img) for _d, img in units}
            else:
                current_ids = {folder_id(d) for d, _imgs in units}
            # 저장돼 있지만 현재 id 집합에 없는 = 파일이 사라진 레코드들.
            stale = [rid for rid in existing if rid not in current_ids]
            if stale:
                self.store.delete(stale)
                for rid in stale:
                    existing.pop(rid, None)  # 아래 증분 비교에서도 빠지도록 맵에서 제거.
                report.pruned = len(stale)
                log_cb(f"Pruned {len(stale)} records whose files no longer exist on disk.")

        # 스토어에 일괄 기록(upsert)하기 전 누적 버퍼들(병렬 리스트: 같은 인덱스가 한 레코드).
        ids: list[str] = []
        embs: list = []
        docs: list[str] = []
        metas: list[dict] = []

        def flush() -> None:
            """누적 버퍼를 스토어에 한 번에 upsert 하고 버퍼를 비운다(아무것도 없으면 무동작).

            nonlocal 로 바깥 리스트를 새 빈 리스트로 교체한다(in-place clear 가 아니라 재바인딩).
            """
            nonlocal ids, embs, docs, metas
            if not ids:
                return
            # 벡터들을 하나의 2차원 배열로 쌓아 한 번에 기록(스토어 호출 횟수 최소화).
            self.store.upsert(ids, np.vstack(embs), docs, metas)
            report.n_written += len(ids)
            ids, embs, docs, metas = [], [], [], []

        def skip(path: str, err: str) -> None:
            """한 항목 처리 실패를 리포트에 기록하고 로그로 알린다(파이프라인은 계속 진행)."""
            report.skipped.append((path, err))
            log_cb(f"Skipped (error) {os.path.basename(path)}: {err}")

        # 이미지 입자도에서는 임베딩 호출을 배치로 묶는다(배치당 GPU forward 1회).
        # pending 의 각 행은 "벡터를 제외한 모든 정보"를 담아 두고, 벡터만 나중에 일괄 계산한다.
        pending: list[tuple[str, str, str, str, str]] = []  # rid, dir, path, mkey, caption

        def embed_pending() -> None:
            """pending 에 모인 이미지들을 한 번의 배치로 임베딩해 출력 버퍼로 옮긴다.

            배치 임베딩이 실패하거나 일부 벡터가 비정상(NaN/Inf)이면 그 항목만 단건으로
            재시도하고, 그래도 실패하면 건너뛴다(한 장의 손상 때문에 배치 전체를 잃지 않게).
            """
            nonlocal pending
            if not pending:
                return
            paths = [t[2] for t in pending]  # 각 행의 인덱스 2 == image_path
            vecs = None
            try:
                got = self.embedder.embed_image(paths)
                # 반환 개수가 입력과 정확히 일치할 때만 배치 결과를 신뢰한다.
                if got is not None and len(got) == len(paths):
                    vecs = np.asarray(got, dtype=np.float32)
            except Exception:
                vecs = None  # 배치 실패 시 아래에서 이미지별로 재시도하도록 둔다.
            for row, (rid, dirpath, image_path, mkey, caption) in enumerate(pending):
                vec = None
                # 배치 벡터가 있고 유한하면 그대로 사용.
                if vecs is not None and np.all(np.isfinite(vecs[row])):
                    vec = vecs[row]
                if vec is None:
                    # 배치 누락/비정상 -> 단건 임베딩으로 복구 시도(검증 포함).
                    try:
                        vec = self._embed_checked(image_path)
                    except Exception as e:
                        skip(image_path, f"{type(e).__name__}: {e}")
                        continue
                doc = caption or os.path.basename(image_path)
                meta = self._base_meta(dirpath, image_path, 1, "image", mkey)
                meta.update(caption=caption, sidecar_text=caption[:400])
                ids.append(rid)
                embs.append(vec)
                docs.append(doc)
                metas.append(meta)
                thumb_cb(image_path)  # 결과 표시용 썸네일을 미리 만들도록 힌트를 준다.
                if len(ids) >= _FLUSH_EVERY:
                    flush()
            pending = []

        # --- 메인 처리 루프: 작업 단위(units)를 하나씩 처리한다 ---
        for i, unit in enumerate(units):
            if should_cancel():
                # 아직 임베딩되지 않은 pending 배치는 버린다: 그 레코드들의 mtime 키가
                # 아직 스토어에 찍히지 않았으므로, 다음 증분 빌드가 이들을 다시 집어 든다.
                flush()  # 이미 임베딩이 끝나 버퍼에 들어간 것만 저장하고 종료.
                report.cancelled = True
                progress_cb(IndexProgress(i, total, "cancelled", ""))
                log_cb(f"Cancelled — {report.n_written} records saved.")
                return report

            # 입자도에 따라 작업 단위를 풀어 레코드 id/변경키/표시 라벨을 준비한다.
            if granularity == "image":
                dirpath, image_path = unit
                rid, mkey = image_id(image_path), _image_mkey(image_path)
                folder_label = os.path.basename(image_path)
            else:
                dirpath, images = unit
                rid, mkey = folder_id(dirpath), _mkey(dirpath)
                # rstrip 으로 끝의 구분자를 떼어 폴더명만 깔끔히 뽑는다(예: ".../cats/" -> "cats").
                folder_label = os.path.basename(dirpath.rstrip("/\\"))

            # 증분 빌드에서 저장된 변경키와 현재 키가 같으면 = 안 바뀐 레코드이므로 건너뛴다.
            if not full_rebuild and existing.get(rid) == mkey:
                progress_cb(IndexProgress(i + 1, total, "index", folder_label))
                continue

            if granularity == "image":
                # 캡션 추출은 단건마다(모델 호출이 포함될 수 있어 실패를 개별 격리).
                try:
                    caption = self._image_caption(image_path)
                except Exception as e:
                    skip(image_path, f"{type(e).__name__}: {e}")
                    progress_cb(IndexProgress(i + 1, total, "index", folder_label))
                    continue
                # 임베딩은 즉시 하지 않고 pending 에 모았다가 배치로 처리(성능 최적화).
                pending.append((rid, dirpath, image_path, mkey, caption))
                if len(pending) >= _EMBED_BATCH:
                    embed_pending()
            else:
                # 폴더 단위는 대표 1장만 다루므로 배치 이득이 작아 단건으로 레코드를 만든다.
                try:
                    rid, vec, doc, meta = self._folder_record(dirpath, images, mkey)
                    thumb_cb(meta["representative_image"])
                except Exception as e:
                    skip(dirpath, f"{type(e).__name__}: {e}")
                    progress_cb(IndexProgress(i + 1, total, "index", folder_label))
                    continue
                ids.append(rid)
                embs.append(vec)
                docs.append(doc)
                metas.append(meta)
                if len(ids) >= _FLUSH_EVERY:
                    flush()
            progress_cb(IndexProgress(i + 1, total, "index", folder_label))

        # 루프 종료 후 남은 잔여분을 마저 처리한다(부분 배치/부분 버퍼).
        embed_pending()
        flush()
        progress_cb(IndexProgress(total, total, "done", ""))
        log_cb(f"Indexing complete — {self.store.count()} records indexed in total.")
        return report

    # ------------------------------------------------------------------ refresh
    def refresh_captions(
        self,
        root: str,
        progress_cb: ProgressCb = _noop,
        log_cb: LogCb = _noop,
        should_cancel: CancelCb = _never,
    ) -> int:
        """라벨/사이드카 캡션만 다시 계산해 스토어 메타데이터를 갱신한다.

        재임베딩도, 어떠한 모델 추론도 하지 않는다(클래스 이름만 바꿨을 때 빠르게
        반영하기 위함). 라벨/사이드카 텍스트가 없는 레코드는 손대지 않는다 — 그렇게
        해야 저장돼 있던(어쩌면 VLM 이 생성한) 캡션이 이름 편집에도 살아남는다.
        이미지 단위 인덱스에서만 지원한다.
        """
        if self.cfg.index_granularity != "image":
            raise ValueError("Caption refresh is only supported for image-granularity indexes")
        if not root or not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root!r}")
        # 저장된 mtime 키를 "그대로 유지"한다: 임베딩은 빌드 시점의 파일을 반영하므로,
        # 여기서 현재 mtime 으로 다시 찍으면 나중에 필요한 재임베딩을 가려 버린다(놓치게 됨).
        # 이 맵의 키들이 곧 저장된 레코드 id 들이므로, 한 번의 조회로 "어떤 id 가 있는지"와
        # "각 id 의 mtime 키"를 동시에 얻는다.
        prev_mtimes = self.store.existing_mtimes()
        existing = set(prev_mtimes)
        if not existing:
            log_cb("Index is empty — nothing to refresh.")
            return 0
        # 저장된 임베딩 시그니처를 보존한다 — 이번 실행의 임베더가 가벼운 mock 일 수 있으므로,
        # model_id/embed_dim/granularity 를 그 mock 값으로 덮어써 인덱스를 오염시키면 안 된다.
        sig = self.store.stored_signature()

        progress_cb(IndexProgress(0, 0, "scan", root))
        # 갱신은 이미지 단위 전용이므로 (폴더, 이미지) 쌍을 곧장 펼쳐 둔다.
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
            """누적된 메타 갱신을 스토어에 일괄 반영한다(벡터는 건드리지 않는 update_meta)."""
            nonlocal ids, docs, metas, n_updated
            if not ids:
                return
            self.store.update_meta(ids, docs, metas)
            n_updated += len(ids)
            ids, docs, metas = [], [], []

        for i, (dirpath, image_path) in enumerate(units):
            if should_cancel():
                flush()  # 메타 갱신은 임베딩과 무관하므로, 모은 것만 안전하게 반영하고 종료.
                progress_cb(IndexProgress(i, total, "cancelled", ""))
                log_cb(f"Cancelled — {n_updated} records refreshed.")
                return n_updated
            rid = image_id(image_path)
            # 인덱스에 이미 있는 레코드만 대상으로 한다(여기서 새 레코드를 추가하지는 않음).
            if rid in existing:
                try:
                    caption = self._label_caption(image_path)  # 모델을 절대 호출하지 않음
                except Exception as e:
                    log_cb(f"Skipped (error) {os.path.basename(image_path)}: {e}")
                    progress_cb(IndexProgress(i + 1, total, "caption", ""))
                    continue
                if not caption:
                    # 라벨/사이드카 텍스트가 없으면 저장된 캡션을 건드리지 않고 그대로 둔다
                    # (VLM 이 만든 캡션을 이름 편집으로 날려 버리지 않기 위함).
                    progress_cb(IndexProgress(i + 1, total, "caption", ""))
                    continue
                # 저장된 mtime 키를 재사용한다(없으면 현재 키로 폴백). 재임베딩이 아니므로
                # 빌드 시점 키를 유지해야 다음 증분 빌드의 변경 판단이 어긋나지 않는다.
                mkey = prev_mtimes.get(rid, _image_mkey(image_path))
                meta = self._base_meta(dirpath, image_path, 1, "image", mkey)
                meta.update(caption=caption, sidecar_text=caption[:400])
                if sig is not None:
                    # 임베딩 시그니처를 저장된 값으로 강제 고정 — mock 임베더 값으로 덮어쓰지 않게.
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
        log_cb(f"Caption refresh complete — new class names applied to {n_updated} records.")
        return n_updated
