"""ChromaDB persistence: one cosine-space record per leaf folder.

We always supply our own (CLIP/mock) embeddings, so no Chroma embedding function is
ever attached. IDs are stable hashes of the leaf-folder path → re-indexing is an
idempotent ``upsert``.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from imgsearch.core.models import FolderHit

COLLECTION = "representatives"
_UPSERT_BATCH = 256


def folder_id(folder: str) -> str:
    norm = str(Path(folder)).replace("\\", "/").rstrip("/").lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


class ChromaStore:
    def __init__(self, persist_dir: str | Path, collection: str = COLLECTION) -> None:
        import chromadb
        from chromadb.config import Settings

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._name = collection
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    # --- writes ---
    def upsert(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray | Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if len(ids) == 0:
            return
        if isinstance(embeddings, np.ndarray):
            emb_list = embeddings.astype(np.float32).tolist()
        else:
            emb_list = [list(map(float, e)) for e in embeddings]
        for i in range(0, len(ids), _UPSERT_BATCH):
            j = i + _UPSERT_BATCH
            self._col.upsert(
                ids=list(ids[i:j]),
                embeddings=emb_list[i:j],
                documents=list(documents[i:j]),
                metadatas=list(metadatas[i:j]),
            )

    def update_meta(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        """Replace documents+metadata of EXISTING records, keeping embeddings.

        The stored embeddings are read and passed back explicitly: updating
        ``documents`` without ``embeddings`` makes Chroma RE-EMBED the text with
        its default embedding function, corrupting our CLIP vector space.
        """
        if len(ids) == 0:
            return
        for i in range(0, len(ids), _UPSERT_BATCH):
            j = i + _UPSERT_BATCH
            batch_ids = list(ids[i:j])
            got = self._col.get(ids=batch_ids, include=["embeddings"])
            emb_by_id = {
                gid: emb for gid, emb in zip(got.get("ids", []), got.get("embeddings", []))
            }
            keep = [k for k, gid in enumerate(batch_ids) if gid in emb_by_id]
            if not keep:
                continue
            self._col.update(
                ids=[batch_ids[k] for k in keep],
                embeddings=[np.asarray(emb_by_id[batch_ids[k]], dtype=np.float32).tolist() for k in keep],
                documents=[documents[i + k] for k in keep],
                metadatas=[metadatas[i + k] for k in keep],
            )

    def delete(self, ids: Sequence[str]) -> None:
        """Remove records by id (e.g. files that vanished from disk)."""
        ids = list(ids)
        for i in range(0, len(ids), _UPSERT_BATCH):
            self._col.delete(ids=ids[i : i + _UPSERT_BATCH])

    def clear(self) -> None:
        """Drop and recreate the collection (full rebuild)."""
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    # --- reads ---
    def count(self) -> int:
        try:
            return int(self._col.count())
        except Exception:
            return 0

    def all_ids(self) -> list[str]:
        try:
            return list(self._col.get(include=[]).get("ids", []))
        except Exception:
            return []

    def get_embedding(self, record_id: str) -> Optional[np.ndarray]:
        """The stored embedding for one record, or None if absent (used by
        query-by-example so no model call is needed)."""
        try:
            got = self._col.get(ids=[record_id], include=["embeddings"])
        except Exception:
            return None
        embs = got.get("embeddings")
        if embs is None or len(embs) == 0:
            return None
        return np.asarray(embs[0], dtype=np.float32)

    _SCAN_PAGE = 5000

    def existing_mtimes(self) -> dict[str, str]:
        """Map id -> stored folder mtime key (string), for incremental skipping.

        Stored as a fixed-precision string so equality is exact across the
        Chroma round-trip (float mtimes sit at float64's precision edge).
        Paged so memory stays bounded on large per-image indexes.
        """
        out: dict[str, str] = {}
        offset = 0
        while True:
            try:
                got = self._col.get(
                    include=["metadatas"], limit=self._SCAN_PAGE, offset=offset
                )
            except Exception:
                return out
            ids = got.get("ids", [])
            for _id, md in zip(ids, got.get("metadatas", []) or []):
                if md and md.get("mtime") is not None:
                    out[_id] = str(md["mtime"])
            if len(ids) < self._SCAN_PAGE:
                return out
            offset += len(ids)

    def stored_signature(self) -> Optional[tuple[str, int, Optional[str]]]:
        """(model_id, embed_dim, granularity) of an existing record, for mismatch
        detection. granularity is None for indexes built before it was recorded."""
        try:
            got = self._col.get(limit=1, include=["metadatas"])
        except Exception:
            return None
        metas = got.get("metadatas") or []
        if metas and metas[0] and "model_id" in metas[0] and "embed_dim" in metas[0]:
            try:
                gran = metas[0].get("granularity")
                return (
                    str(metas[0]["model_id"]),
                    int(metas[0]["embed_dim"]),
                    str(gran) if gran is not None else None,
                )
            except (TypeError, ValueError):
                return None
        return None

    def query(self, embedding: np.ndarray, k: int) -> list[FolderHit]:
        if self.count() == 0:
            return []
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1).tolist()
        res = self._col.query(
            query_embeddings=[vec],
            n_results=max(1, int(k)),
            include=["metadatas", "documents", "distances"],
        )
        metas = (res.get("metadatas") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits: list[FolderHit] = []
        for md, doc, dist in zip(metas, docs, dists):
            md = md or {}
            sim = 1.0 - float(dist)  # cosine distance -> similarity
            score = max(0.0, sim) if math.isfinite(sim) else 0.0
            hits.append(
                FolderHit(
                    folder=str(md.get("leaf_folder", "")),
                    image_path=str(md.get("representative_image", "")),
                    caption=str(md.get("caption", "") or doc or ""),
                    score=score,
                    member_count=int(md.get("member_count", 0) or 0),
                    sidecar_text=str(md.get("sidecar_text", "")),
                )
            )
        return hits
