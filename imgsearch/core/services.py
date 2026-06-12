"""SearchService — the only object the UI talks to for querying."""

from __future__ import annotations

import os
from typing import Optional

from imgsearch.backends.base import ChatLLM, Embedder
from imgsearch.config import AppConfig
from imgsearch.core.models import FolderHit, QueryResult
from imgsearch.index import walker
from imgsearch.store.chroma_store import ChromaStore, folder_id


class SearchService:
    def __init__(
        self,
        embedder: Embedder,
        store: ChromaStore,
        chat: Optional[ChatLLM],
        config: AppConfig,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chat = chat
        self.cfg = config

    def query(self, text: str, k: Optional[int] = None) -> QueryResult:
        text = (text or "").strip()
        k = int(k or self.cfg.top_k)
        if not text:
            return QueryResult(text, "", [], "검색어를 입력하세요.")

        refined = text
        if self.chat is not None:
            try:
                r = self.chat.refine_query(text)
                if r and r.strip():
                    refined = r.strip()
            except Exception:
                refined = text

        qvec = self.embedder.embed_text([refined])[0]
        hits = self.store.query(qvec, k)

        summary = ""
        if self.cfg.summarize_enabled and self.chat is not None:
            try:
                summary = self.chat.summarize(text, hits)
            except Exception:
                summary = ""

        return QueryResult(query=text, refined=refined, hits=hits, summary=summary)

    def query_by_example(self, hit: FolderHit, k: Optional[int] = None) -> QueryResult:
        """Find records similar to an existing result, reusing its STORED embedding
        (no model call — works instantly, including in mock mode). Falls back to
        embedding the image only if the record vanished from the store."""
        from imgsearch.index.indexer import image_id  # local: avoids cycle at import time

        k = int(k or self.cfg.top_k)
        if self.cfg.index_granularity == "image":
            rid = image_id(hit.image_path)
        else:
            rid = folder_id(hit.folder)
        vec = self.store.get_embedding(rid)
        if vec is None:
            vec = self.embedder.embed_image([hit.image_path])[0]
        hits = self.store.query(vec, k + 1)
        hits = [h for h in hits if h.image_path != hit.image_path][:k]
        label = os.path.basename(hit.image_path or hit.folder)
        return QueryResult(query=f"유사 이미지: {label}", refined="", hits=hits, summary="")

    def member_images(self, folder: str) -> list[str]:
        """All images in a leaf folder (for the viewer's prev/next navigation)."""
        return walker.list_images(folder, self.cfg.image_exts)

    def count(self) -> int:
        return self.store.count()
