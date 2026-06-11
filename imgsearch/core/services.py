"""SearchService — the only object the UI talks to for querying."""

from __future__ import annotations

from typing import Optional

from imgsearch.backends.base import ChatLLM, Embedder
from imgsearch.config import AppConfig
from imgsearch.core.models import QueryResult
from imgsearch.index import walker
from imgsearch.store.chroma_store import ChromaStore


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

    def member_images(self, folder: str) -> list[str]:
        """All images in a leaf folder (for the viewer's prev/next navigation)."""
        return walker.list_images(folder, self.cfg.image_exts)

    def count(self) -> int:
        return self.store.count()
