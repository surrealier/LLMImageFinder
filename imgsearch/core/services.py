"""SearchService — the only object the UI talks to for querying.

Three retrieval modes (``cfg.search_mode``):
  * "vector"  — CLIP cosine only (the v0.1/v0.2 behavior).
  * "keyword" — BM25 over the stored captions only.
  * "hybrid"  — both, fused by Reciprocal Rank Fusion (ordering); each hit keeps its
                true cosine in ``score`` so the score-threshold filter stays meaningful.
The BM25 (lexical) index is built lazily from the store's documents and rebuilt only
when the store's write-revision changes (so it never serves pruned/stale ids).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from imgsearch.backends.base import ChatLLM, Embedder
from imgsearch.config import AppConfig
from imgsearch.core.models import FolderHit, QueryResult
from imgsearch.index import walker
from imgsearch.index.lexical import LexicalIndex
from imgsearch.store.chroma_store import ChromaStore, folder_id

_RRF_K = 60  # reciprocal-rank-fusion damping constant (standard default)


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
        self._lexical: LexicalIndex | None = None
        self._lex_sig: tuple | None = None  # store.revision() the index was built at

    # ------------------------------------------------------------------ public
    def query(self, text: str, k: Optional[int] = None) -> QueryResult:
        text = (text or "").strip()
        k = int(k or self.cfg.top_k)
        if not text:
            return QueryResult(text, "", [], "검색어를 입력하세요.")

        refined = self._refine(text)
        mode = (self.cfg.search_mode or "vector").lower()
        if mode not in ("vector", "hybrid", "keyword"):
            mode = "vector"
        hits = self._retrieve(text, refined, k, mode)
        summary = self._summarize(text, hits)
        return QueryResult(query=text, refined=refined, hits=hits, summary=summary)

    def query_by_example(self, hit: FolderHit, k: Optional[int] = None) -> QueryResult:
        """Find records similar to an existing result, reusing its STORED embedding
        (no model call in the common path). Vector-only by design."""
        from imgsearch.index.indexer import image_id

        k = int(k or self.cfg.top_k)
        rid = image_id(hit.image_path) if self.cfg.index_granularity == "image" else folder_id(hit.folder)
        vec = self.store.get_embedding(rid)
        if vec is None:
            # the stored vector is gone (e.g. granularity changed without a rebuild).
            # Do NOT embed here — this may run on the GUI thread; return empty instead.
            return QueryResult(query=f"유사 이미지: {os.path.basename(hit.image_path or hit.folder)}",
                               refined="", hits=[],
                               summary="저장된 임베딩을 찾지 못했습니다 — 전체 재빌드 후 다시 시도하세요.")
        hits = self.store.query(vec, k + 1)
        hits = [h for h in hits if h.image_path != hit.image_path][:k]
        for h in hits:
            h.match = "vector"
        label = os.path.basename(hit.image_path or hit.folder)
        return QueryResult(query=f"유사 이미지: {label}", refined="", hits=hits, summary="")

    def search(self, text: str, k: Optional[int] = None, mode: Optional[str] = None,
               refine: bool = True) -> list[FolderHit]:
        """Retrieve hits only (no summary). Used by the agentic pipeline, which has
        already produced its own search text and so passes refine=False."""
        text = (text or "").strip()
        if not text:
            return []
        k = int(k or self.cfg.top_k)
        mode = (mode or self.cfg.search_mode or "vector").lower()
        if mode not in ("vector", "hybrid", "keyword"):
            mode = "vector"
        refined = self._refine(text) if refine else text
        return self._retrieve(text, refined, k, mode)

    def member_images(self, folder: str) -> list[str]:
        return walker.list_images(folder, self.cfg.image_exts)

    def count(self) -> int:
        return self.store.count()

    def invalidate_lexical(self) -> None:
        """Drop the cached BM25 index so the next keyword/hybrid query rebuilds it
        from the current store documents (call after an index build/refresh/prune)."""
        self._lexical = None
        self._lex_sig = None

    # ------------------------------------------------------------------ steps
    def _refine(self, text: str) -> str:
        refined = text
        if self.chat is not None:
            try:
                r = self.chat.refine_query(text)
                if r and r.strip():
                    refined = r.strip()
            except Exception:
                refined = text
        return refined

    def _summarize(self, query: str, hits: list[FolderHit]) -> str:
        if not (self.cfg.summarize_enabled and self.chat is not None):
            return ""
        try:
            return self.chat.summarize(query, hits)
        except Exception:
            return ""

    def _ensure_lexical(self) -> LexicalIndex:
        sig = self.store.revision()
        if self._lexical is None or self._lex_sig != sig:
            idx = LexicalIndex()
            idx.build(self.store.all_documents())
            self._lexical = idx
            self._lex_sig = sig
        return self._lexical

    def _embed_query(self, refined: str) -> np.ndarray:
        return np.asarray(self.embedder.embed_text([refined])[0], dtype=np.float32)

    @staticmethod
    def _rrf(rank_lists: list[list[str]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ids in rank_lists:
            for rank, rid in enumerate(ids):
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        return scores

    def _retrieve(self, text: str, refined: str, k: int, mode: str) -> list[FolderHit]:
        if mode == "vector":
            hits = self.store.query(self._embed_query(refined), k)
            for h in hits:
                h.match = "vector"
            return hits

        fan = max(k * 3, 50)
        lexical = self._ensure_lexical()
        lex_pairs = lexical.search(refined, fan)  # [(id, bm25)], desc
        lex_ids = [rid for rid, _ in lex_pairs]

        qvec: np.ndarray | None = None
        cos: dict[str, float] = {}
        vec_ids: list[str] = []
        if mode == "hybrid":
            qvec = self._embed_query(refined)
            vec_pairs = self.store.search_vector(qvec, fan)  # [(id, cosine)]
            cos = {rid: s for rid, s in vec_pairs}
            vec_ids = [rid for rid, _ in vec_pairs]
            fused = self._rrf([vec_ids, lex_ids])
        else:  # keyword
            fused = {rid: 1.0 / (_RRF_K + i + 1) for i, rid in enumerate(lex_ids)}

        ranked = sorted(fused, key=lambda r: (-fused[r], r))[:k]
        hit_map = self.store.fetch(ranked)  # drops ids no longer in the store
        # keyword-only ranked ids need a cosine for the score badge/threshold —
        # batch-fetch their embeddings in one round-trip (not one get() per hit)
        need_cos = [rid for rid in ranked if rid in hit_map and rid not in cos]
        emb_map: dict = {}
        if need_cos:
            if qvec is None:
                qvec = self._embed_query(refined)
            emb_map = self.store.get_embeddings(need_cos)

        vec_set, lex_set = set(vec_ids), set(lex_ids)
        out: list[FolderHit] = []
        for rid in ranked:
            h = hit_map.get(rid)
            if h is None:  # vanished between fusion and fetch
                continue
            if rid in cos:
                h.score = float(cos[rid])
            else:
                emb = emb_map.get(rid)
                # vectors are L2-normalized, so dot product == cosine similarity
                h.score = float(max(0.0, np.dot(emb, qvec))) if emb is not None and qvec is not None else 0.0
            h.fused_score = float(fused.get(rid, 0.0))
            in_vec, in_lex = rid in vec_set, rid in lex_set
            h.match = "both" if (in_vec and in_lex) else ("vector" if in_vec else "keyword")
            out.append(h)
        return out
