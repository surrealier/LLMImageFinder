"""Keyword (BM25) retrieval — the lexical arm of hybrid search.

Pure-python via ``rank_bm25`` (a BASE dependency, no native build), tokenized with
the shared :mod:`imgsearch.koutil` Korean tokenizer (+ josa stripping) so it works on
mock captions too and is fully deterministic. Keyed by the SAME chroma record id as
the vector arm, so the two can be fused by id (reciprocal-rank fusion).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from imgsearch import koutil


def lexical_tokens(text: str) -> list[str]:
    """Korean-aware tokens for BM25: each token, its josa-stripped form, and any
    KO->EN synonyms (so a Korean query matches an English-named caption/folder, and
    the same tokenization applies symmetrically to both the corpus and the query)."""
    toks: list[str] = []
    for t in koutil.tokenize(text):
        toks.append(t)
        st = koutil.strip_one_josa(t)
        if st and st != t:
            toks.append(st)
        for en in koutil.KO2EN.get(t, []) + koutil.KO2EN.get(st, []):
            toks.append(en)
    return toks


class LexicalIndex:
    """In-memory BM25 index over (record_id, document_text) pairs."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._doc_sets: list[set[str]] = []
        self._bm25 = None
        self.size = 0

    def build(self, items: Sequence[tuple[str, str]]) -> None:
        from rank_bm25 import BM25Okapi

        self._ids = [rid for rid, _ in items]
        if not self._ids:
            self._bm25 = None
            self._doc_sets = []
            self.size = 0
            return
        # a token-less doc would break BM25 idf math -> sentinel keeps lengths sane
        corpus = [lexical_tokens(doc) or ["∅"] for _, doc in items]
        self._doc_sets = [set(toks) for toks in corpus]
        self._bm25 = BM25Okapi(corpus)
        self.size = len(self._ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """(record_id, bm25_score) for the top-k docs that actually SHARE a token
        with the query. Candidacy is by token overlap (not score>0): Okapi IDF is 0
        for a term in ~half the docs and negative for very common ones, so a real
        match must not be dropped just because the term is frequent."""
        if self._bm25 is None or not self._ids:
            return []
        q = lexical_tokens(query)
        if not q:
            return []
        qset = set(q)
        scores = self._bm25.get_scores(q)
        cand = [
            (i, float(scores[i]), len(qset & self._doc_sets[i]))
            for i in range(len(self._ids))
            if qset & self._doc_sets[i]
        ]
        # by BM25 score desc, then overlap count desc, then id (deterministic ties)
        cand.sort(key=lambda t: (-t[1], -t[2], self._ids[t[0]]))
        return [(self._ids[i], s) for i, s, _ov in cand[: max(1, int(k))]]
