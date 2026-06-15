"""In-app multi-agent search — a small A2A graph the user can watch run.

    planner ─▶ retriever(hybrid) ─▶ graph-filter ─▶ summarizer

Each stage emits a Korean trace line via ``step_cb`` (a plain callable; the UI wires
it to a Qt signal so it marshals to the GUI thread). Everything is deterministic on
the mock backends: the planner falls back to a rule-based plan (koutil) when the chat
backend has no ``plan()``, the retriever is hybrid BM25+vector, and the graph filter is
the in-memory object graph — so the whole pipeline runs with zero ML deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from imgsearch import koutil
from imgsearch.config import AppConfig
from imgsearch.core.models import FolderHit, QueryResult
from imgsearch.core.services import SearchService
from imgsearch.graph.base import GraphStore
from imgsearch.index.indexer import image_id

StepCb = Callable[[str], None]


def _noop(_msg: str) -> None:
    return None


@dataclass
class Plan:
    semantic: str
    required_objects: list[str] = field(default_factory=list)
    excluded_objects: list[str] = field(default_factory=list)


def _class_name_set(class_names) -> list[str]:
    if isinstance(class_names, dict):
        return [str(v) for v in class_names.values()]
    return [str(v) for v in (class_names or [])]


def rule_based_plan(query: str, class_names) -> Plan:
    """Deterministic, ML-free plan: object names that appear in the query become
    required filters; negated ones (없는/제외/말고/빼고/없이) become exclusions.
    The semantic text is the koutil-expanded query."""
    names = _class_name_set(class_names)
    q = query or ""
    required: list[str] = []
    excluded: list[str] = []
    neg_markers = ("없는", "없이", "제외", "말고", "빼고", "아닌")
    for nm in names:
        if nm and nm in q:
            # look at the ~6 chars following the mention for a negation marker
            tail = q[q.find(nm) + len(nm): q.find(nm) + len(nm) + 6]
            (excluded if any(m in tail for m in neg_markers) else required).append(nm)
    # de-dup, preserve order
    required = list(dict.fromkeys(required))
    excluded = [e for e in dict.fromkeys(excluded) if e not in required]
    return Plan(semantic=koutil.expand_query(query) or query,
                required_objects=required, excluded_objects=excluded)


def _validate_plan(raw, class_names) -> Optional[Plan]:
    if not isinstance(raw, dict):
        return None
    names = set(_class_name_set(class_names))
    semantic = str(raw.get("semantic") or "").strip()
    req = [str(x) for x in (raw.get("required_objects") or []) if str(x) in names]
    exc = [str(x) for x in (raw.get("excluded_objects") or []) if str(x) in names]
    if not semantic:
        return None
    return Plan(semantic=semantic, required_objects=req, excluded_objects=exc)


class AgenticSearch:
    def __init__(
        self,
        service: SearchService,
        graph: Optional[GraphStore],
        chat,
        config: AppConfig,
    ) -> None:
        self.service = service
        self.graph = graph
        self.chat = chat
        self.cfg = config

    # -- planner --
    def _plan(self, query: str) -> Plan:
        plan_fn = getattr(self.chat, "plan", None) if self.chat is not None else None
        if callable(plan_fn):
            try:
                raw = plan_fn(query, self.cfg.class_names)
                p = _validate_plan(raw, self.cfg.class_names)
                if p is not None:
                    return p
            except Exception:
                pass  # bad JSON / dead endpoint -> deterministic rule-based plan
        return rule_based_plan(query, self.cfg.class_names)

    # -- graph filter --
    def _graph_ids(self, names: Sequence[str], mode: str) -> Optional[set[str]]:
        """Chroma ids for images matching the object constraint, or None if the
        graph is unusable / mismatched with the current index."""
        if not self.graph or not names:
            return None
        try:
            if self.graph.count() == 0:  # graph not built yet -> skip the filter
                return None
        except Exception:
            return None
        try:
            paths = (
                self.graph.images_with_all(names)
                if mode == "all"
                else self.graph.images_with_any(names)
            )
        except Exception:
            return None
        if not paths:
            return set()
        # join by image_id (graph keeps raw disk paths; the index is keyed by the
        # normalized image_id) — never by raw path (Windows case/slash differences)
        ids = {image_id(p) for p in paths}
        # keep only ids that actually exist in the current index (order-independent,
        # proportional — a partial overlap returns the present subset, not all-or-nothing)
        present = set(self.service.store.fetch(list(ids)))
        if not present:  # graph built from a DIFFERENT/moved root than the index
            return None
        return ids & present

    def run(self, query: str, step_cb: StepCb = _noop, k: Optional[int] = None) -> QueryResult:
        query = (query or "").strip()
        k = int(k or self.cfg.top_k)
        if not query:
            return QueryResult(query, "", [], "검색어를 입력하세요.")

        plan = self._plan(query)
        req = ", ".join(plan.required_objects) or "없음"
        exc = ", ".join(plan.excluded_objects) or "없음"
        step_cb(f"🧭 계획 — 의미: ‘{plan.semantic}’ · 필수 객체: {req} · 제외: {exc}")

        fan = max(k * 3, 30)
        hits = self.service.search(plan.semantic, k=fan, mode="hybrid", refine=False)
        step_cb(f"🔍 하이브리드 검색 — 후보 {len(hits)}건")

        if plan.required_objects:
            req_ids = self._graph_ids(plan.required_objects, "all")
            if req_ids is None:
                step_cb("🕸 그래프 필터 — 인덱스와 그래프가 일치하지 않아 건너뜁니다 (전체 재빌드 권장)")
            else:
                kept = [h for h in hits if image_id(h.image_path) in req_ids]
                # backfill co-occurring images semantic ranking may have missed —
                # give them a REAL cosine vs the plan's query so score stays meaningful
                have = {image_id(h.image_path) for h in kept}
                missing = [i for i in req_ids if i not in have]
                if missing:
                    qvec = self.service._embed_query(plan.semantic)
                    embs = self.service.store.get_embeddings(missing)
                    for rid, h in self.service.store.fetch(missing).items():
                        emb = embs.get(rid)
                        h.score = float(max(0.0, np.dot(emb, qvec))) if emb is not None else 0.0
                        h.match = "graph"
                        kept.append(h)
                    kept.sort(key=lambda h: h.score, reverse=True)
                hits = kept
                step_cb(f"🕸 그래프 필터 — 필수[{req}] 모두 포함 → {len(hits)}건")

        if plan.excluded_objects:
            exc_ids = self._graph_ids(plan.excluded_objects, "any")
            if exc_ids:
                before = len(hits)
                hits = [h for h in hits if image_id(h.image_path) not in exc_ids]
                step_cb(f"🚫 제외 필터 — 제외[{exc}] → {before - len(hits)}건 제거")

        hits = hits[:k]
        summary = self.service._summarize(query, hits)
        if summary:
            step_cb("✍ 요약 완료")
        step_cb(f"✅ 최종 결과 {len(hits)}건")
        return QueryResult(query=query, refined=plan.semantic, hits=hits, summary=summary)
