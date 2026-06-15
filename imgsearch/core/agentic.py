"""앱 내장 멀티 에이전트 검색 — 사용자가 실행 과정을 지켜볼 수 있는 작은 A2A 그래프.

    planner(계획) ─▶ retriever(하이브리드 검색) ─▶ graph-filter(객체 그래프 필터) ─▶ summarizer(요약)

각 단계는 ``step_cb``(평범한 callable)를 통해 한국어 추적(trace) 한 줄을 내보낸다. UI는 이
콜백을 Qt 시그널에 연결해 메시지가 GUI 스레드로 안전하게 전달되도록 한다(워커 스레드에서
직접 위젯을 건드리면 안 되므로).

mock 백엔드 위에서도 모든 단계가 '결정론적'으로 동작하도록 설계되어 있다:
  * planner — 채팅 백엔드에 ``plan()``이 없으면 규칙 기반 계획(koutil)으로 폴백.
  * retriever — BM25 + 벡터 하이브리드.
  * graph filter — 메모리 내 객체 그래프.
따라서 ML 의존성이 전혀 없어도 전체 파이프라인이 끝까지 돌아간다.
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

# 각 단계가 진행 상황을 알릴 때 호출하는 콜백 타입(문자열 한 줄 -> None).
StepCb = Callable[[str], None]


def _noop(_msg: str) -> None:
    """기본 step 콜백(아무것도 하지 않음).

    UI 없이(예: 테스트) run()을 호출할 때, 콜백 None 체크를 매번 하지 않으려고 쓰는
    무동작 기본값이다.
    """
    return None


@dataclass
class Plan:
    """planner가 산출한 검색 계획.

    질의를 '의미 검색용 텍스트(semantic)'와 '객체 제약(필수/제외)'으로 분해한 것이다.
    이후 하이브리드 검색은 semantic을, 그래프 필터는 required/excluded 객체 목록을 쓴다.
    """

    semantic: str  # 의미 검색에 쓸 (확장된) 질의 텍스트
    required_objects: list[str] = field(default_factory=list)  # 반드시 포함돼야 하는 객체 클래스명
    excluded_objects: list[str] = field(default_factory=list)  # 결과에서 빼야 하는 객체 클래스명


def _class_name_set(class_names) -> list[str]:
    """객체 클래스명 컬렉션을 형태와 무관하게 문자열 리스트로 정규화한다.

    설정에 따라 class_names가 {id: name} dict로 올 수도, 단순 리스트로 올 수도 있어
    양쪽을 모두 받아 동일한 list[str] 형태로 맞춘다(None도 안전하게 빈 리스트 처리).
    """
    if isinstance(class_names, dict):
        return [str(v) for v in class_names.values()]
    return [str(v) for v in (class_names or [])]


def rule_based_plan(query: str, class_names) -> Plan:
    """결정론적·ML 없는 계획 수립: 질의에 등장하는 객체명을 필수 필터로 삼고,
    부정 표현(없는/제외/말고/빼고/없이)이 붙은 객체는 제외 필터로 분류한다.
    semantic 텍스트는 koutil로 확장한 질의다.

    LLM이 없거나 LLM 계획이 무효일 때의 폴백 경로로, 입력이 같으면 결과가 항상 같다.
    """
    names = _class_name_set(class_names)
    q = query or ""
    required: list[str] = []
    excluded: list[str] = []
    neg_markers = ("없는", "없이", "제외", "말고", "빼고", "아닌")
    for nm in names:
        if nm and nm in q:
            # 객체명이 언급된 '바로 뒤 ~6글자'만 살펴 부정 표현이 있는지 확인한다.
            # (예: "사람 없는" -> '사람'은 제외로 분류) 전체 문장이 아닌 인접 꼬리만
            # 보는 휴리스틱이라, 같은 객체가 다른 맥락에서 또 쓰여도 영향을 덜 받는다.
            tail = q[q.find(nm) + len(nm): q.find(nm) + len(nm) + 6]
            (excluded if any(m in tail for m in neg_markers) else required).append(nm)
    # 중복 제거 + 등장 순서 보존(dict.fromkeys는 입력 순서를 유지한다).
    required = list(dict.fromkeys(required))
    # 같은 객체가 필수와 제외에 동시에 잡히면 모순이므로 필수를 우선해 제외에서 뺀다.
    excluded = [e for e in dict.fromkeys(excluded) if e not in required]
    # semantic: 확장 실패(빈 값) 시 원문 질의로 폴백.
    return Plan(semantic=koutil.expand_query(query) or query,
                required_objects=required, excluded_objects=excluded)


def _validate_plan(raw, class_names) -> Optional[Plan]:
    """LLM이 돌려준 원시(raw) 계획을 검증·정제해 Plan으로 만든다(실패 시 None).

    신뢰할 수 없는 LLM 출력을 그대로 믿지 않는 방어 계층이다:
      * dict가 아니거나 semantic이 비어 있으면 무효(None) — 호출부는 규칙 기반으로 폴백.
      * required/excluded는 '실제 존재하는 클래스명'만 통과시켜 환각(hallucinated) 객체명을 차단.
    """
    if not isinstance(raw, dict):
        return None
    names = set(_class_name_set(class_names))
    semantic = str(raw.get("semantic") or "").strip()
    # 알려진 클래스명에 속하는 항목만 남긴다(존재하지 않는 객체명은 버림).
    req = [str(x) for x in (raw.get("required_objects") or []) if str(x) in names]
    exc = [str(x) for x in (raw.get("excluded_objects") or []) if str(x) in names]
    if not semantic:  # 의미 검색 텍스트가 없으면 계획 자체가 쓸모없다
        return None
    return Plan(semantic=semantic, required_objects=req, excluded_objects=exc)


class AgenticSearch:
    """계획→검색→그래프 필터→요약으로 이어지는 에이전트형 검색 오케스트레이터.

    SearchService(벡터/어휘 검색)와 GraphStore(객체 동시출현 그래프)를 조합해, 단순
    검색으로는 어려운 '객체 포함/제외' 같은 제약을 만족하는 결과를 단계적으로 좁혀 나간다.
    """

    def __init__(
        self,
        service: SearchService,
        graph: Optional[GraphStore],
        chat,
        config: AppConfig,
    ) -> None:
        """검색 서비스·객체 그래프(선택)·채팅 LLM·설정을 주입받는다.

        ``graph``는 None일 수 있으며(그래프 미구축 등), 그 경우 객체 필터는 자동으로 건너뛴다.
        """
        self.service = service
        self.graph = graph
        self.chat = chat
        self.cfg = config

    # -- planner --
    def _plan(self, query: str) -> Plan:
        """질의에 대한 검색 계획을 세운다. LLM 계획을 우선 시도하고, 안 되면 규칙 기반으로 폴백.

        LLM에 ``plan()``이 있고 호출이 성공하며 출력이 검증을 통과할 때만 LLM 계획을 쓴다.
        그 외 모든 경우(메서드 없음 · 예외 · 잘못된 JSON · 죽은 엔드포인트 · 검증 실패)는
        결정론적 규칙 기반 계획으로 안전하게 떨어진다.
        """
        # 채팅 백엔드가 plan()을 제공하는지 동적으로 확인(mock에는 없을 수 있음).
        plan_fn = getattr(self.chat, "plan", None) if self.chat is not None else None
        if callable(plan_fn):
            try:
                raw = plan_fn(query, self.cfg.class_names)
                p = _validate_plan(raw, self.cfg.class_names)
                if p is not None:
                    return p
            except Exception:
                pass  # 잘못된 JSON / 죽은 엔드포인트 -> 결정론적 규칙 기반 계획으로 폴백
        return rule_based_plan(query, self.cfg.class_names)

    # -- graph filter --
    def _graph_ids(self, names: Sequence[str], mode: str) -> Optional[set[str]]:
        """객체 제약을 만족하는 이미지의 Chroma id 집합을 돌려준다.

        반환값의 의미를 세 가지로 구분하는 것이 핵심이다:
          * None      — 그래프를 쓸 수 없거나 현재 색인과 불일치 → 필터를 '건너뛰라'는 신호.
          * set()(빈 집합) — 그래프는 정상이나 조건에 맞는 이미지가 0건 → '아무것도 통과 못 함'.
          * 비어있지 않은 집합 — 조건을 만족하면서 현재 색인에도 존재하는 id들.

        mode는 "all"(모든 객체 동시 포함) 또는 그 외("any", 하나라도 포함)다.
        """
        if not self.graph or not names:
            return None  # 그래프가 없거나 제약 객체가 없으면 필터링 대상 아님
        try:
            if self.graph.count() == 0:  # 그래프가 아직 안 만들어짐 -> 필터 건너뜀
                return None
        except Exception:
            return None  # 그래프 접근 자체가 실패해도 안전하게 건너뜀
        try:
            # mode에 따라 'AND(모두 포함)' 또는 'OR(하나라도 포함)' 질의를 그래프에 던진다.
            paths = (
                self.graph.images_with_all(names)
                if mode == "all"
                else self.graph.images_with_any(names)
            )
        except Exception:
            return None
        if not paths:
            return set()  # 조건은 유효하나 매칭 0건 -> 빈 집합(None과 의미가 다름!)
        # image_id로 조인한다. 그래프는 원시 디스크 경로를 보관하지만 색인은 정규화된
        # image_id를 키로 쓰기 때문이다. 원시 경로로 직접 맞추면 Windows의 대소문자/
        # 슬래시 차이 때문에 매칭이 깨지므로 반드시 정규화 id로 비교한다.
        ids = {image_id(p) for p in paths}
        # 현재 색인에 '실제로 존재하는' id만 남긴다. 순서와 무관하며, 부분 겹침이면 전부-아니면-전무가
        # 아니라 '겹치는 부분집합'을 그대로 반환한다(비례적 동작).
        present = set(self.service.store.fetch(list(ids)))
        if not present:  # 그래프가 색인과 '다른/이동된' 루트에서 만들어진 경우
            return None  # -> 신뢰할 수 없으니 필터를 건너뛰라고 None 반환
        return ids & present

    def run(self, query: str, step_cb: StepCb = _noop, k: Optional[int] = None) -> QueryResult:
        """에이전트형 검색 전체 파이프라인을 실행한다.

        흐름: 계획 → 하이브리드 검색 → (필수 객체 필터 + 누락분 보충) → (제외 객체 필터) → 요약.
        각 단계마다 ``step_cb``로 진행 상황을 한 줄씩 알려, UI에서 실시간으로 추적 로그를
        보여줄 수 있다.
        """
        query = (query or "").strip()
        k = int(k or self.cfg.top_k)
        if not query:
            return QueryResult(query, "", [], "검색어를 입력하세요.")

        # 1) 계획 수립 — 의미 텍스트와 필수/제외 객체로 질의를 분해한다.
        plan = self._plan(query)
        req = ", ".join(plan.required_objects) or "없음"  # 추적 메시지 표시용 문자열
        exc = ", ".join(plan.excluded_objects) or "없음"
        step_cb(f"🧭 계획 — 의미: ‘{plan.semantic}’ · 필수 객체: {req} · 제외: {exc}")

        # 2) 하이브리드 검색 — 이후 그래프 필터로 줄어들 것을 감안해 넉넉히(fan) 뽑는다.
        #    refine=False: 이미 plan.semantic이 확장된 텍스트라 재정제 불필요.
        fan = max(k * 3, 30)
        hits = self.service.search(plan.semantic, k=fan, mode="hybrid", refine=False)
        step_cb(f"🔍 하이브리드 검색 — 후보 {len(hits)}건")

        # 3) 필수 객체 필터 — '모든' 필수 객체를 포함하는 이미지만 남긴다.
        if plan.required_objects:
            req_ids = self._graph_ids(plan.required_objects, "all")
            if req_ids is None:
                # 그래프와 색인이 불일치(또는 그래프 미구축) -> 필터를 건너뛴다고 안내.
                step_cb("🕸 그래프 필터 — 인덱스와 그래프가 일치하지 않아 건너뜁니다 (전체 재빌드 권장)")
            else:
                # 후보 중 조건을 만족하는 것만 유지.
                kept = [h for h in hits if image_id(h.image_path) in req_ids]
                # 의미 검색 순위에서 누락됐지만 객체 조건은 만족하는 이미지를 보충(backfill)한다.
                # 단순히 끼워 넣지 않고 plan 질의 대비 '진짜 코사인'을 계산해 부여해야
                # score가 다른 히트와 같은 척도를 유지하고 정렬/임계값이 어긋나지 않는다.
                have = {image_id(h.image_path) for h in kept}
                missing = [i for i in req_ids if i not in have]
                if missing:
                    qvec = self.service._embed_query(plan.semantic)
                    embs = self.service.store.get_embeddings(missing)  # 임베딩 일괄 조회
                    for rid, h in self.service.store.fetch(missing).items():
                        emb = embs.get(rid)
                        # L2 정규화 벡터이므로 내적=코사인. 임베딩이 없으면 0.0.
                        h.score = float(max(0.0, np.dot(emb, qvec))) if emb is not None else 0.0
                        h.match = "graph"  # 출처: 그래프 필터를 통해 보충된 항목
                        kept.append(h)
                    kept.sort(key=lambda h: h.score, reverse=True)  # 보충분 포함 재정렬
                hits = kept
                step_cb(f"🕸 그래프 필터 — 필수[{req}] 모두 포함 → {len(hits)}건")

        # 4) 제외 객체 필터 — 제외 객체를 '하나라도' 포함하면 결과에서 뺀다.
        if plan.excluded_objects:
            exc_ids = self._graph_ids(plan.excluded_objects, "any")
            if exc_ids:  # None(건너뜀)이나 빈 집합(제외 대상 없음)이면 아무것도 안 한다
                before = len(hits)
                hits = [h for h in hits if image_id(h.image_path) not in exc_ids]
                step_cb(f"🚫 제외 필터 — 제외[{exc}] → {before - len(hits)}건 제거")

        # 5) 최종 top-k로 자르고 요약을 만든다(요약 비활성/실패 시 빈 문자열).
        hits = hits[:k]
        summary = self.service._summarize(query, hits)
        if summary:
            step_cb("✍ 요약 완료")
        step_cb(f"✅ 최종 결과 {len(hits)}건")
        # refined에는 (원문이 아니라) 실제 검색에 쓴 plan.semantic을 담아 UI에서 확인 가능하게 한다.
        return QueryResult(query=query, refined=plan.semantic, hits=hits, summary=summary)
