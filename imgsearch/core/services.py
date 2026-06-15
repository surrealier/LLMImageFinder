"""SearchService — UI가 '검색'을 위해 대화하는 유일한 진입점 객체.

세 가지 검색(retrieval) 모드를 지원한다(``cfg.search_mode``로 선택):
  * "vector"  — CLIP 코사인 유사도만 사용(v0.1/v0.2 시절의 기본 동작).
  * "keyword" — 저장된 캡션에 대한 BM25 어휘 검색만 사용.
  * "hybrid"  — 둘 다 수행한 뒤 RRF(역순위 융합)로 '정렬 순서'를 융합. 단, 각 히트는
                실제 코사인 값을 ``score``에 그대로 유지하므로 점수 임계값 필터가
                모드와 무관하게 일관된 의미를 갖는다.

BM25(어휘) 색인은 스토어의 문서들로부터 '지연 생성(lazy)'되며, 스토어의 쓰기 리비전
(write-revision)이 바뀔 때만 다시 만든다. 이렇게 해야 prune/갱신으로 사라진 오래된 id를
검색 결과로 내보내는 일이 없다.
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

# RRF(Reciprocal Rank Fusion)의 감쇠 상수. 60은 원 논문에서 제시된 표준 기본값으로,
# 상위권 순위 차이를 과도하게 벌리지 않도록 완충한다.
_RRF_K = 60  # reciprocal-rank-fusion damping constant (standard default)


class SearchService:
    """검색 파이프라인을 한곳에 캡슐화한 서비스.

    임베더(벡터)·스토어(ChromaDB)·채팅 LLM(선택)을 주입받아, 질의를 받아 정제→검색→요약
    단계를 거쳐 결과를 돌려준다. UI는 백엔드 세부사항을 몰라도 이 객체만 호출하면 된다.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: ChromaStore,
        chat: Optional[ChatLLM],
        config: AppConfig,
    ) -> None:
        """검색에 필요한 의존성을 주입받는다.

        ``chat``은 선택 사항(None 가능)이며, 있으면 질의 정제·요약에 활용한다.
        어휘 색인(``_lexical``)은 여기서 만들지 않고 첫 키워드/하이브리드 질의 때
        지연 생성한다.
        """
        self.embedder = embedder
        self.store = store
        self.chat = chat
        self.cfg = config
        self._lexical: LexicalIndex | None = None
        # 캐시된 어휘 색인을 만든 시점의 store.revision() 값. 이 값이 달라지면
        # 스토어 내용이 바뀐 것이므로 어휘 색인을 다시 만들어야 한다.
        self._lex_sig: tuple | None = None  # store.revision() the index was built at

    # ------------------------------------------------------------------ public
    def query(self, text: str, k: Optional[int] = None) -> QueryResult:
        """사용자 질의 1건을 받아 정제→검색→요약까지 수행한 전체 결과를 돌려준다.

        UI의 메인 검색 경로다. 빈 입력은 즉시 안내 메시지를 담은 결과로 반환한다.
        """
        text = (text or "").strip()
        k = int(k or self.cfg.top_k)  # k 미지정 시 설정의 기본 top_k 사용
        if not text:
            return QueryResult(text, "", [], "검색어를 입력하세요.")

        refined = self._refine(text)  # LLM이 있으면 질의를 확장/정제
        mode = (self.cfg.search_mode or "vector").lower()
        # 알 수 없는 모드 값이 들어오면 가장 안전한 벡터 모드로 폴백한다.
        if mode not in ("vector", "hybrid", "keyword"):
            mode = "vector"
        hits = self._retrieve(text, refined, k, mode)
        summary = self._summarize(text, hits)
        return QueryResult(query=text, refined=refined, hits=hits, summary=summary)

    def query_by_example(self, hit: FolderHit, k: Optional[int] = None) -> QueryResult:
        """기존 결과(예시 이미지)와 유사한 레코드를 찾는다 — '이미지로 검색'.

        핵심은 새로 임베딩을 계산하지 않고 '이미 저장된' 임베딩을 재사용한다는 점이다
        (일반 경로에서는 모델 호출이 없음). 설계상 벡터 전용 검색이다.
        """
        # 색인 시점의 id 규칙과 동일한 image_id를 써야 저장된 벡터를 찾을 수 있어
        # 함수 안에서 지연 import 한다(모듈 로딩 순환 의존 회피 목적도 있음).
        from imgsearch.index.indexer import image_id

        k = int(k or self.cfg.top_k)
        # 색인 단위(granularity)에 따라 조회 키가 다르다: 이미지 단위면 image_id, 폴더 단위면 folder_id.
        rid = image_id(hit.image_path) if self.cfg.index_granularity == "image" else folder_id(hit.folder)
        vec = self.store.get_embedding(rid)
        if vec is None:
            # 저장된 벡터가 사라진 경우(예: 재빌드 없이 granularity만 바뀜).
            # 여기서 즉석 임베딩을 만들지 '않는다' — 이 메서드는 GUI 스레드에서 돌 수 있어
            # 무거운 모델 호출로 UI를 멈추면 안 되기 때문. 대신 빈 결과 + 안내를 돌려준다.
            return QueryResult(query=f"유사 이미지: {os.path.basename(hit.image_path or hit.folder)}",
                               refined="", hits=[],
                               summary="저장된 임베딩을 찾지 못했습니다 — 전체 재빌드 후 다시 시도하세요.")
        # 자기 자신이 1순위로 잡히므로 k+1개를 받아 자신을 제외하고 k개로 자른다.
        hits = self.store.query(vec, k + 1)
        hits = [h for h in hits if h.image_path != hit.image_path][:k]
        for h in hits:
            h.match = "vector"  # 출처는 전부 벡터 검색
        label = os.path.basename(hit.image_path or hit.folder)
        return QueryResult(query=f"유사 이미지: {label}", refined="", hits=hits, summary="")

    def search(self, text: str, k: Optional[int] = None, mode: Optional[str] = None,
               refine: bool = True) -> list[FolderHit]:
        """히트 목록만 반환한다(요약 없음).

        에이전트형(agentic) 파이프라인이 이 메서드를 쓴다. 그쪽은 이미 자체 계획으로
        검색 텍스트를 만들어 두므로 ``refine=False``로 호출해 중복 정제를 막는다.
        """
        text = (text or "").strip()
        if not text:
            return []
        k = int(k or self.cfg.top_k)
        # 모드 우선순위: 인자 > 설정값 > "vector"(폴백). 알 수 없는 값은 벡터로 보정.
        mode = (mode or self.cfg.search_mode or "vector").lower()
        if mode not in ("vector", "hybrid", "keyword"):
            mode = "vector"
        # refine=False면 호출자가 넘긴 text를 그대로 검색어로 사용한다.
        refined = self._refine(text) if refine else text
        return self._retrieve(text, refined, k, mode)

    def member_images(self, folder: str) -> list[str]:
        """대표 이미지가 아니라, 해당 폴더에 속한 '모든' 이미지 경로 목록을 돌려준다.

        결과 상세 보기에서 한 폴더의 멤버 전체를 펼쳐 보여줄 때 사용한다.
        """
        return walker.list_images(folder, self.cfg.image_exts)

    def count(self) -> int:
        """현재 스토어에 색인된 레코드 수(상태 표시줄 등에 사용)."""
        return self.store.count()

    def invalidate_lexical(self) -> None:
        """캐시된 BM25 색인을 버려, 다음 키워드/하이브리드 질의 때 현재 스토어 문서로
        다시 만들도록 한다(색인 빌드/갱신/prune 직후에 호출).

        리비전 비교만으로도 자동 재생성되지만, 변경을 즉시 반영하고 싶을 때 명시적으로 호출한다.
        """
        self._lexical = None
        self._lex_sig = None

    # ------------------------------------------------------------------ steps
    def _refine(self, text: str) -> str:
        """채팅 LLM으로 질의를 정제/확장한다(가능할 때만).

        LLM이 없거나 호출이 실패하면 원문을 그대로 쓴다 — 정제는 어디까지나 '있으면 좋은'
        보조 단계이지, 실패해도 검색 자체는 진행돼야 하므로 예외를 삼켜 폴백한다.
        """
        refined = text
        if self.chat is not None:
            try:
                r = self.chat.refine_query(text)
                if r and r.strip():
                    refined = r.strip()
            except Exception:
                refined = text  # 빈 결과/오류 시 원문 유지
        return refined

    def _summarize(self, query: str, hits: list[FolderHit]) -> str:
        """검색 결과를 자연어로 요약한다(설정으로 켜져 있고 LLM이 있을 때만).

        요약 역시 부가 기능이라 실패하면 빈 문자열을 돌려 검색 흐름을 깨지 않는다.
        """
        if not (self.cfg.summarize_enabled and self.chat is not None):
            return ""
        try:
            return self.chat.summarize(query, hits)
        except Exception:
            return ""

    def _ensure_lexical(self) -> LexicalIndex:
        """BM25 어휘 색인을 (필요하면) 만들어 캐시한 뒤 돌려준다.

        스토어의 리비전(``revision()``)이 캐시 생성 시점과 다르면 내용이 바뀐 것이므로
        색인을 새로 만든다. 이 리비전 기반 무효화가 prune/갱신으로 사라진 id를 검색에
        노출하지 않게 막는 핵심 장치다.
        """
        sig = self.store.revision()
        if self._lexical is None or self._lex_sig != sig:
            idx = LexicalIndex()
            idx.build(self.store.all_documents())
            self._lexical = idx
            self._lex_sig = sig
        return self._lexical

    def _embed_query(self, refined: str) -> np.ndarray:
        """질의 텍스트 1건을 임베딩해 float32 벡터로 변환한다.

        embed_text는 배치(리스트) API라 단건도 리스트로 감싸 넘기고 [0]만 꺼낸다.
        Chroma/넘파이 연산과 dtype을 맞추기 위해 float32로 캐스팅한다.
        """
        return np.asarray(self.embedder.embed_text([refined])[0], dtype=np.float32)

    @staticmethod
    def _rrf(rank_lists: list[list[str]]) -> dict[str, float]:
        """여러 순위 목록을 RRF(역순위 융합) 점수로 합산한다.

        각 목록에서 어떤 id가 rank번째(0-base)에 있으면 1/(_RRF_K + rank + 1)을 더한다.
        상위권일수록 큰 가중치를 받고, 여러 목록에 공통으로 등장하면 점수가 누적되어
        '두 검색 모두에서 잘 잡힌' 항목이 자연스럽게 위로 올라온다. 점수의 절대 스케일이
        달라도 '순위'만으로 결합하므로 코사인과 BM25처럼 척도가 다른 신호를 안전히 섞을 수 있다.
        """
        scores: dict[str, float] = {}
        for ids in rank_lists:
            for rank, rid in enumerate(ids):
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        return scores

    def _retrieve(self, text: str, refined: str, k: int, mode: str) -> list[FolderHit]:
        """모드별 실제 검색 로직. query/search가 공통으로 호출하는 핵심 엔진이다.

        벡터 모드는 스토어에 바로 위임하고 끝난다. 키워드/하이브리드 모드는
        후보를 넉넉히(fan-out) 뽑아 RRF로 융합한 뒤, 모든 히트에 '진짜 코사인'을 채워 넣어
        점수 임계값 필터가 일관되게 동작하도록 보정하는 것이 이 메서드의 요점이다.
        """
        if mode == "vector":
            # 가장 단순한 경로: 임베딩 후 스토어가 top-k 코사인 검색을 처리.
            hits = self.store.query(self._embed_query(refined), k)
            for h in hits:
                h.match = "vector"
            return hits

        # --- 키워드 / 하이브리드 공통: 후보를 넓게 뽑아야 융합 후에도 k개가 남는다 ---
        # fan-out 크기: 최종 k의 3배(최소 50). 융합/필터로 줄어들 것을 감안한 여유분.
        fan = max(k * 3, 50)
        lexical = self._ensure_lexical()
        lex_pairs = lexical.search(refined, fan)  # [(id, bm25)], 점수 내림차순
        lex_ids = [rid for rid, _ in lex_pairs]

        qvec: np.ndarray | None = None
        cos: dict[str, float] = {}  # id -> 코사인 유사도(벡터 검색에서 이미 알게 된 값)
        vec_ids: list[str] = []
        if mode == "hybrid":
            # 하이브리드: 벡터 순위와 어휘 순위를 모두 구해 RRF로 융합한다.
            qvec = self._embed_query(refined)
            vec_pairs = self.store.search_vector(qvec, fan)  # [(id, cosine)]
            cos = {rid: s for rid, s in vec_pairs}  # 코사인을 미리 확보해 재계산을 줄임
            vec_ids = [rid for rid, _ in vec_pairs]
            fused = self._rrf([vec_ids, lex_ids])
        else:  # keyword
            # 키워드 전용: 어휘 순위 하나만으로 RRF 형태의 점수를 부여(정렬 일관성 유지).
            fused = {rid: 1.0 / (_RRF_K + i + 1) for i, rid in enumerate(lex_ids)}

        # 융합 점수 내림차순으로 정렬하되, 동점이면 id 사전순으로 안정 정렬(결과 재현성 확보).
        ranked = sorted(fused, key=lambda r: (-fused[r], r))[:k]
        hit_map = self.store.fetch(ranked)  # 스토어에 더 이상 없는 id는 알아서 빠진다
        # 키워드 쪽에서만 올라온 id는 코사인 값이 없어 점수 배지/임계값에 쓸 수가 없다.
        # 이런 id들의 임베딩을 '한 번의 왕복'으로 일괄 조회한다(히트마다 get() 호출 금지 — N+1 회피).
        need_cos = [rid for rid in ranked if rid in hit_map and rid not in cos]
        emb_map: dict = {}
        if need_cos:
            if qvec is None:  # 키워드 모드라 아직 질의 벡터가 없으면 이때 한 번만 만든다
                qvec = self._embed_query(refined)
            emb_map = self.store.get_embeddings(need_cos)

        vec_set, lex_set = set(vec_ids), set(lex_ids)  # match 출처 판정을 위한 빠른 조회용 집합
        out: list[FolderHit] = []
        for rid in ranked:
            h = hit_map.get(rid)
            if h is None:  # 융합 시점과 fetch 시점 사이에 사라진 레코드 -> 건너뜀
                continue
            if rid in cos:
                h.score = float(cos[rid])  # 벡터 검색에서 이미 알아낸 코사인 재사용
            else:
                emb = emb_map.get(rid)
                # 임베딩은 L2 정규화되어 있으므로 내적(dot)이 곧 코사인 유사도다.
                # max(0.0, ...)로 음수 코사인은 0으로 깎아 임계값 필터를 깔끔하게 유지.
                h.score = float(max(0.0, np.dot(emb, qvec))) if emb is not None and qvec is not None else 0.0
            h.fused_score = float(fused.get(rid, 0.0))  # 정렬에 쓴 융합 점수도 함께 기록(디버깅/표시용)
            # 출처 판정: 양쪽 다면 both, 벡터에만/키워드에만 잡혔는지로 구분.
            in_vec, in_lex = rid in vec_set, rid in lex_set
            h.match = "both" if (in_vec and in_lex) else ("vector" if in_vec else "keyword")
            out.append(h)
        return out
