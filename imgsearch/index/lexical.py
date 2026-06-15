"""키워드(BM25) 검색 — 하이브리드 검색의 어휘(lexical) 축.

``rank_bm25`` 기반의 순수 파이썬 구현(네이티브 빌드가 필요 없는 BASE 의존성)이며,
공용 :mod:`imgsearch.koutil` 한국어 토크나이저(+ 조사 제거)로 토큰화한다. 그래서
mock 캡션에서도 동작하고 완전히 결정적(deterministic)이다. 벡터 축과 "동일한"
chroma 레코드 id 를 키로 쓰므로, 두 결과를 id 기준으로 융합(상호 순위 융합, RRF)할 수 있다.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from imgsearch import koutil


def lexical_tokens(text: str) -> list[str]:
    """BM25 용 한국어 인지 토큰을 만든다.

    각 토큰에 더해 (1) 조사를 한 번 떼어낸 형태, (2) 한->영 동의어를 함께 넣는다.
    덕분에 한국어 질의가 영어로 이름 붙은 캡션/폴더와도 매칭되고, 동일한 토큰화가
    말뭉치(corpus)와 질의(query) 양쪽에 대칭적으로 적용된다(같은 규칙이어야 매칭이 성립).
    """
    toks: list[str] = []
    for t in koutil.tokenize(text):
        toks.append(t)
        st = koutil.strip_one_josa(t)  # 예: "고양이가" -> "고양이"
        if st and st != t:  # 조사가 실제로 떨어진 경우에만 추가(중복 방지).
            toks.append(st)
        # 원형과 조사 제거형 모두에 대해 한->영 동의어를 펼쳐 넣는다(예: "고양이" -> "cat").
        for en in koutil.KO2EN.get(t, []) + koutil.KO2EN.get(st, []):
            toks.append(en)
    return toks


class LexicalIndex:
    """(레코드 id, 문서 텍스트) 쌍에 대한 인메모리 BM25 인덱스."""

    def __init__(self) -> None:
        """빈 인덱스를 초기화한다(build() 호출 전까지는 검색이 빈 결과를 반환)."""
        self._ids: list[str] = []          # 코퍼스 순서와 1:1 대응하는 레코드 id 목록
        self._doc_sets: list[set[str]] = []  # 문서별 토큰 집합(질의와의 교집합 검사용)
        self._bm25 = None                  # BM25Okapi 인스턴스(미구축이면 None)
        self.size = 0                      # 인덱싱된 문서 수

    def build(self, items: Sequence[tuple[str, str]]) -> None:
        """(id, 문서) 쌍들로 BM25 인덱스를 (재)구축한다. 빈 입력이면 인덱스를 비운다."""
        # rank_bm25 는 무거운/선택적 의존성이라 빌드 시점에만 늦게 import.
        from rank_bm25 import BM25Okapi

        self._ids = [rid for rid, _ in items]
        if not self._ids:
            # 입력이 없으면 모든 상태를 초기화하고 종료(search 가 안전하게 빈 결과 반환).
            self._bm25 = None
            self._doc_sets = []
            self.size = 0
            return
        # 토큰이 0개인 문서는 BM25 의 idf 계산을 깨뜨리므로, 센티넬 토큰("∅")을 넣어
        # 모든 문서가 최소 1개의 토큰을 갖도록 보장한다(길이 통계가 망가지지 않게).
        corpus = [lexical_tokens(doc) or ["∅"] for _, doc in items]
        # 후보 선별 시 질의 토큰과의 교집합을 빠르게 보기 위해 문서별 토큰 집합을 미리 만든다.
        self._doc_sets = [set(toks) for toks in corpus]
        self._bm25 = BM25Okapi(corpus)
        self.size = len(self._ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """질의와 토큰을 실제로 "공유"하는 상위 k개 문서의 (레코드 id, BM25 점수)를 반환한다.

        후보 자격은 점수>0 이 아니라 "토큰 겹침" 기준이다: Okapi IDF 는 약 절반의 문서에
        등장하는 단어에서 0, 아주 흔한 단어에서는 음수가 되므로, 단지 흔하다는 이유로
        진짜 매칭을 떨어뜨리면 안 되기 때문이다.
        """
        if self._bm25 is None or not self._ids:
            return []
        q = lexical_tokens(query)
        if not q:  # 질의가 토큰을 만들지 못하면 매칭할 것이 없다.
            return []
        qset = set(q)
        scores = self._bm25.get_scores(q)
        # 질의 토큰과 한 개 이상 겹치는 문서만 후보로 추린다. 각 후보는
        # (문서 인덱스, BM25 점수, 겹친 토큰 수)로 구성한다.
        cand = [
            (i, float(scores[i]), len(qset & self._doc_sets[i]))
            for i in range(len(self._ids))
            if qset & self._doc_sets[i]
        ]
        # 정렬 기준: BM25 점수 내림차순 -> 겹침 수 내림차순 -> id 오름차순(동점을 결정적으로 깸).
        cand.sort(key=lambda t: (-t[1], -t[2], self._ids[t[0]]))
        # 상위 k개만 (id, 점수)로 돌려준다. k 가 0/음수여도 최소 1개는 반환하도록 보정.
        return [(self._ids[i], s) for i, s, _ov in cand[: max(1, int(k))]]
