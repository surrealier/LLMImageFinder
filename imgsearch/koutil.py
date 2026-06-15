"""결정론적 MOCK 백엔드가 사용하는 한국어 인식 텍스트 유틸리티.

ML을 전혀 쓰지 않고 완전히 결정론적이다(md5 해시 버킷 기반). 오프라인에서도 텍스트→이미지
검색을 *시연 가능하게* 만들 정도면 충분하다: 질의 토큰이 폴더명이나 사이드카 캡션과 겹치면
(직접 일치, 소규모 KO->EN 사전을 통한 일치, 또는 공유 문자 n-gram을 통한 일치) 더 높은
점수를 받는다. 진짜 의미 기반 검색은 jina-clip 백엔드의 몫이다.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

# 끝에 붙는 조사 목록. 원본 토큰을 *파괴하지 않고* 조사를 뗀 형태를 추가 피처로만 넣는다.
# 따라서 우연히 조사 글자로 끝나는 명사를 잘못 잘라도(예: 오토바이 -> 오토바) 무해하다:
# 원본 토큰도 그대로 매칭에 쓰이기 때문이다.
_JOSA = [
    "으로서", "으로써", "이라고", "라고서", "에서는", "에서도", "에게서",
    "에서", "에게", "한테", "께서", "부터", "까지", "보다", "처럼", "마다",
    "이라", "라고", "이나", "라도", "에는", "에도", "으로", "로서", "로써",
    "은", "는", "이", "가", "을", "를", "와", "과", "도", "만", "의", "에",
    "로", "랑", "께", "들",
]
# 긴 조사를 먼저 시도하도록 길이 내림차순 정렬(예: "에서는"을 "는"보다 먼저 매칭).
_JOSA.sort(key=len, reverse=True)

# 시각적 의미가 없는 검색 메타 단어(불용어). 피처에서 제외해 잡음을 줄인다.
STOPWORDS = {
    "이미지", "사진", "그림", "사진들", "리스트", "목록", "장면", "모습", "느낌",
    "찾아줘", "찾아", "찾기", "검색", "보여줘", "보여", "줘", "관련", "그리고",
    "또는", "좀", "것", "거", "등", "수", "그", "이런", "저런", "어떤", "해줘",
}

# 한국어 토큰 -> 추가 영어/유의어 항목. mock 모드에서 한국어 질의가 영어로 명명된
# 폴더명과도 매칭될 수 있도록 질의를 확장하는 용도다.
KO2EN: dict[str, list[str]] = {
    "오토바이": ["motorcycle", "motorbike", "bike", "scooter"],
    "바이크": ["motorcycle", "bike"],
    "스쿠터": ["scooter", "motorcycle"],
    "밤": ["night", "nighttime", "dark"],
    "야간": ["night", "nighttime"],
    "밤중": ["night", "midnight"],
    "낮": ["day", "daytime"],
    "주간": ["day", "daytime"],
    "주차": ["parking", "parked"],
    "주차장": ["parking", "lot"],
    "불": ["fire", "flame"],
    "화재": ["fire", "blaze"],
    "연기": ["smoke"],
    "사람": ["person", "people", "pedestrian"],
    "사람들": ["people", "pedestrians", "crowd"],
    "보행자": ["pedestrian", "people"],
    "군중": ["crowd", "people"],
    "대로변": ["street", "roadside", "avenue"],
    "도로": ["road", "street"],
    "거리": ["street", "downtown"],
    "골목": ["alley", "street"],
    "자동차": ["car", "vehicle"],
    "차량": ["vehicle", "car"],
    "차": ["car", "vehicle"],
    "트럭": ["truck"],
    "버스": ["bus"],
    "자전거": ["bicycle", "bike"],
    "강아지": ["dog", "puppy"],
    "개": ["dog"],
    "고양이": ["cat"],
    "바다": ["sea", "ocean", "beach"],
    "해변": ["beach", "shore"],
    "건물": ["building"],
    "사고": ["accident", "crash"],
    "비": ["rain", "rainy"],
    "눈": ["snow", "snowy"],
    "안개": ["fog", "mist"],
    "신호등": ["traffic light", "signal"],
    "횡단보도": ["crosswalk", "crossing"],
    "공원": ["park"],
    "실내": ["indoor", "inside"],
    "실외": ["outdoor", "outside"],
    "야외": ["outdoor"],
}

# 개념 트리거(한국어 또는 소문자 영어) -> mock 캡션에 쓰일 한국어 표현.
# 토큰이 이 사전에 걸리면 해당 한국어 개념 문구를 캡션 힌트로 끌어온다.
CONCEPTS: dict[str, str] = {
    "fire": "불/화재", "불": "불/화재", "화재": "불/화재", "flame": "불꽃",
    "smoke": "연기", "연기": "연기",
    "motorcycle": "오토바이", "motorbike": "오토바이", "오토바이": "오토바이",
    "bike": "이륜차", "scooter": "스쿠터",
    "night": "야간", "밤": "야간", "야간": "야간", "dark": "어두움",
    "day": "주간", "낮": "주간",
    "parking": "주차", "parked": "주차", "주차": "주차",
    "person": "사람", "people": "사람", "pedestrian": "보행자",
    "사람": "사람", "보행자": "보행자", "crowd": "군중",
    "street": "거리/도로", "road": "도로", "거리": "거리", "도로": "도로",
    "대로변": "대로변",
    "car": "자동차", "vehicle": "차량", "자동차": "자동차", "truck": "트럭",
    "bus": "버스", "bicycle": "자전거", "자전거": "자전거",
    "dog": "강아지", "개": "강아지", "cat": "고양이", "고양이": "고양이",
    "sea": "바다", "ocean": "바다", "beach": "해변", "바다": "바다", "해변": "해변",
    "building": "건물", "건물": "건물",
    "rain": "비", "snow": "눈", "fog": "안개",
    "crosswalk": "횡단보도", "횡단보도": "횡단보도",
    "indoor": "실내", "outdoor": "야외",
}

# 토큰 경계 정규식: 숫자/영문/한글이 연속된 덩어리만 토큰으로 인정(공백·기호 등은 구분자).
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def tokenize(text: str) -> list[str]:
    """영문은 소문자화하고, (영숫자|한글)이 아닌 문자로 분리한 뒤, 검색 메타 불용어를 제거한다."""
    if not text:
        return []
    out: list[str] = []
    for m in _TOKEN_RE.findall(text):
        # 한글은 대소문자 개념이 없으므로 ASCII일 때만 소문자화한다.
        t = m.lower() if m.isascii() else m
        if t in STOPWORDS or not t:
            continue
        out.append(t)
    return out


def strip_one_josa(token: str) -> str:
    """가장 긴 후행 조사 하나를 제거한다(최소 1글자는 남김). 조사가 없으면 원본 그대로 반환."""
    for j in _JOSA:
        # _JOSA는 길이 내림차순이라, 첫 매치가 곧 가장 긴 조사다.
        # len(token) > len(j) 조건으로 토큰 전체가 조사로 사라지는 것을 막는다.
        if len(token) > len(j) and token.endswith(j):
            return token[: -len(j)]
    return token


def char_ngrams(token: str, n: int) -> list[str]:
    """토큰을 길이 n의 연속 문자 n-gram 리스트로 자른다(짧으면 빈 리스트)."""
    if len(token) < n:
        return []
    return [token[i : i + n] for i in range(len(token) - n + 1)]


def _bucket(s: str, dim: int) -> int:
    """문자열을 md5 해시로 [0, dim) 범위의 버킷 인덱스로 사상한다(해싱 트릭)."""
    # md5의 앞 4바이트만 little-endian 정수로 읽어 dim으로 나눈 나머지를 인덱스로 쓴다.
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little") % dim


def feature_tokens(text: str) -> list[tuple[str, float]]:
    """텍스트에서 (피처 문자열, 가중치) 쌍 목록을 만든다. 가중치는 신뢰도 순으로 차등을 둔다.

    원형 단어(3.0) > 조사 제거형/영어 확장(2.0) > 2-gram(1.0) > 3-gram(0.7) 순으로,
    정확히 일치할수록 높은 점수를 주도록 설계했다.
    """
    feats: list[tuple[str, float]] = []
    for tok in tokenize(text):
        # 원형 단어는 가장 강한 신호.
        feats.append(("w:" + tok, 3.0))
        st = strip_one_josa(tok)
        if st and st != tok:
            # 조사를 뗀 형태도 단어 피처로 추가(원형은 그대로 유지됨).
            feats.append(("w:" + st, 2.0))
        # KO->EN 확장: 한국어 질의가 영어로 명명된 폴더명까지 닿도록 한다.
        for en in KO2EN.get(tok, []) + KO2EN.get(st, []):
            feats.append(("w:" + en, 2.0))
        # 부분 일치/오타 내성을 위해 문자 n-gram도 약한 신호로 섞는다.
        for g in char_ngrams(tok, 2):
            feats.append(("g2:" + g, 1.0))
        for g in char_ngrams(tok, 3):
            feats.append(("g3:" + g, 0.7))
    return feats


def text_features(text: str, dim: int) -> np.ndarray:
    """결정론적이고 L2 정규화된 해시 BoW(bag-of-features) 벡터를 만든다(절대 영벡터 아님)."""
    vec = np.zeros(dim, dtype=np.float32)
    for feat, w in feature_tokens(text):
        # 같은 버킷에 떨어지는 피처들의 가중치를 누적(해시 충돌은 의도적으로 허용).
        vec[_bucket(feat, dim)] += w
    if not np.any(vec):  # 코사인 유사도가 정의되도록 영벡터를 피한다(퇴화 방지)
        # 추출된 피처가 하나도 없으면 원문 자체를 한 버킷에 넣어 비영(non-zero)으로 만든다.
        vec[_bucket("raw:" + (text or "∅"), dim)] = 1.0
    n = float(np.linalg.norm(vec))
    if n > 0:
        # L2 정규화: 코사인 유사도가 단순 내적으로 계산되도록 단위 벡터화.
        vec /= n
    return vec


def expand_query(ko_text: str) -> str:
    """질의를 정제하고 KO->EN 확장해 공백으로 이은 검색어 문자열로 만든다(mock의 질의 정제)."""
    seen: set[str] = set()  # 중복 제거용
    out: list[str] = []     # 입력 순서를 보존하기 위한 리스트

    def add(term: str) -> None:
        """아직 보지 않은 항목만 순서를 유지하며 추가한다."""
        if term and term not in seen:
            seen.add(term)
            out.append(term)

    for tok in tokenize(ko_text):
        add(tok)
        st = strip_one_josa(tok)
        if st != tok:
            add(st)
        for en in KO2EN.get(tok, []) + KO2EN.get(st, []):
            add(en)
    # 확장 결과가 비면(예: 전부 불용어) 원문을 그대로 돌려 빈 질의를 만들지 않는다.
    return " ".join(out) if out else ko_text


def concept_hints(text: str) -> list[str]:
    """텍스트에서 감지된 한국어 개념 문구 목록(중복 제거, 등장 순서 유지)."""
    hints: list[str] = []
    for tok in tokenize(text):
        # 원형으로 먼저 찾고, 없으면 조사 제거형으로 한 번 더 시도한다.
        phrase = CONCEPTS.get(tok) or CONCEPTS.get(strip_one_josa(tok))
        if phrase and phrase not in hints:
            hints.append(phrase)
    return hints
