"""Korean-aware text utilities used by the deterministic MOCK backends.

No ML, fully deterministic (md5-hashed buckets). Good enough to make text->image
retrieval *demonstrable* offline: query tokens that overlap a folder name / sidecar
caption (directly, via a small KO->EN lexicon, or via shared character n-grams) score
higher. Real semantic retrieval is the job of the jina-clip backend.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

# Trailing particles (josa). We never *destroy* the original token — we add the
# stripped form as an extra feature — so corrupting a noun that happens to end in a
# josa char (e.g. 오토바이 -> 오토바) is harmless: the original is still matched.
_JOSA = [
    "으로서", "으로써", "이라고", "라고서", "에서는", "에서도", "에게서",
    "에서", "에게", "한테", "께서", "부터", "까지", "보다", "처럼", "마다",
    "이라", "라고", "이나", "라도", "에는", "에도", "으로", "로서", "로써",
    "은", "는", "이", "가", "을", "를", "와", "과", "도", "만", "의", "에",
    "로", "랑", "께", "들",
]
_JOSA.sort(key=len, reverse=True)

# Search-meta words that carry no visual meaning.
STOPWORDS = {
    "이미지", "사진", "그림", "사진들", "리스트", "목록", "장면", "모습", "느낌",
    "찾아줘", "찾아", "찾기", "검색", "보여줘", "보여", "줘", "관련", "그리고",
    "또는", "좀", "것", "거", "등", "수", "그", "이런", "저런", "어떤", "해줘",
}

# KO token -> extra English / synonym terms, so a Korean query can match
# English-named folders in mock mode.
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

# Concept triggers (KO or EN, lowercased) -> a Korean phrase used in mock captions.
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

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def tokenize(text: str) -> list[str]:
    """Lowercase latin, split on non-(alnum|Hangul), drop search-meta stopwords."""
    if not text:
        return []
    out: list[str] = []
    for m in _TOKEN_RE.findall(text):
        t = m.lower() if m.isascii() else m
        if t in STOPWORDS or not t:
            continue
        out.append(t)
    return out


def strip_one_josa(token: str) -> str:
    """Strip the single longest matching trailing josa (leaving >=1 char)."""
    for j in _JOSA:
        if len(token) > len(j) and token.endswith(j):
            return token[: -len(j)]
    return token


def char_ngrams(token: str, n: int) -> list[str]:
    if len(token) < n:
        return []
    return [token[i : i + n] for i in range(len(token) - n + 1)]


def _bucket(s: str, dim: int) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little") % dim


def feature_tokens(text: str) -> list[tuple[str, float]]:
    feats: list[tuple[str, float]] = []
    for tok in tokenize(text):
        feats.append(("w:" + tok, 3.0))
        st = strip_one_josa(tok)
        if st and st != tok:
            feats.append(("w:" + st, 2.0))
        # KO->EN expansion so Korean queries reach English folder names
        for en in KO2EN.get(tok, []) + KO2EN.get(st, []):
            feats.append(("w:" + en, 2.0))
        for g in char_ngrams(tok, 2):
            feats.append(("g2:" + g, 1.0))
        for g in char_ngrams(tok, 3):
            feats.append(("g3:" + g, 0.7))
    return feats


def text_features(text: str, dim: int) -> np.ndarray:
    """A deterministic, L2-normalized hashed bag-of-features vector (never all-zero)."""
    vec = np.zeros(dim, dtype=np.float32)
    for feat, w in feature_tokens(text):
        vec[_bucket(feat, dim)] += w
    if not np.any(vec):  # keep it non-degenerate for cosine
        vec[_bucket("raw:" + (text or "∅"), dim)] = 1.0
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


def expand_query(ko_text: str) -> str:
    """Clean + KO->EN-expand a query into space-joined search terms (mock refine)."""
    seen: set[str] = set()
    out: list[str] = []

    def add(term: str) -> None:
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
    return " ".join(out) if out else ko_text


def concept_hints(text: str) -> list[str]:
    """Korean concept phrases detected in the text (deduped, order-stable)."""
    hints: list[str] = []
    for tok in tokenize(text):
        phrase = CONCEPTS.get(tok) or CONCEPTS.get(strip_one_josa(tok))
        if phrase and phrase not in hints:
            hints.append(phrase)
    return hints
