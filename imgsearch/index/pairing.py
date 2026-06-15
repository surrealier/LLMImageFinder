"""이미지와 그 옆에 놓인 사이드카(sidecar) 텍스트를 짝지어 캡션 텍스트를 추출한다.

사이드카란 이미지와 같은 파일명 줄기(stem)를 가진 .txt/.caption/.json 파일을 말하며,
폴더 단위 메모(caption.txt/readme.txt 등)도 함께 수집한다. 이렇게 모은 텍스트는
검색 시 어휘(BM25) 매칭과 임베딩 문서(document)의 재료가 된다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

from imgsearch.config import SIDECAR_TEXT_EXTS

# JSON 사이드카에서 캡션으로 쓸 후보 키. 앞쪽 키가 우선순위가 높다(가장 "설명다운" 키부터).
_JSON_KEYS = ("caption", "text", "description", "desc", "prompt", "label")
# 폴더 전체를 설명하는 메모 파일 이름들(개별 이미지가 아니라 폴더 공통 설명).
_FOLDER_NOTE_NAMES = ("caption.txt", "readme.txt", "description.txt", "info.txt")


def _read_text_file(path: Path) -> str:
    """사이드카 파일 하나를 읽어 "캡션으로 쓸 만한 순수 텍스트"로 환원한다.

    확장자에 따라 처리가 다르다:
      * .txt  : YOLO 라벨 형식이면 캡션이 아니므로 빈 문자열로 거른다.
      * .json : 알려진 키에서 문자열 값을 뽑고, 실패하면 문자열 값들을 이어 붙인다.
      * 그 외 : 내용을 그대로 사용한다.
    읽기 실패/파싱 실패 시에도 예외를 던지지 않고 안전한 기본값을 돌려준다.
    """
    try:
        # errors="ignore": 인코딩이 깨진 바이트는 버려서, 한 파일 때문에 인덱싱이 멈추지 않게 한다.
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""
    if path.suffix.lower() == ".txt":
        # 순환 import 회피: labels 모듈은 pairing 을 쓰지 않지만, 함수 내부에서 늦게 import 한다.
        from imgsearch.index.labels import is_yolo_text

        if is_yolo_text(raw):  # YOLO 탐지 라벨은 사람이 읽는 캡션이 아니므로 텍스트로 취급하지 않는다.
            return ""
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # JSON 이 깨졌으면 원문 자체를 캡션으로 사용(아예 버리는 것보다 낫다).
            return raw
        if isinstance(data, dict):
            # 우선순위가 높은 키부터 문자열 값을 찾아 즉시 반환.
            for k in _JSON_KEYS:
                if k in data and isinstance(data[k], str):
                    return data[k].strip()
            # 알려진 키가 없으면 dict 안의 모든 문자열 값을 공백으로 이어 붙여 사용.
            return " ".join(str(v) for v in data.values() if isinstance(v, str)).strip()
        # dict 가 아닌 JSON(리스트/문자열 등)은 원문을 그대로 사용.
        return raw
    return raw


def sidecar_for(image_path: str | Path, text_exts: Sequence[str] = SIDECAR_TEXT_EXTS) -> str:
    """이미지와 같은 줄기를 가진 사이드카 파일의 텍스트를 반환한다(없으면 빈 문자열).

    예) ``cat.jpg`` -> ``cat.txt`` / ``cat.json``. ``text_exts`` 순서대로 첫 번째로
    "내용이 있는" 사이드카를 채택한다.
    """
    p = Path(image_path)
    for ext in text_exts:
        # 확장자만 교체해 후보 경로를 만든다(cat.jpg -> cat.txt 등).
        cand = p.with_suffix(ext)
        # cand != p 가드: 이미지 확장자가 text_exts 에 들어 있는 경우 자기 자신을 사이드카로
        # 오인해 읽는 사고를 막는다.
        if cand.exists() and cand != p:
            txt = _read_text_file(cand)
            if txt:  # 비어 있지 않은 첫 사이드카에서 즉시 반환.
                return txt
    return ""


def folder_sidecar_text(
    dirpath: str | Path,
    image_paths: Sequence[str],
    text_exts: Sequence[str] = SIDECAR_TEXT_EXTS,
    max_items: int = 8,
    max_chars: int = 800,
) -> str:
    """리프 폴더 하나에 대해 사이드카 텍스트들을 중복 제거 후 이어 붙인 문자열(상한 적용).

    인자:
        max_items: 폴더 내 이미지 중 사이드카를 훑을 최대 개수(이미지가 수천 장이어도
                   대표적인 앞쪽 일부만 보고 비용을 제한).
        max_chars: 최종 문자열 길이 상한(메타데이터/문서 크기 폭증 방지).
    """
    parts: list[str] = []
    seen: set[str] = set()  # 동일 텍스트 중복 추가 방지(여러 이미지가 같은 캡션을 가질 수 있음).

    # 폴더 공통 메모 파일을 가장 먼저 수집한다(폴더 전체를 설명하므로 우선순위가 높다).
    d = Path(dirpath)
    for name in _FOLDER_NOTE_NAMES:
        f = d / name
        if f.exists():
            t = _read_text_file(f)
            if t and t not in seen:
                seen.add(t)
                parts.append(t)

    # 이어서 개별 이미지의 사이드카를 앞쪽 max_items 장까지만 수집한다.
    for img in image_paths[:max_items]:
        t = sidecar_for(img, text_exts)
        if t and t not in seen:
            seen.add(t)
            parts.append(t)

    # 각 조각의 내부 공백을 단일 공백으로 정규화(split/join)한 뒤 하나로 합치고,
    # 마지막에 max_chars 로 잘라 길이를 제한한다.
    joined = " ".join(" ".join(p.split()) for p in parts)
    return joined[:max_chars]
