"""YOLO 탐지 라벨 처리 도우미: 박스를 파싱하고 사람이 읽는 캡션으로 요약한다.

많은 데이터셋이 이미지마다 YOLO 형식의 ``.txt`` 파일을 함께 제공한다
(정규화된 ``class_id cx cy w h`` 형식). 이 숫자들은 캡션이 아니므로, 이 모듈은
(있으면 class-id -> 이름 맵을 써서) 읽을 수 있는 객체 요약으로 바꿔 주고,
인덱서/페어링이 라벨 숫자를 텍스트로 오인하지 않도록 판별 기능도 제공한다.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

# YOLO 한 줄 패턴: 맨 앞 정수(class_id) 뒤에 부동소수 값이 4개 이상 오는 줄.
#   ^\d+            : 클래스 id(정수)로 시작
#   (?:\s+-?\d*\.?\d+){4,} : 공백+숫자(cx cy w h, 음수/소수 허용)가 4개 이상
# 4개 "이상"인 이유: cx cy w h(4개) 외에 신뢰도 등 추가 컬럼이 붙는 변형도 YOLO 로 인정하기 위함.
_YOLO_LINE = re.compile(r"^\d+(?:\s+-?\d*\.?\d+){4,}\s*$")

Box = tuple[int, float, float, float, float]  # (class_id, cx, cy, w, h) — 정규화 좌표


def is_yolo_text(text: str) -> bool:
    """텍스트 전체가 YOLO 라벨 형식인지 판정한다(빈 줄 제외 모든 줄이 패턴과 일치해야 True).

    캡션 텍스트와 라벨 파일을 구분하는 게이트 역할을 한다(pairing 이 라벨을 캡션으로
    오인하지 않게 하려고 사용).
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # 한 줄이라도 있어야 하고(bool(lines)), 모든 비어있지 않은 줄이 YOLO 패턴과 맞아야 한다.
    return bool(lines) and all(_YOLO_LINE.match(ln) for ln in lines)


def is_yolo_file(path: str | Path) -> bool:
    """파일 내용을 읽어 YOLO 라벨 파일인지 판정한다(읽기 실패 시 False)."""
    try:
        return is_yolo_text(Path(path).read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return False


def label_path_for(image_path: str | Path, text_exts: Sequence[str] = (".txt",)) -> Optional[Path]:
    """이미지와 같은 줄기를 가진 라벨 파일 경로를 찾아 반환한다(없으면 None).

    기본은 ``.txt`` 만 대상으로 한다(YOLO 라벨의 표준 확장자).
    """
    p = Path(image_path)
    for ext in text_exts:
        cand = p.with_suffix(ext)
        # cand != p: 이미지 확장자가 text_exts 에 포함된 경우 자기 자신을 라벨로 오인하지 않게 가드.
        if cand.exists() and cand != p:
            return cand
    return None


def parse_yolo(path: str | Path) -> list[Box]:
    """YOLO 라벨 파일을 파싱해 ``Box`` 목록을 반환한다(읽기 실패 시 빈 목록).

    형식이 어긋난 줄은 조용히 건너뛰어, 일부가 깨져도 나머지 박스는 살린다.
    """
    boxes: list[Box] = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return boxes
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) >= 5:  # 최소 (class_id cx cy w h) 5개 토큰이 있어야 유효.
            try:
                # class_id 가 "0.0" 같은 부동소수로 적힌 변형도 허용하려고 float -> int 로 변환.
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(x) for x in parts[1:5])
            except ValueError:
                # 숫자로 변환되지 않는 줄(헤더/주석 등)은 건너뛴다.
                continue
            boxes.append((cid, cx, cy, w, h))
    return boxes


def class_histogram(boxes: Sequence[Box]) -> Counter:
    """박스 목록에서 클래스 id별 등장 횟수 히스토그램(Counter)을 만든다."""
    return Counter(b[0] for b in boxes)


def _name(class_names: Optional[dict], cid: int) -> str:
    """클래스 id 를 표시용 이름으로 변환한다(맵에 없으면 ``#<id>`` 형태로 대체).

    class_names 의 키가 문자열("3")일 수도 정수(3)일 수도 있어 둘 다 조회한다.
    어느 쪽도 없으면 사람이 식별할 수 있도록 ``#3`` 같은 임시 이름을 만든다.
    """
    if class_names:
        # 문자열 키 우선 -> 정수 키 -> 최후의 #id 순으로 폴백(or 단락 평가).
        return str(class_names.get(str(cid)) or class_names.get(cid) or f"#{cid}")
    return f"#{cid}"


def caption_from_boxes(boxes: Sequence[Box], class_names: Optional[dict] = None) -> str:
    """박스들을 사람이 읽는 한 줄 캡션으로 요약한다(예: ``탐지 객체: 사람×3, 자동차``).

    박스가 없으면 빈 문자열을 반환한다.
    """
    if not boxes:
        return ""
    # 표시 "이름" 기준으로 집계한다(서로 다른 class id 가 같은 이름으로 매핑될 수 있으므로).
    name_counts: Counter = Counter()
    for b in boxes:
        name_counts[_name(class_names, b[0])] += 1
    parts = []
    # 개수 많은 순(-cnt), 동률이면 이름 가나다/사전순(nm)으로 정렬해 출력을 결정적으로 만든다.
    for nm, cnt in sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        # 2개 이상이면 "이름×개수", 1개면 이름만 표기.
        parts.append(f"{nm}×{cnt}" if cnt > 1 else nm)
    return "탐지 객체: " + ", ".join(parts)


def yolo_caption(path: str | Path, class_names: Optional[dict] = None) -> str:
    """YOLO 라벨 파일에서 읽을 수 있는 캡션을 만든다(YOLO 가 아니거나 비면 '')."""
    return caption_from_boxes(parse_yolo(path), class_names)


def names_in(boxes: Sequence[Box], class_names: Optional[dict] = None) -> list[str]:
    """박스들에 등장하는 표시 이름을 중복 없이, 등장 순서를 유지하며 반환한다(객체 그래프용)."""
    out: list[str] = []
    seen: set[str] = set()
    for b in boxes:
        nm = _name(class_names, b[0])
        # 처음 본 이름만 추가해 순서를 보존하면서 중복을 제거한다(dict 정렬이 아닌 등장 순).
        if nm not in seen:
            seen.add(nm)
            out.append(nm)
    return out


def scan_class_ids(root: str | Path, image_exts: Sequence[str]) -> Counter:
    """루트 아래 모든 YOLO 사이드카를 훑어 class-id -> 총 박스 수 Counter 를 만든다.

    인덱싱 전에 클래스 이름 편집기를 미리 채우는 용도다(어떤 클래스가 얼마나 많은지
    사용자에게 보여 주고 이름을 붙이게 한다).
    """
    # 순환 import 회피를 위해 함수 내부에서 walker 를 늦게 import.
    from imgsearch.index import walker

    counts: Counter = Counter()
    # 데이터셋 전체를 폴더 단위로 순회하며 각 이미지의 라벨 파일을 파싱한다.
    for _dirpath, images in walker.iter_leaf_folders(root, image_exts):
        for img in images:
            lp = label_path_for(img)
            if lp is None:  # 라벨 파일이 없는 이미지는 건너뛴다.
                continue
            for box in parse_yolo(lp):
                counts[box[0]] += 1  # box[0] == class_id 누적.
    return counts
