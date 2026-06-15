"""YOLO detection-label helpers: parse boxes, summarize as a human caption.

Many datasets ship per-image ``.txt`` files in YOLO format
(``class_id cx cy w h``, normalized). Those are NOT captions — this module turns
them into a readable object summary (using a class-id -> name map when available)
and lets the indexer/pairing avoid treating raw label numbers as text.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

_YOLO_LINE = re.compile(r"^\d+(?:\s+-?\d*\.?\d+){4,}\s*$")

Box = tuple[int, float, float, float, float]  # (class_id, cx, cy, w, h)


def is_yolo_text(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return bool(lines) and all(_YOLO_LINE.match(ln) for ln in lines)


def is_yolo_file(path: str | Path) -> bool:
    try:
        return is_yolo_text(Path(path).read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return False


def label_path_for(image_path: str | Path, text_exts: Sequence[str] = (".txt",)) -> Optional[Path]:
    p = Path(image_path)
    for ext in text_exts:
        cand = p.with_suffix(ext)
        if cand.exists() and cand != p:
            return cand
    return None


def parse_yolo(path: str | Path) -> list[Box]:
    boxes: list[Box] = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return boxes
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) >= 5:
            try:
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(x) for x in parts[1:5])
            except ValueError:
                continue
            boxes.append((cid, cx, cy, w, h))
    return boxes


def class_histogram(boxes: Sequence[Box]) -> Counter:
    return Counter(b[0] for b in boxes)


def _name(class_names: Optional[dict], cid: int) -> str:
    if class_names:
        return str(class_names.get(str(cid)) or class_names.get(cid) or f"#{cid}")
    return f"#{cid}"


def caption_from_boxes(boxes: Sequence[Box], class_names: Optional[dict] = None) -> str:
    if not boxes:
        return ""
    # aggregate by display NAME (several class ids may map to the same name)
    name_counts: Counter = Counter()
    for b in boxes:
        name_counts[_name(class_names, b[0])] += 1
    parts = []
    for nm, cnt in sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        parts.append(f"{nm}×{cnt}" if cnt > 1 else nm)
    return "탐지 객체: " + ", ".join(parts)


def yolo_caption(path: str | Path, class_names: Optional[dict] = None) -> str:
    """Readable caption from a YOLO label file ('' if not a YOLO file / empty)."""
    return caption_from_boxes(parse_yolo(path), class_names)


def names_in(boxes: Sequence[Box], class_names: Optional[dict] = None) -> list[str]:
    """Unique display names present in the boxes, order-stable (for the object graph)."""
    out: list[str] = []
    seen: set[str] = set()
    for b in boxes:
        nm = _name(class_names, b[0])
        if nm not in seen:
            seen.add(nm)
            out.append(nm)
    return out


def scan_class_ids(root: str | Path, image_exts: Sequence[str]) -> Counter:
    """Counter of class-id -> total box count across all YOLO sidecars under root.

    Used to pre-populate the class-name editor before indexing.
    """
    from imgsearch.index import walker

    counts: Counter = Counter()
    for _dirpath, images in walker.iter_leaf_folders(root, image_exts):
        for img in images:
            lp = label_path_for(img)
            if lp is None:
                continue
            for box in parse_yolo(lp):
                counts[box[0]] += 1
    return counts
