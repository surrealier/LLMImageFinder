"""Pair images with their sidecar text (same-stem .txt/.caption/.json, plus folder notes)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

from imgsearch.config import SIDECAR_TEXT_EXTS

_JSON_KEYS = ("caption", "text", "description", "desc", "prompt", "label")
_FOLDER_NOTE_NAMES = ("caption.txt", "readme.txt", "description.txt", "info.txt")


def _read_text_file(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""
    if path.suffix.lower() == ".txt":
        from imgsearch.index.labels import is_yolo_text

        if is_yolo_text(raw):  # YOLO detection labels are not caption text
            return ""
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(data, dict):
            for k in _JSON_KEYS:
                if k in data and isinstance(data[k], str):
                    return data[k].strip()
            return " ".join(str(v) for v in data.values() if isinstance(v, str)).strip()
        return raw
    return raw


def sidecar_for(image_path: str | Path, text_exts: Sequence[str] = SIDECAR_TEXT_EXTS) -> str:
    """Text of a same-stem sidecar file next to the image, if any."""
    p = Path(image_path)
    for ext in text_exts:
        cand = p.with_suffix(ext)
        if cand.exists() and cand != p:
            txt = _read_text_file(cand)
            if txt:
                return txt
    return ""


def folder_sidecar_text(
    dirpath: str | Path,
    image_paths: Sequence[str],
    text_exts: Sequence[str] = SIDECAR_TEXT_EXTS,
    max_items: int = 8,
    max_chars: int = 800,
) -> str:
    """Concatenated, de-duplicated sidecar text for a leaf folder (capped)."""
    parts: list[str] = []
    seen: set[str] = set()

    # folder-level note files first
    d = Path(dirpath)
    for name in _FOLDER_NOTE_NAMES:
        f = d / name
        if f.exists():
            t = _read_text_file(f)
            if t and t not in seen:
                seen.add(t)
                parts.append(t)

    for img in image_paths[:max_items]:
        t = sidecar_for(img, text_exts)
        if t and t not in seen:
            seen.add(t)
            parts.append(t)

    joined = " ".join(" ".join(p.split()) for p in parts)
    return joined[:max_chars]
