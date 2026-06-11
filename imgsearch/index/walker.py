"""Walk a dataset root and yield leaf folders (directories that contain images)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Sequence

_SKIP_DIR_NAMES = {".git", "__pycache__", "thumbs", ".thumbs", ".cache", "$recycle.bin"}


def _is_image(name: str, exts: Sequence[str]) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in exts


def list_images(dirpath: str | Path, exts: Sequence[str]) -> list[str]:
    """Sorted absolute paths of image files directly inside ``dirpath``."""
    d = Path(dirpath)
    try:
        entries = os.scandir(d)
    except OSError:
        return []
    out: list[str] = []
    with entries:
        for e in entries:
            try:
                if e.is_file() and _is_image(e.name, exts):
                    out.append(str(Path(e.path)))
            except OSError:
                continue
    out.sort(key=lambda p: p.lower())
    return out


def iter_leaf_folders(
    root: str | Path, exts: Sequence[str]
) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(dirpath, image_paths)`` for every directory that holds >=1 image."""
    root = str(root)
    for dirpath, dirnames, _filenames in os.walk(root):
        # prune hidden / cache dirs in place
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d.lower() not in _SKIP_DIR_NAMES
        ]
        images = list_images(dirpath, exts)
        if images:
            yield str(Path(dirpath)), images
