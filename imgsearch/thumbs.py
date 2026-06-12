"""On-disk thumbnail cache (PIL only — safe to call from any thread)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from imgsearch.paths import app_paths


def _thumbs_root() -> Path:
    d = app_paths().thumbs_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def thumb_path(image_path: str, size: int) -> Path:
    try:
        mt = os.path.getmtime(image_path)
    except OSError:
        mt = 0.0
    key = f"{os.path.abspath(image_path)}|{mt}|{size}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return _thumbs_root() / f"{h}.jpg"


def ensure_thumb(image_path: str, size: int = 256) -> Path | None:
    """Return a cached <=size JPEG thumbnail, generating it if needed.

    Writes via a per-thread temp file + atomic replace: two pools (index-time
    pre-warm and the gallery's on-demand tasks) may generate the same thumb
    concurrently, and a torn write must never end up in the cache.
    """
    out = thumb_path(image_path, size)
    try:
        if out.exists() and out.stat().st_size > 0:
            return out
    except OSError:
        pass
    import threading

    tmp = out.with_name(f"{out.stem}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        from PIL import Image, ImageOps

        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im.convert("RGB"))
            im.thumbnail((size, size))
            im.save(tmp, "JPEG", quality=85)
        os.replace(tmp, out)
        return out
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None
