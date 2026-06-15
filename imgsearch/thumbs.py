"""디스크 기반 썸네일 캐시(PIL만 사용 — 어떤 스레드에서 호출해도 안전)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from imgsearch.paths import app_paths


def _thumbs_root() -> Path:
    """썸네일 캐시 루트 디렉터리를 보장(생성)하고 반환한다."""
    d = app_paths().thumbs_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def thumb_path(image_path: str, size: int) -> Path:
    """주어진 (원본 경로, 크기)에 대응하는 캐시 파일 경로를 계산한다(아직 생성하지 않음).

    캐시 키에 수정 시각(mtime)을 포함하므로, 원본이 바뀌면 자연스럽게 다른 경로가 되어
    오래된 썸네일이 재사용되지 않는다(자동 무효화).
    """
    try:
        mt = os.path.getmtime(image_path)
    except OSError:
        # 경로가 없거나 접근 불가하면 0으로 두고, 생성 단계에서 실패하도록 둔다.
        mt = 0.0
    # 절대경로 + mtime + 크기를 합쳐 키로 삼아 충돌과 오재사용을 피한다.
    key = f"{os.path.abspath(image_path)}|{mt}|{size}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return _thumbs_root() / f"{h}.jpg"


def ensure_thumb(image_path: str, size: int = 256) -> Path | None:
    """size 이하 크기의 JPEG 썸네일을 캐시에서 반환하되, 없으면 생성한다.

    스레드별 임시 파일에 쓴 뒤 원자적 교체(os.replace)로 마무리한다: 인덱싱 시점의
    예열(pre-warm) 풀과 갤러리의 온디맨드 작업이라는 두 풀이 같은 썸네일을 동시에
    생성할 수 있는데, 쓰다 만(torn) 파일이 캐시에 남는 일이 절대 없도록 하기 위함이다.
    """
    out = thumb_path(image_path, size)
    try:
        # 이미 유효한(크기>0) 캐시가 있으면 재생성 없이 그대로 사용.
        if out.exists() and out.stat().st_size > 0:
            return out
    except OSError:
        pass
    import threading

    # 임시 파일명에 PID와 스레드 id를 넣어 동시 생성 시 서로 충돌하지 않게 한다.
    tmp = out.with_name(f"{out.stem}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        from PIL import Image, ImageOps

        with Image.open(image_path) as im:
            # EXIF 회전 정보를 적용해 세워진/뉘인 사진의 방향을 바로잡고 RGB로 통일.
            im = ImageOps.exif_transpose(im.convert("RGB"))
            # thumbnail은 종횡비를 유지하며 (size, size) 박스 안에 들어가도록 축소.
            im.thumbnail((size, size))
            im.save(tmp, "JPEG", quality=85)
        # 완성된 임시 파일을 원자적으로 최종 경로로 옮긴다(부분 파일 노출 방지).
        os.replace(tmp, out)
        return out
    except Exception:
        # 실패 시 남은 임시 파일을 정리하고 None을 반환(호출부에서 빈 이미지로 처리).
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None
