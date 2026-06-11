"""Generate a small synthetic image-text dataset for demos and tests.

Creates a nested folder tree where each leaf folder holds several *near-duplicate*
frames of one scene plus per-image Korean sidecar ``.txt`` captions. Folder names use
English concept words so the deterministic mock backend can match Korean queries
(via the KO->EN lexicon in :mod:`imgsearch.koutil`).
"""

from __future__ import annotations

import random
from pathlib import Path

# (category, scene_folder_name, builder_key, korean_caption)
_SCENES = [
    ("outdoor_night", "motorcycle_parked_night", "motorcycle_night",
     "밤에 도로변에 오토바이가 주차되어 있는 야간 장면"),
    ("outdoor_night", "street_people_night", "people_night",
     "야간 거리에 사람들이 지나다니는 모습"),
    ("street_day", "people_crosswalk_day", "people_day",
     "낮에 대로변 횡단보도를 사람들이 건너는 장면"),
    ("street_day", "cars_traffic_day", "cars_day",
     "주간 도로 위 자동차 차량 통행 장면"),
    ("emergency", "fire_smoke_building", "fire",
     "건물에서 불과 연기가 발생한 화재 장면"),
    ("emergency", "smoke_only_field", "smoke",
     "들판에 연기가 자욱하게 퍼진 장면"),
    ("parking", "cars_parking_lot_day", "cars_day",
     "주간 주차장에 자동차들이 주차되어 있는 장면"),
    ("nature", "beach_sea_day", "beach",
     "맑은 낮 바다 해변 풍경"),
]


def _draw_scene(key: str, w: int, h: int, jitter: random.Random):
    from PIL import Image, ImageDraw

    night = key.endswith("_night") or key in {"fire", "smoke"} and False
    if key.endswith("_night"):
        bg = (14, 18, 32)
    elif key == "beach":
        bg = (120, 190, 225)
    else:
        bg = (175, 198, 214)
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    ox, oy = jitter.randint(-8, 8), jitter.randint(-6, 6)

    def rect(x0, y0, x1, y1, fill):
        d.rectangle([x0 + ox, y0 + oy, x1 + ox, y1 + oy], fill=fill)

    def ell(x0, y0, x1, y1, fill):
        d.ellipse([x0 + ox, y0 + oy, x1 + ox, y1 + oy], fill=fill)

    # ground / road
    if key != "beach":
        rect(0, h - 70, w, h, (45, 45, 52) if key.endswith("_night") else (95, 95, 105))
    else:
        rect(0, h - 70, w, h, (210, 200, 150))

    if "motorcycle" in key:
        rect(110, 120, 210, 150, (30, 30, 38))           # body
        ell(108, 150, 148, 190, (10, 10, 10))            # wheel
        ell(178, 150, 218, 190, (10, 10, 10))            # wheel
        d.line([130 + ox, 122 + oy, 120 + ox, 100 + oy], fill=(60, 60, 70), width=4)
    if "people" in key:
        for i in range(jitter.randint(3, 5)):
            x = 40 + i * 55 + jitter.randint(-6, 6)
            ell(x, 150, x + 18, 200, (40, 60, 110))      # body
            ell(x + 2, 132, x + 16, 150, (225, 200, 170))  # head
    if "cars" in key:
        for i in range(2):
            x = 60 + i * 130
            rect(x, 140, x + 90, 180, (150, 40, 40) if i == 0 else (40, 70, 150))
            rect(x + 12, 122, x + 78, 142, (190, 210, 230))
    if key == "fire":
        rect(120, 90, 200, 180, (70, 60, 55))            # building
        d.polygon([(150 + ox, 170 + oy), (130 + ox, 110 + oy), (160 + ox, 130 + oy),
                   (150 + ox, 80 + oy), (175 + ox, 125 + oy), (180 + ox, 165 + oy)],
                  fill=(235, 120, 20))
        ell(140, 60, 210, 110, (120, 120, 125))          # smoke
    if key == "smoke":
        for i in range(4):
            ell(70 + i * 45, 70 + jitter.randint(-10, 10), 150 + i * 45, 150,
                (140, 140, 145))
    if key == "beach":
        ell(230, 30, 290, 90, (250, 240, 120))           # sun
    return img


def generate_sample_dataset(
    root: str | Path, per_scene: int = 5, seed: int = 7, size=(320, 240)
) -> Path:
    """Build the synthetic dataset under ``root`` and return its path."""
    root = Path(root)
    w, h = size
    for category, scene, key, caption in _SCENES:
        leaf = root / category / scene
        leaf.mkdir(parents=True, exist_ok=True)
        rng = random.Random(f"{seed}:{scene}")
        for i in range(per_scene):
            img = _draw_scene(key, w, h, rng)
            stem = f"{scene}_{i:03d}"
            img.save(leaf / f"{stem}.jpg", quality=88)
            (leaf / f"{stem}.txt").write_text(
                f"{caption} (프레임 {i + 1}/{per_scene})", encoding="utf-8"
            )
    return root


if __name__ == "__main__":  # pragma: no cover
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "sample_dataset"
    p = generate_sample_dataset(out)
    print(f"Sample dataset created at: {p.resolve()}")
