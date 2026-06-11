"""Crop YOLO boxes grouped by class id, so each class can be visually identified.

Picks the clearest, most diverse boxes per class (largest non-whole-image boxes,
<=2 per source image), crops with padding, writes JPGs + a manifest.json.
"""

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from imgsearch.config import DEFAULT_IMAGE_EXTS
from imgsearch.index import labels, walker

PER_CLASS = 16
MAX_PER_IMAGE = 2
PAD = 0.02  # near-exact bounding box (tight crop)


def main(root: str) -> None:
    out = Path(tempfile.gettempdir()) / "argos_crops"
    if out.exists():
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # gather candidate (image, box) per class with normalized area + box stats
    cand: dict[int, list[tuple[str, tuple]]] = defaultdict(list)
    stats: dict[int, list[tuple[float, float]]] = defaultdict(list)  # (area, w/h)
    for dirpath, images in walker.iter_leaf_folders(root, DEFAULT_IMAGE_EXTS):
        for img in images:
            lp = labels.label_path_for(img)
            if lp is None:
                continue
            for (cid, cx, cy, w, h) in labels.parse_yolo(lp):
                area = w * h
                stats[cid].append((area, (w / h) if h > 0 else 0.0))
                if area > 0.88:  # skip whole-image boxes (less discriminative)
                    continue
                cand[cid].append((img, (cx, cy, w, h, area)))

    manifest: dict[str, list[str]] = {}
    for cid in sorted(cand):
        items = sorted(cand[cid], key=lambda t: -t[1][4])  # largest first
        chosen = []
        per_img = defaultdict(int)
        for img, box in items:
            if per_img[img] >= MAX_PER_IMAGE:
                continue
            per_img[img] += 1
            chosen.append((img, box))
            if len(chosen) >= PER_CLASS:
                break

        cdir = out / f"class_{cid}"
        cdir.mkdir(parents=True, exist_ok=True)
        saved = []
        for idx, (img, (cx, cy, w, h, _a)) in enumerate(chosen):
            try:
                with Image.open(img) as im:
                    im = ImageOps.exif_transpose(im.convert("RGB"))
                    W, H = im.size
                    bw, bh = w * W, h * H
                    x1 = max(0, int((cx - w / 2) * W - bw * PAD))
                    y1 = max(0, int((cy - h / 2) * H - bh * PAD))
                    x2 = min(W, int((cx + w / 2) * W + bw * PAD))
                    y2 = min(H, int((cy + h / 2) * H + bh * PAD))
                    crop = im.crop((x1, y1, x2, y2))
                    crop.thumbnail((256, 256))
                    p = cdir / f"c{idx:02d}.jpg"
                    crop.save(p, "JPEG", quality=90)
                    saved.append(str(p))
            except Exception as e:
                print(f"  crop error cls={cid} {img}: {e}")
        manifest[str(cid)] = saved
        print(f"class {cid}: {len(saved)} crops")

    import statistics

    print("\nper-class box stats (all boxes, normalized):")
    for cid in sorted(stats):
        areas = [a for a, _ in stats[cid]]
        aspects = [r for _, r in stats[cid] if r > 0]
        med_a = statistics.median(areas) * 100
        med_r = statistics.median(aspects) if aspects else 0
        print(f"  class {cid}: n={len(areas):5d}  median_area={med_a:5.2f}%  median_w/h={med_r:.2f}")

    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MANIFEST:", mpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:\Datasets\Argosphoto\학습데이터\v1.0.0")
