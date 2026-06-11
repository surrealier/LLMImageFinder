"""Inspect a real dataset: structure, scale, pairing, leaf-folder semantics."""

import os
import sys
from collections import Counter
from pathlib import Path

from imgsearch.config import DEFAULT_IMAGE_EXTS, SIDECAR_TEXT_EXTS

IMG = set(DEFAULT_IMAGE_EXTS)
TXT = set(SIDECAR_TEXT_EXTS)


def main(root: str) -> None:
    root = root.rstrip("\\/")
    if not os.path.isdir(root):
        print("NOT A DIRECTORY:", root)
        return
    print("ROOT:", root)

    total_dirs = 0
    leaf_folders = 0          # dirs directly containing >=1 image
    total_images = 0
    img_ext = Counter()
    txt_ext = Counter()
    depths = Counter()
    members_per_leaf = []
    dirs_with_img_and_subimg = 0
    sample_leaf = []
    sample_images = []
    sidecar_hits = 0
    sidecar_miss = 0

    # precompute which dirs contain images (for "image-bearing subdir" check)
    has_img: dict[str, bool] = {}
    walk = list(os.walk(root))
    for dirpath, _dirs, files in walk:
        has_img[dirpath] = any(os.path.splitext(f)[1].lower() in IMG for f in files)

    for dirpath, dirnames, files in walk:
        total_dirs += 1
        imgs = [f for f in files if os.path.splitext(f)[1].lower() in IMG]
        txts = [f for f in files if os.path.splitext(f)[1].lower() in TXT]
        for f in imgs:
            img_ext[os.path.splitext(f)[1].lower()] += 1
        for f in txts:
            txt_ext[os.path.splitext(f)[1].lower()] += 1
        total_images += len(imgs)
        if imgs:
            leaf_folders += 1
            members_per_leaf.append(len(imgs))
            rel = os.path.relpath(dirpath, root)
            depths[rel.count(os.sep)] += 1
            if len(sample_leaf) < 12:
                sample_leaf.append((rel, len(imgs), len(txts)))
            if len(sample_images) < 5:
                sample_images.append(os.path.join(dirpath, imgs[0]))
            # image-bearing subdir?
            if any(has_img.get(os.path.join(dirpath, d), False) for d in dirnames):
                dirs_with_img_and_subimg += 1
            # sidecar pairing on first few images
            for f in imgs[:3]:
                stem = os.path.splitext(f)[0]
                if any((Path(dirpath) / (stem + e)).exists() for e in TXT):
                    sidecar_hits += 1
                else:
                    sidecar_miss += 1

    print(f"\ntotal_dirs={total_dirs}  leaf_folders(with images)={leaf_folders}  total_images={total_images}")
    print("image exts:", dict(img_ext))
    print("sidecar text exts:", dict(txt_ext))
    if members_per_leaf:
        import statistics
        print(f"images/leaf: min={min(members_per_leaf)} max={max(members_per_leaf)} "
              f"mean={statistics.mean(members_per_leaf):.1f} median={statistics.median(members_per_leaf)}")
    print("leaf depth (from root) histogram:", dict(sorted(depths.items())))
    print(f"leaf dirs that ALSO have image-bearing subdirs: {dirs_with_img_and_subimg}")
    print(f"sidecar pairing (first 3 imgs/leaf sampled): hits={sidecar_hits} miss={sidecar_miss}")

    print("\nsample leaf folders (relpath, #imgs, #txts):")
    for rel, ni, nt in sample_leaf:
        print(f"   {ni:4d} img {nt:4d} txt   {rel}")

    print("\nsample image sizes:")
    for p in sample_images:
        try:
            from PIL import Image
            with Image.open(p) as im:
                print(f"   {im.size}  {im.mode}  {os.path.getsize(p)//1024}KB  {os.path.basename(p)}")
        except Exception as e:
            print(f"   <error {e}> {p}")

    # peek a sidecar text file to see if it's a caption or annotation labels
    print("\nsample sidecar content (first match):")
    shown = 0
    for dirpath, _dirs, files in walk:
        if shown >= 2:
            break
        for f in files:
            if os.path.splitext(f)[1].lower() in TXT:
                p = Path(dirpath) / f
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")[:300]
                    print(f"   --- {f} ---\n   {txt!r}")
                    shown += 1
                except Exception as e:
                    print(f"   <error {e}>")
                break


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:\Datasets\Argosphoto\학습데이터\v1.0.0")
