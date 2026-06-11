"""Find class IDs in YOLO labels + hunt for any class-name mapping nearby."""

import os
import re
from collections import Counter
from pathlib import Path

ROOT = r"E:\Datasets\Argosphoto\学習데이터\v1.0.0".replace("学習", "학습")
PARENT = r"E:\Datasets\Argosphoto"

YOLO_LINE = re.compile(r"^\d+(\s+-?\d*\.?\d+){4,}\s*$")


def is_yolo(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(YOLO_LINE.match(ln) for ln in lines)


def main() -> None:
    class_ids = Counter()
    non_yolo_txt = []
    n_labels = 0
    for dirpath, _d, files in os.walk(ROOT):
        for f in files:
            if not f.lower().endswith(".txt"):
                continue
            p = Path(dirpath) / f
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if is_yolo(txt):
                n_labels += 1
                for ln in txt.splitlines():
                    ln = ln.strip()
                    if ln:
                        class_ids[ln.split()[0]] += 1
            else:
                non_yolo_txt.append((str(p), txt[:400]))

    print(f"YOLO label files: {n_labels}")
    print("distinct class ids (id: #boxes):", dict(sorted(class_ids.items(), key=lambda x: int(x[0]))))
    print(f"\nnon-YOLO .txt files under v1.0.0: {len(non_yolo_txt)}")
    for p, txt in non_yolo_txt[:10]:
        print(f"   --- {p} ---\n   {txt!r}")

    # top-level listings
    for d in (ROOT, os.path.join(ROOT, "train"), os.path.join(ROOT, "val"), PARENT):
        print(f"\nlisting {d}:")
        try:
            for e in sorted(os.listdir(d)):
                full = os.path.join(d, e)
                tag = "DIR " if os.path.isdir(full) else "FILE"
                print(f"   {tag} {e}")
        except OSError as ex:
            print("   <error>", ex)

    # peek the detection CSV header (might carry class names)
    csv = r"E:\Datasets\Argosphoto\초기데이터\이상행위_데이터셋 촬영본 (2)\samplelist__with_detection_confidence.csv"
    if os.path.exists(csv):
        print(f"\nCSV head: {csv}")
        try:
            with open(csv, encoding="utf-8", errors="ignore") as fh:
                for _ in range(3):
                    print("   ", fh.readline().rstrip()[:300])
        except OSError as ex:
            print("   <error>", ex)


if __name__ == "__main__":
    main()
