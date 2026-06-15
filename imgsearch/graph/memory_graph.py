"""Pure-python in-memory object graph — the default backend (always available).

Holds image->classes and class->images adjacency as plain dict/set, which is enough
for co-occurrence, AND/OR membership, and edge listing on datasets of this scale
(thousands of images). No external dependency, fully deterministic.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Sequence

from imgsearch.graph.base import GraphRecord


class MemoryGraphStore:
    name = "memory"

    def __init__(self) -> None:
        self._img_classes: dict[str, set[str]] = {}
        self._class_images: dict[str, set[str]] = {}
        # build() runs on the GraphBuildWorker thread while reads run on the query
        # thread / GUI thread — serialize so a reader never sees a half-built graph
        self._lock = threading.RLock()

    def build(self, records: Sequence[GraphRecord]) -> None:
        img_classes: dict[str, set[str]] = {}
        class_images: dict[str, set[str]] = {}
        for r in records:
            classes = {c for c in r.classes if c}
            if not classes:
                continue
            img_classes[r.image_path] = classes
            for c in classes:
                class_images.setdefault(c, set()).add(r.image_path)
        # swap in fully-built maps atomically (under the lock) so readers see all-or-nothing
        with self._lock:
            self._img_classes = img_classes
            self._class_images = class_images

    def count(self) -> int:
        with self._lock:
            return len(self._img_classes)

    def class_counts(self) -> dict[str, int]:
        with self._lock:
            return {c: len(imgs) for c, imgs in self._class_images.items()}

    def cooccurring(self, class_name: str, top: int = 20) -> list[tuple[str, int]]:
        co: Counter = Counter()
        with self._lock:
            for img in self._class_images.get(class_name, set()):
                for c in self._img_classes.get(img, ()):  # classes in that image
                    if c != class_name:
                        co[c] += 1
        return co.most_common(top)

    def images_with_all(self, class_names: Sequence[str]) -> list[str]:
        names = [c for c in class_names if c]
        if not names:
            return []
        with self._lock:
            sets = [self._class_images.get(c, set()) for c in names]
            if any(not s for s in sets):  # a requested class is absent -> no image qualifies
                return []
            return sorted(set.intersection(*sets))

    def images_with_any(self, class_names: Sequence[str]) -> list[str]:
        out: set[str] = set()
        with self._lock:
            for c in class_names:
                out |= self._class_images.get(c, set())
        return sorted(out)

    def edges(self, top: int = 50) -> list[tuple[str, str, int]]:
        co: Counter = Counter()
        with self._lock:
            for classes in self._img_classes.values():
                ordered = sorted(classes)
                for i in range(len(ordered)):
                    for j in range(i + 1, len(ordered)):
                        co[(ordered[i], ordered[j])] += 1
        return [(a, b, n) for (a, b), n in co.most_common(top)]

    def clear(self) -> None:
        with self._lock:
            self._img_classes = {}
            self._class_images = {}

    def close(self) -> None:
        pass
