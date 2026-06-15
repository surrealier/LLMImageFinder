"""Build GraphRecords by walking the dataset's YOLO labels.

The graph's source of truth is the dataset labels on disk (not the chroma documents),
so it reflects the actual annotations. Class ids map to display names via
``cfg.class_names`` exactly as captions do, keeping one naming convention.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable

from imgsearch.config import AppConfig
from imgsearch.graph.base import GraphRecord
from imgsearch.index import labels, walker

ProgressCb = Callable[[int, str], None]


def iter_graph_records(root: str, cfg: AppConfig) -> Iterable[GraphRecord]:
    """Yield a GraphRecord for every image that has YOLO label boxes."""
    if not root or not os.path.isdir(root):
        return
    for dirpath, images in walker.iter_leaf_folders(root, cfg.image_exts):
        for img in images:
            lp = labels.label_path_for(img)
            if lp is None:
                continue
            boxes = labels.parse_yolo(lp)
            if not boxes:
                continue
            names = labels.names_in(boxes, cfg.class_names)
            if names:
                yield GraphRecord(image_path=img, folder=dirpath, classes=names)


def build_records(root: str, cfg: AppConfig) -> list[GraphRecord]:
    return list(iter_graph_records(root, cfg))
