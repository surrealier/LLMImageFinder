"""Build the object graph off the UI thread (walks dataset labels, ~1s for 1k images)."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, Signal, Slot

from imgsearch.config import AppConfig
from imgsearch.graph.base import GraphStore
from imgsearch.graph.builder import build_records


class GraphBuildWorker(QObject):
    finished = Signal(int)  # number of labeled images in the graph
    error = Signal(str)

    def __init__(self, graph: GraphStore, root: str, config: AppConfig) -> None:
        super().__init__()
        self._graph = graph
        self._root = root
        self._cfg = config

    @Slot()
    def run(self) -> None:
        try:
            records = build_records(self._root, self._cfg)
            self._graph.build(records)
            self.finished.emit(self._graph.count())
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
