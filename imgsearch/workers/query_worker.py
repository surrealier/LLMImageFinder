"""Run a search off the UI thread so the window never blocks on embedding/LLM calls."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, Signal, Slot

from imgsearch.core.services import SearchService


class QueryWorker(QObject):
    finished = Signal(object)  # QueryResult
    error = Signal(str)

    def __init__(self, service: SearchService, text: str, top_k: int) -> None:
        super().__init__()
        self._service = service
        self._text = text
        self._k = top_k

    @Slot()
    def run(self) -> None:
        try:
            result = self._service.query(self._text, self._k)
            self.finished.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
