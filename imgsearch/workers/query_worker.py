"""Run a search off the UI thread so the window never blocks on embedding/LLM calls.

Supports two modes: a plain SearchService.query, or the agentic A2A pipeline whose
per-step trace lines are emitted via the ``trace`` signal (queued to the GUI thread).
"""

from __future__ import annotations

import traceback
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from imgsearch.core.services import SearchService


class QueryWorker(QObject):
    finished = Signal(object)  # QueryResult
    error = Signal(str)
    trace = Signal(str)  # agentic per-step trace line (agentic mode only)

    def __init__(
        self,
        service: SearchService,
        text: str,
        top_k: int,
        agent: Optional[object] = None,  # AgenticSearch when agentic mode is on
    ) -> None:
        super().__init__()
        self._service = service
        self._text = text
        self._k = top_k
        self._agent = agent

    @Slot()
    def run(self) -> None:
        try:
            if self._agent is not None:
                result = self._agent.run(self._text, step_cb=self.trace.emit, k=self._k)
            else:
                result = self._service.query(self._text, self._k)
            self.finished.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
