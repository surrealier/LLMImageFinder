"""Load the real backends (jina-clip weights onto the GPU) off the UI thread.

The window shows immediately with a "모델 로딩 중…" status; when this worker
finishes, MainWindow swaps the live backends in on the GUI thread.
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, Signal, Slot

from imgsearch.config import AppConfig
from imgsearch.core.registry import build_backends


class PreloadWorker(QObject):
    finished = Signal(object)  # (embedder, captioner, chat)
    error = Signal(str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._cfg = config

    @Slot()
    def run(self) -> None:
        try:
            # build_backends already falls back to mock + cfg.mark_degraded on
            # failure, so `error` only fires on truly unexpected exceptions
            backends = build_backends(self._cfg)
            self.finished.emit(backends)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
