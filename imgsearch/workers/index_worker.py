"""Long-running indexing on a background thread (progress/log/finished/error + cancel)."""

from __future__ import annotations

import traceback
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal, Slot

from imgsearch.index.indexer import Indexer


class IndexWorker(QObject):
    progress = Signal(object)  # IndexProgress
    log = Signal(str)
    finished = Signal(int)  # records written
    error = Signal(str)

    def __init__(
        self,
        indexer: Indexer,
        root: str,
        full_rebuild: bool = False,
        thumb_cb: Optional[Callable[[str], None]] = None,
        mode: str = "build",  # "build" | "refresh" (caption-only, no re-embedding)
    ) -> None:
        super().__init__()
        self._indexer = indexer
        self._root = root
        self._full = full_rebuild
        self._thumb_cb = thumb_cb or (lambda _p: None)
        self._mode = mode
        self._cancel = False

    @Slot()
    def cancel(self) -> None:
        self._cancel = True

    @Slot()
    def run(self) -> None:
        try:
            if self._mode == "refresh":
                n = self._indexer.refresh_captions(
                    self._root,
                    progress_cb=self.progress.emit,
                    log_cb=self.log.emit,
                    should_cancel=lambda: self._cancel,
                )
            else:
                n = self._indexer.build(
                    self._root,
                    progress_cb=self.progress.emit,
                    log_cb=self.log.emit,
                    should_cancel=lambda: self._cancel,
                    thumb_cb=self._thumb_cb,
                    full_rebuild=self._full,
                )
            self.finished.emit(n)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
