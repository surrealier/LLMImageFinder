"""Run a worker QObject on its own QThread with a correct lifecycle.

Holds references to both the thread and worker (the #1 cause of crashes in this pattern
is letting them get garbage-collected mid-run). The worker must expose ``run()`` plus
terminal signals ``finished`` and ``error``.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread


class ThreadRunner(QObject):
    def __init__(self, worker: QObject, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # workers MUST expose terminal signals or the thread would never quit
        assert hasattr(worker, "finished") and hasattr(worker, "error"), (
            "worker must expose 'finished' and 'error' signals"
        )
        self._worker = worker
        self._thread = QThread()
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)

        for name in ("finished", "error"):
            getattr(worker, name).connect(self._thread.quit)

        self._thread.finished.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

    @property
    def worker(self) -> QObject:
        return self._worker

    def start(self) -> None:
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread.isRunning()

    def wait(self, ms: int = 8000) -> None:
        if self._thread.isRunning():
            self._thread.wait(ms)
