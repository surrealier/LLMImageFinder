"""Decode/generate thumbnails on a QThreadPool; deliver QImage to the UI thread.

QImage is built off-thread (allowed); the gallery converts it to a QPixmap in its slot
(QPixmap must be created on the GUI thread).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, Signal

from imgsearch.thumbs import ensure_thumb


class ThumbSignals(QObject):
    done = Signal(str, object, int)  # (image_path, QImage, generation)


class ThumbTask(QRunnable):
    def __init__(self, image_path: str, size: int, signals: ThumbSignals, generation: int = 0) -> None:
        super().__init__()
        self.image_path = image_path
        self.size = size
        self.signals = signals
        self.generation = generation
        self.setAutoDelete(True)

    def run(self) -> None:
        from PySide6.QtGui import QImage

        path = ensure_thumb(self.image_path, self.size)
        img = QImage(str(path)) if path else QImage()
        self.signals.done.emit(self.image_path, img, self.generation)
