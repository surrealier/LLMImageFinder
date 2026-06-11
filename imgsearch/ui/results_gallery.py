"""Responsive gallery of representative tiles (one per matching leaf folder)."""

from __future__ import annotations

import os
from typing import Sequence

from PySide6.QtCore import QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QAbstractItemView, QListView

from imgsearch.core.models import FolderHit
from imgsearch.ui.gallery_delegate import (
    CAPTION_ROLE,
    FOLDER_ROLE,
    GalleryDelegate,
    HIT_ROLE,
    MEMBERS_ROLE,
    PATH_ROLE,
    PIXMAP_ROLE,
    SCORE_ROLE,
    TILE_H,
    TILE_W,
)
from imgsearch.workers.thumb_worker import ThumbSignals, ThumbTask


class ResultsGallery(QListView):
    activated_hit = Signal(object)  # FolderHit (double-click)
    selected_hit = Signal(object)  # FolderHit (selection change)

    def __init__(self, thumb_size: int = 256, parent=None) -> None:
        super().__init__(parent)
        self._thumb_size = thumb_size
        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setMovement(QListView.Static)
        self.setWrapping(True)
        self.setUniformItemSizes(True)
        self.setSpacing(8)
        self.setGridSize(QSize(TILE_W, TILE_H))
        self.setMouseTracking(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self.setItemDelegate(GalleryDelegate(self))

        self._pool = QThreadPool(self)  # gallery-owned so it drains on close
        self._pool.setMaxThreadCount(max(2, min(8, (os.cpu_count() or 4) - 1)))
        self._sig = ThumbSignals()
        self._sig.done.connect(self._on_thumb)
        self._by_path: dict[str, QStandardItem] = {}
        self._gen = 0  # bumped each query; stale-task deliveries are ignored

        self.doubleClicked.connect(self._on_double)
        self.selectionModel().currentChanged.connect(self._on_current)

    def set_results(self, hits: Sequence[FolderHit]) -> None:
        self._model.clear()
        self._by_path.clear()
        self._gen += 1
        gen = self._gen
        for hit in hits:
            item = QStandardItem()
            item.setEditable(False)
            item.setData(hit.caption, CAPTION_ROLE)
            item.setData(hit.folder, FOLDER_ROLE)
            item.setData(hit.image_path, PATH_ROLE)
            item.setData(float(hit.score), SCORE_ROLE)
            item.setData(int(hit.member_count), MEMBERS_ROLE)
            item.setData(hit, HIT_ROLE)
            self._model.appendRow(item)
            if hit.image_path:
                self._by_path[hit.image_path] = item
                self._pool.start(ThumbTask(hit.image_path, self._thumb_size, self._sig, gen))
        if hits:
            self.setCurrentIndex(self._model.index(0, 0))

    def _on_thumb(self, image_path: str, image: QImage, generation: int) -> None:
        if generation != self._gen:  # stale task from a previous query/size
            return
        item = self._by_path.get(image_path)
        if item is not None and not image.isNull():
            item.setData(QPixmap.fromImage(image), PIXMAP_ROLE)

    def shutdown(self) -> None:
        """Drain in-flight thumbnail tasks before the view is destroyed."""
        self._gen += 1  # ignore any late deliveries
        self._pool.clear()
        self._pool.waitForDone(3000)

    def _on_double(self, index) -> None:
        hit = index.data(HIT_ROLE)
        if hit is not None:
            self.activated_hit.emit(hit)

    def _on_current(self, current, _previous) -> None:
        if current.isValid():
            hit = current.data(HIT_ROLE)
            if hit is not None:
                self.selected_hit.emit(hit)

    def current_hit(self) -> FolderHit | None:
        idx = self.currentIndex()
        return idx.data(HIT_ROLE) if idx.isValid() else None
