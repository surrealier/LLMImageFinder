"""Responsive gallery of result tiles with context menu + keyboard actions."""

from __future__ import annotations

import os
from typing import Sequence

from PySide6.QtCore import QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QGuiApplication, QImage, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QAbstractItemView, QListView, QMenu

from imgsearch.core.models import FolderHit
from imgsearch.ui import icons
from imgsearch.ui.gallery_delegate import (
    CAPTION_ROLE,
    FAILED_ROLE,
    FOLDER_ROLE,
    GalleryDelegate,
    HIT_ROLE,
    MATCH_ROLE,
    MEMBERS_ROLE,
    PATH_ROLE,
    PIXMAP_ROLE,
    SCORE_ROLE,
    THUMB_BOX_H,
    THUMB_BOX_W,
    TILE_H,
    TILE_W,
)
from imgsearch.ui.osutil import reveal_in_explorer
from imgsearch.workers.thumb_worker import ThumbSignals, ThumbTask


class ResultsGallery(QListView):
    activated_hit = Signal(object)  # FolderHit (double-click / Enter)
    selected_hit = Signal(object)  # FolderHit | None (selection change)
    similar_requested = Signal(object)  # FolderHit ("비슷한 이미지 검색")

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

    def set_thumb_size(self, size: int) -> None:
        self._thumb_size = int(size)

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
            item.setData(str(getattr(hit, "match", "") or ""), MATCH_ROLE)
            item.setData(hit, HIT_ROLE)
            item.setToolTip(
                f"{hit.caption}\n{hit.image_path}\n점수: {hit.score:.3f}"
                if hit.caption
                else f"{hit.image_path}\n점수: {hit.score:.3f}"
            )
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
        if item is None:
            return
        if image.isNull():
            item.setData(True, FAILED_ROLE)
            return
        pm = QPixmap.fromImage(image)
        # pre-scale ONCE to the delegate's fixed paint box so every repaint
        # (hover/selection/scroll) is a plain blit instead of a smooth rescale
        if pm.width() > THUMB_BOX_W or pm.height() > THUMB_BOX_H:
            pm = pm.scaled(
                THUMB_BOX_W, THUMB_BOX_H, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        item.setData(pm, PIXMAP_ROLE)

    def shutdown(self) -> None:
        """Drain in-flight thumbnail tasks before the view is destroyed."""
        self._gen += 1  # ignore any late deliveries
        self._pool.clear()
        if not self._pool.waitForDone(3000):
            # a decode is wedged (corrupt file / dead network share). Detach the
            # pool so ~QThreadPool doesn't block window destruction forever —
            # leaking one worker thread beats an app that never closes.
            self._pool.setParent(None)

    # --- interaction ---
    def _on_double(self, index) -> None:
        hit = index.data(HIT_ROLE)
        if hit is not None:
            self.activated_hit.emit(hit)

    def _on_current(self, current, _previous) -> None:
        hit = current.data(HIT_ROLE) if current.isValid() else None
        # always emit — a cleared/empty result set must disable the '폴더 열기' button
        self.selected_hit.emit(hit)

    def current_hit(self) -> FolderHit | None:
        idx = self.currentIndex()
        return idx.data(HIT_ROLE) if idx.isValid() else None

    def keyPressEvent(self, event) -> None:  # noqa: N802
        hit = self.current_hit()
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter) and hit is not None:
            self.activated_hit.emit(hit)
            return
        if key == Qt.Key_C and event.modifiers() & Qt.ControlModifier and hit is not None:
            QGuiApplication.clipboard().setText(hit.image_path)
            return
        if key == Qt.Key_E and event.modifiers() & Qt.ControlModifier and hit is not None:
            reveal_in_explorer(hit.image_path)
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        index = self.indexAt(event.pos())
        hit = index.data(HIT_ROLE) if index.isValid() else None
        if hit is None:
            return
        self.setCurrentIndex(index)
        menu = QMenu(self)
        menu.addAction(icons.eye(), "이미지 보기 (Enter)", lambda: self.activated_hit.emit(hit))
        menu.addAction(
            icons.similar(), "비슷한 이미지 검색", lambda: self.similar_requested.emit(hit)
        )
        menu.addSeparator()
        clip = QGuiApplication.clipboard()
        menu.addAction(icons.copy(), "이미지 경로 복사 (Ctrl+C)", lambda: clip.setText(hit.image_path))
        menu.addAction(icons.copy(), "폴더 경로 복사", lambda: clip.setText(hit.folder))
        if hit.caption:
            menu.addAction(icons.copy(), "캡션 복사", lambda: clip.setText(hit.caption))
        menu.addSeparator()
        menu.addAction(
            icons.folder(), "탐색기에서 열기 (Ctrl+E)", lambda: reveal_in_explorer(hit.image_path)
        )
        menu.exec(event.globalPos())
