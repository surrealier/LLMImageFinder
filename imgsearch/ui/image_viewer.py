"""Modal lightbox: zoom/pan + ←/→ navigation through a folder's member images."""

from __future__ import annotations

import os
from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QDialog,
)

from imgsearch.ui import icons
from imgsearch.ui.osutil import reveal_in_explorer


class ImageViewer(QDialog):
    def __init__(self, image_paths: Sequence[str], start_index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.paths = [p for p in image_paths if p]
        if not self.paths:
            self.paths = [""]
        self.i = max(0, min(start_index, len(self.paths) - 1))
        self._fit_mode = True

        self.setWindowTitle("이미지 보기")
        self.resize(960, 720)
        self.setStyleSheet("QDialog{background:#0f141c;}")

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setBackgroundBrush(Qt.black)
        self.view.setAlignment(Qt.AlignCenter)
        self.pix_item = QGraphicsPixmapItem()
        self.pix_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(self.pix_item)
        root.addWidget(self.view, 1)

        bar = QHBoxLayout()
        self.prev_btn = QPushButton(icons.prev(), "")
        self.prev_btn.setToolTip("이전 (←)")
        self.prev_btn.clicked.connect(self.prev)
        bar.addWidget(self.prev_btn)

        self.next_btn = QPushButton(icons.nxt(), "")
        self.next_btn.setToolTip("다음 (→)")
        self.next_btn.clicked.connect(self.next)
        bar.addWidget(self.next_btn)

        bar.addStretch(1)
        self.counter = QLabel("")
        self.counter.setStyleSheet("color:#cbd5e1;")
        bar.addWidget(self.counter)
        bar.addStretch(1)

        zin = QPushButton(icons.zoom_in(), "")
        zin.setToolTip("확대 (+)")
        zin.clicked.connect(lambda: self.zoom(1.25))
        bar.addWidget(zin)
        zout = QPushButton(icons.zoom_out(), "")
        zout.setToolTip("축소 (-)")
        zout.clicked.connect(lambda: self.zoom(0.8))
        bar.addWidget(zout)
        fit = QPushButton(icons.fit(), "")
        fit.setToolTip("창에 맞춤 (F)")
        fit.clicked.connect(self.fit)
        bar.addWidget(fit)

        openf = QPushButton(icons.folder(), " 폴더 열기")
        openf.setToolTip("탐색기에서 이 이미지 위치 열기")
        openf.clicked.connect(self._open_folder)
        bar.addWidget(openf)
        root.addLayout(bar)

        self._load()

    # --- navigation ---
    @staticmethod
    def _read_bounded(path: str, max_side: int = 2560) -> QPixmap:
        """Decode at most max_side on the long edge (bounds UI-thread decode/memory)."""
        if not path:
            return QPixmap()
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid() and max(size.width(), size.height()) > max_side:
            reader.setScaledSize(size.scaled(max_side, max_side, Qt.KeepAspectRatio))
        img = reader.read()
        return QPixmap.fromImage(img) if not img.isNull() else QPixmap(path)

    def _load(self) -> None:
        path = self.paths[self.i]
        pm = self._read_bounded(path)
        self.pix_item.setPixmap(pm)
        self.scene.setSceneRect(self.pix_item.boundingRect())
        self._fit_mode = True
        self.fit()
        self.counter.setText(f"{self.i + 1} / {len(self.paths)}   —   {os.path.basename(path)}")
        self.prev_btn.setEnabled(len(self.paths) > 1)
        self.next_btn.setEnabled(len(self.paths) > 1)

    def show_index(self, i: int) -> None:
        if not self.paths:
            return
        self.i = i % len(self.paths)
        self._load()

    def prev(self) -> None:
        self.show_index(self.i - 1)

    def next(self) -> None:
        self.show_index(self.i + 1)

    # --- zoom ---
    def fit(self) -> None:
        if not self.pix_item.pixmap().isNull():
            self.view.resetTransform()
            self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)
            self._fit_mode = True

    def zoom(self, factor: float) -> None:
        self._fit_mode = False
        self.view.scale(factor, factor)

    def wheelEvent(self, event) -> None:  # noqa: N802
        self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key_Right, Qt.Key_Space, Qt.Key_Down):
            self.next()
        elif key in (Qt.Key_Left, Qt.Key_Up):
            self.prev()
        elif key == Qt.Key_Escape:
            self.accept()
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom(1.25)
        elif key == Qt.Key_Minus:
            self.zoom(0.8)
        elif key == Qt.Key_F:
            self.fit()
        else:
            super().keyPressEvent(event)

    def _open_folder(self) -> None:
        if self.paths and self.paths[self.i]:
            reveal_in_explorer(self.paths[self.i])
