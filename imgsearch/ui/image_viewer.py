"""Modal lightbox: zoom/pan, ←/→ navigation, YOLO label overlay, hit metadata."""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

from PySide6.QtCore import QEvent, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from imgsearch.index import labels
from imgsearch.ui import icons
from imgsearch.ui.osutil import reveal_in_explorer


def _class_color(cid: int) -> QColor:
    return QColor.fromHsv((int(cid) * 57) % 360, 200, 255)


class ImageViewer(QDialog):
    # remembered across viewer instances within the session
    show_boxes = True

    def __init__(
        self,
        image_paths: Sequence[str],
        start_index: int = 0,
        parent=None,
        class_names: Optional[dict] = None,
        meta_provider: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__(parent)
        self.paths = [p for p in image_paths if p]
        if not self.paths:
            self.paths = [""]
        self.i = max(0, min(start_index, len(self.paths) - 1))
        self._fit_mode = True
        self._class_names = dict(class_names or {})
        self._meta_provider = meta_provider
        self._box_items: list = []

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
        # the viewport consumes wheel events for scrolling once zoomed in —
        # intercept them there so the wheel ALWAYS zooms (see eventFilter)
        self.view.viewport().installEventFilter(self)
        self.pix_item = QGraphicsPixmapItem()
        self.pix_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(self.pix_item)
        root.addWidget(self.view, 1)

        self.meta_label = QLabel("")
        self.meta_label.setWordWrap(True)
        self.meta_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.meta_label.setStyleSheet("color:#9fb2c8; padding:2px 4px;")
        self.meta_label.setVisible(False)
        root.addWidget(self.meta_label)

        bar = QHBoxLayout()
        self.prev_btn = QPushButton(icons.prev(), "")
        self.prev_btn.setToolTip("이전 (←)")
        self.prev_btn.clicked.connect(self.prev)
        bar.addWidget(self.prev_btn)

        self.next_btn = QPushButton(icons.nxt(), "")
        self.next_btn.setToolTip("다음 (→)")
        self.next_btn.clicked.connect(self.next)
        bar.addWidget(self.next_btn)

        self.boxes_btn = QPushButton(icons.tags(), " 라벨")
        self.boxes_btn.setCheckable(True)
        self.boxes_btn.setChecked(ImageViewer.show_boxes)
        self.boxes_btn.setToolTip("YOLO 라벨 박스 표시/숨김 (B)")
        self.boxes_btn.toggled.connect(self._on_boxes_toggled)
        bar.addWidget(self.boxes_btn)

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
        self._refresh_boxes()
        self._refresh_meta(path)

    def _refresh_meta(self, path: str) -> None:
        meta = ""
        if self._meta_provider is not None:
            try:
                meta = self._meta_provider(path) or ""
            except Exception:
                meta = ""
        self.meta_label.setText(meta)
        self.meta_label.setVisible(bool(meta))
        self.meta_label.setToolTip(path)

    # --- YOLO label overlay ---
    def _clear_boxes(self) -> None:
        for item in self._box_items:
            self.scene.removeItem(item)
        self._box_items = []

    def _refresh_boxes(self) -> None:
        self._clear_boxes()
        path = self.paths[self.i]
        pm = self.pix_item.pixmap()
        label_path = labels.label_path_for(path) if path else None
        self.boxes_btn.setEnabled(label_path is not None)
        if not ImageViewer.show_boxes or label_path is None or pm.isNull():
            return
        w, h = pm.width(), pm.height()
        for cid, cx, cy, bw, bh in labels.parse_yolo(label_path):
            # normalized YOLO coords -> displayed-pixmap coords, so the overlay is
            # correct even though _read_bounded may decode at reduced resolution
            rect = QRectF((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h)
            color = _class_color(cid)
            box = QGraphicsRectItem(rect, self.pix_item)
            pen = QPen(color, 2)
            pen.setCosmetic(True)  # constant on-screen width at any zoom
            box.setPen(pen)
            name = str(self._class_names.get(str(cid), f"#{cid}"))
            text = QGraphicsSimpleTextItem(name, box)  # child: removed with its box
            text.setBrush(QBrush(color))
            text.setFlag(QGraphicsSimpleTextItem.ItemIgnoresTransformations)
            text.setPos(rect.x(), max(0.0, rect.y() - 2))
            self._box_items.append(box)

    def _on_boxes_toggled(self, checked: bool) -> None:
        ImageViewer.show_boxes = bool(checked)
        self._refresh_boxes()

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

    def eventFilter(self, obj, event):  # noqa: N802
        # zoom on wheel even when the zoomed-in view would otherwise scroll
        if obj is self.view.viewport() and event.type() == QEvent.Wheel:
            self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
            return True
        return super().eventFilter(obj, event)

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
        elif key == Qt.Key_B:
            self.boxes_btn.toggle()
        else:
            super().keyPressEvent(event)

    def _open_folder(self) -> None:
        if self.paths and self.paths[self.i]:
            reveal_in_explorer(self.paths[self.i])
