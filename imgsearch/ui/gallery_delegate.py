"""Custom-painted gallery tile: thumbnail + caption + folder name + score badge."""

from __future__ import annotations

import os

from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

# item data roles
PIXMAP_ROLE = Qt.UserRole + 1
CAPTION_ROLE = Qt.UserRole + 2
FOLDER_ROLE = Qt.UserRole + 3
SCORE_ROLE = Qt.UserRole + 4
MEMBERS_ROLE = Qt.UserRole + 5
HIT_ROLE = Qt.UserRole + 6
PATH_ROLE = Qt.UserRole + 7
FAILED_ROLE = Qt.UserRole + 8  # thumbnail could not be generated (file gone/corrupt)

TILE_W = 224
TILE_H = 250
_THUMB_H = 176
_PAD = 8
# the fixed pixel box a thumbnail is painted into (tile rect is adjusted by ±5):
# pixmaps are pre-scaled to this size at delivery so paint() is a plain blit
THUMB_BOX_W = TILE_W - 10 - 2 * _PAD
THUMB_BOX_H = _THUMB_H


class GalleryDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return QSize(TILE_W, TILE_H)

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: N802
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        rect = option.rect.adjusted(5, 5, -5, -5)
        selected = bool(option.state & QStyle.State_Selected)
        hovered = bool(option.state & QStyle.State_MouseOver)

        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 9, 9)
        painter.fillPath(path, QColor("#222a38" if (selected or hovered) else "#1b212c"))
        painter.setPen(QPen(QColor("#3b82f6") if selected else QColor("#2b3342"), 2 if selected else 1))
        painter.drawPath(path)

        # thumbnail
        thumb = QRect(rect.x() + _PAD, rect.y() + _PAD, rect.width() - 2 * _PAD, _THUMB_H)
        pm = index.data(PIXMAP_ROLE)
        if isinstance(pm, QPixmap) and not pm.isNull():
            if pm.width() > thumb.width() or pm.height() > thumb.height():
                pm = pm.scaled(thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = thumb.x() + (thumb.width() - pm.width()) // 2
            y = thumb.y() + (thumb.height() - pm.height()) // 2
            painter.drawPixmap(x, y, pm)
        elif index.data(FAILED_ROLE):
            painter.setPen(QColor("#f59e0b"))
            painter.drawText(thumb, Qt.AlignCenter, "썸네일 표시 불가\n(파일 확인 필요)")
        else:
            painter.setPen(QColor("#5b6675"))
            painter.drawText(thumb, Qt.AlignCenter, "이미지 로딩…")

        fm = option.fontMetrics
        inner_w = rect.width() - 2 * _PAD

        # caption (1 line, elided)
        cap = str(index.data(CAPTION_ROLE) or "")
        y_cap = thumb.bottom() + 6 + fm.ascent()
        painter.setPen(QColor("#e5e7eb"))
        painter.drawText(rect.x() + _PAD, y_cap, fm.elidedText(cap, Qt.ElideRight, inner_w))

        # folder name (elided middle)
        name = os.path.basename(str(index.data(FOLDER_ROLE) or "").rstrip("/\\"))
        members = index.data(MEMBERS_ROLE)
        if members and int(members) > 1:
            name = f"{name}  ·  {int(members)}장"
        y_name = y_cap + fm.height()
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(rect.x() + _PAD, y_name, fm.elidedText(name, Qt.ElideMiddle, inner_w))

        # score badge (top-right of thumbnail)
        score = index.data(SCORE_ROLE)
        if score is not None:
            txt = f"{float(score):.2f}"
            bw = fm.horizontalAdvance(txt) + 12
            bh = fm.height() + 4
            badge = QRect(thumb.right() - bw, thumb.y() + 4, bw, bh)
            bpath = QPainterPath()
            bpath.addRoundedRect(QRectF(badge), 6, 6)
            painter.fillPath(bpath, QColor(0, 0, 0, 160))
            painter.setPen(QColor("#7dd3fc"))
            painter.drawText(badge, Qt.AlignCenter, txt)

        painter.restore()
