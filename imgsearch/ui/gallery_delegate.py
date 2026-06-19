"""커스텀으로 직접 그리는 갤러리 타일: 썸네일 + 캡션 + 폴더 이름 + 점수 배지.

기본 위젯을 얹는 대신 QStyledItemDelegate.paint를 직접 구현해 한 타일을 한 번에
그린다(수천 개 결과를 가볍게 렌더링하기 위함). 각 타일이 보여줄 데이터는 모델의
사용자 정의 롤(아래 *_ROLE 상수)로 전달받는다.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

# 모델 아이템에 값을 실어 나르는 사용자 정의 데이터 롤. Qt.UserRole 이후의 정수에
# 1씩 더해 충돌 없이 고유 키를 만든다(ResultsGallery.set_results가 이 키로 값을 넣는다).
PIXMAP_ROLE = Qt.UserRole + 1     # 미리 스케일된 썸네일 QPixmap
CAPTION_ROLE = Qt.UserRole + 2    # 캡션 문자열(한 줄, 말줄임 표시)
FOLDER_ROLE = Qt.UserRole + 3     # 리프 폴더 경로(폴더명 표시에 사용)
SCORE_ROLE = Qt.UserRole + 4      # 코사인 유사도(점수 배지)
MEMBERS_ROLE = Qt.UserRole + 5    # 폴더 단위일 때 멤버 이미지 수
HIT_ROLE = Qt.UserRole + 6        # 원본 FolderHit 객체(더블클릭/선택 시 그대로 전달)
PATH_ROLE = Qt.UserRole + 7       # 대표 이미지 경로
FAILED_ROLE = Qt.UserRole + 8     # 썸네일 생성 실패(파일 없음/손상)
MATCH_ROLE = Qt.UserRole + 9      # 매칭 출처: ""|"vector"|"keyword"|"both"|"graph"

# 매칭 출처 배지의 (표시 문자, 색). 어느 검색 팔이 이 결과를 올렸는지 한눈에 보여준다.
_MATCH_BADGE = {
    "vector": ("V", "#60a5fa"),    # 벡터(의미) 검색
    "keyword": ("K", "#fbbf24"),   # 키워드(BM25) 검색
    "both": ("V+K", "#34d399"),    # 두 팔 모두에서 매칭(하이브리드)
    "graph": ("G", "#a78bfa"),     # 객체 그래프 멤버십으로 들어온 결과
}

# 타일 한 칸의 고정 크기와 내부 여백. 격자 레이아웃을 빠르게 하려고 모든 타일을 같은 크기로 둔다.
TILE_W = 224
TILE_H = 250
_THUMB_H = 176   # 타일 안에서 썸네일이 차지하는 높이
_PAD = 8         # 타일 안쪽 여백
# 썸네일이 실제로 그려지는 고정 픽셀 박스(타일 rect는 그리기 직전 ±5로 줄여 둔다).
# 썸네일은 전달 시점(ResultsGallery._on_thumb)에 이 크기로 한 번만 스케일되므로,
# paint()는 매번 리스케일 없이 단순 blit(복사)만 하면 된다 → 스크롤/호버가 가볍다.
THUMB_BOX_W = TILE_W - 10 - 2 * _PAD
THUMB_BOX_H = _THUMB_H


class GalleryDelegate(QStyledItemDelegate):
    """결과 타일 하나를 직접 그리는 델리게이트(썸네일·캡션·폴더명·점수/출처 배지)."""

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        """모든 타일은 같은 고정 크기(setUniformItemSizes와 짝을 이뤄 레이아웃을 빠르게)."""
        return QSize(TILE_W, TILE_H)

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: N802
        """타일 한 칸을 그린다 — 배경 카드 → 썸네일 → 캡션 → 폴더명 → 점수/출처 배지 순."""
        # save/restore로 이 타일에만 적용할 렌더 상태를 격리한다(다른 타일에 영향 X).
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 타일 경계에서 5px 안쪽을 실제 카드 영역으로 쓴다(타일 사이 간격 확보).
        rect = option.rect.adjusted(5, 5, -5, -5)
        selected = bool(option.state & QStyle.State_Selected)
        hovered = bool(option.state & QStyle.State_MouseOver)

        # 둥근 모서리 카드 배경. 선택/호버면 약간 밝게, 선택이면 파란 테두리를 굵게.
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 9, 9)
        painter.fillPath(path, QColor("#222a38" if (selected or hovered) else "#1b212c"))
        painter.setPen(QPen(QColor("#3b82f6") if selected else QColor("#2b3342"), 2 if selected else 1))
        painter.drawPath(path)

        # --- 썸네일 영역 ---
        thumb = QRect(rect.x() + _PAD, rect.y() + _PAD, rect.width() - 2 * _PAD, _THUMB_H)
        pm = index.data(PIXMAP_ROLE)
        if isinstance(pm, QPixmap) and not pm.isNull():
            # 보통은 이미 전달 시점에 박스 크기로 스케일돼 있어 이 분기는 거의 타지 않는다.
            # (혹시 박스보다 크면 비율 유지로 한 번 줄인다.)
            if pm.width() > thumb.width() or pm.height() > thumb.height():
                pm = pm.scaled(thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # 썸네일을 박스 중앙에 배치(가로/세로 남는 공간을 반씩 나눠 offset).
            x = thumb.x() + (thumb.width() - pm.width()) // 2
            y = thumb.y() + (thumb.height() - pm.height()) // 2
            painter.drawPixmap(x, y, pm)
        elif index.data(FAILED_ROLE):
            # 썸네일 생성 실패(파일 이동/손상) — '영원히 로딩 중'으로 오해하지 않도록 명시.
            painter.setPen(QColor("#f59e0b"))
            painter.drawText(thumb, Qt.AlignCenter, "Thumbnail unavailable\n(check the file)")
        else:
            # 아직 디코딩 전(백그라운드 작업 대기) — 곧 _on_thumb가 PIXMAP_ROLE을 채운다.
            painter.setPen(QColor("#5b6675"))
            painter.drawText(thumb, Qt.AlignCenter, "Loading image…")

        fm = option.fontMetrics
        inner_w = rect.width() - 2 * _PAD  # 텍스트가 쓸 수 있는 가로 폭

        # --- 캡션(한 줄, 오른쪽 말줄임) ---
        cap = str(index.data(CAPTION_ROLE) or "")
        y_cap = thumb.bottom() + 6 + fm.ascent()  # 썸네일 아래 baseline 위치
        painter.setPen(QColor("#e5e7eb"))
        # elidedText: 폭을 넘으면 끝을 '…'로 줄여 한 줄에 맞춘다.
        painter.drawText(rect.x() + _PAD, y_cap, fm.elidedText(cap, Qt.ElideRight, inner_w))

        # --- 폴더 이름(가운데 말줄임) ---
        # 경로 끝의 구분자를 떼고 마지막 폴더명만 표시. 폴더 단위 결과면 멤버 수도 덧붙인다.
        name = os.path.basename(str(index.data(FOLDER_ROLE) or "").rstrip("/\\"))
        members = index.data(MEMBERS_ROLE)
        if members and int(members) > 1:
            name = f"{name}  ·  {int(members)} images"
        y_name = y_cap + fm.height()
        painter.setPen(QColor("#94a3b8"))
        # 폴더명은 가운데 말줄임(ElideMiddle)이라 앞/뒤 정보를 모두 남긴다.
        painter.drawText(rect.x() + _PAD, y_name, fm.elidedText(name, Qt.ElideMiddle, inner_w))

        # --- 점수 배지(썸네일 우상단) ---
        # 점수가 없거나 0이면 그리지 않는다(예: 코사인으로 순위 매기지 않는 그래프 멤버십 결과).
        score = index.data(SCORE_ROLE)
        if score is not None and float(score) > 0.0:
            txt = f"{float(score):.2f}"
            bw = fm.horizontalAdvance(txt) + 12  # 글자 폭 + 좌우 여백
            bh = fm.height() + 4
            badge = QRect(thumb.right() - bw, thumb.y() + 4, bw, bh)
            bpath = QPainterPath()
            bpath.addRoundedRect(QRectF(badge), 6, 6)
            painter.fillPath(bpath, QColor(0, 0, 0, 160))  # 반투명 검정 바탕(가독성)
            painter.setPen(QColor("#7dd3fc"))
            painter.drawText(badge, Qt.AlignCenter, txt)

        # --- 매칭 출처 배지(썸네일 좌상단): V / K / V+K / G ---
        mb = _MATCH_BADGE.get(str(index.data(MATCH_ROLE) or ""))
        if mb:
            mtxt, mcolor = mb
            mbw = fm.horizontalAdvance(mtxt) + 10
            mbh = fm.height() + 4
            mbadge = QRect(thumb.x() + 4, thumb.y() + 4, mbw, mbh)
            mpath = QPainterPath()
            mpath.addRoundedRect(QRectF(mbadge), 6, 6)
            painter.fillPath(mpath, QColor(0, 0, 0, 160))
            painter.setPen(QColor(mcolor))  # 출처별 색으로 구분
            painter.drawText(mbadge, Qt.AlignCenter, mtxt)

        painter.restore()  # save()로 격리한 렌더 상태 복원
