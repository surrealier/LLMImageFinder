"""모달 라이트박스(전체화면 이미지 뷰어): 확대/이동(zoom/pan), ←/→ 키 탐색,
YOLO 라벨 박스 오버레이, 검색 결과(hit) 메타데이터 표시를 담당한다.

검색 결과 갤러리에서 썸네일을 클릭하면 이 다이얼로그가 떠서 원본 이미지를
크게 보여주고, 같은 결과 묶음 안에서 좌우로 넘겨볼 수 있게 한다.
"""

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
    """클래스 ID로부터 항상 동일한 색상을 만들어 낸다(결정적 색상 매핑).

    같은 클래스에는 항상 같은 색이 나오도록 ID에 57(서로소에 가까운 값)을 곱한 뒤
    360으로 나눈 나머지를 HSV의 색상(hue)으로 쓴다. 채도/명도는 고정해 선명하게 표시한다.
    """
    # 57을 곱해 인접 클래스끼리 색상이 충분히 벌어지도록(눈에 잘 구분되도록) 한다.
    return QColor.fromHsv((int(cid) * 57) % 360, 200, 255)


class ImageViewer(QDialog):
    # 같은 세션(앱 실행) 동안 여러 번 뷰어를 열어도 '라벨 표시' 상태를 기억하기 위한
    # 클래스 변수. 인스턴스가 아닌 클래스에 두었기 때문에 모든 뷰어가 값을 공유한다.
    show_boxes = True

    def __init__(
        self,
        image_paths: Sequence[str],
        start_index: int = 0,
        parent=None,
        class_names: Optional[dict] = None,
        meta_provider: Optional[Callable[[str], str]] = None,
    ) -> None:
        """뷰어를 초기화한다.

        매개변수
        - image_paths: 좌우로 넘겨볼 이미지 경로 목록(검색 결과 묶음).
        - start_index: 처음에 보여줄 이미지의 인덱스.
        - class_names: YOLO 클래스 ID -> 표시 이름 매핑(라벨 박스 위 글자에 사용).
        - meta_provider: 경로를 받아 화면 하단에 보여줄 메타데이터 문자열을 돌려주는
          콜백. 호출자(메인 윈도우)가 검색 점수 등 부가정보를 주입할 수 있게 한다.
        """
        super().__init__(parent)
        # 빈 문자열/None 경로는 걸러낸다(잘못된 항목으로 인한 오류 방지).
        self.paths = [p for p in image_paths if p]
        if not self.paths:
            # 보여줄 이미지가 하나도 없으면 빈 항목 하나를 두어 인덱스 계산이 안전하게 동작하도록 한다.
            self.paths = [""]
        # start_index를 [0, 마지막 인덱스] 범위로 강제 보정(범위를 벗어나도 죽지 않게).
        self.i = max(0, min(start_index, len(self.paths) - 1))
        # True이면 창 크기에 맞춰 자동 맞춤(fit) 상태. 사용자가 확대/축소하면 False가 된다.
        self._fit_mode = True
        self._class_names = dict(class_names or {})
        self._meta_provider = meta_provider
        # 현재 그려둔 라벨 박스 그래픽 아이템들(이미지를 바꿀 때 지우기 위해 추적).
        self._box_items: list = []

        self.setWindowTitle("View image")
        self.resize(960, 720)
        self.setStyleSheet("QDialog{background:#0f141c;}")

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # QGraphicsScene/View를 쓰는 이유: 확대/이동(zoom/pan)과 라벨 박스 오버레이를
        # 좌표 변환과 함께 자연스럽게 처리할 수 있기 때문이다(단순 QLabel로는 어렵다).
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        # 안티앨리어싱 + 부드러운 픽스맵 변환으로 확대 시에도 깔끔하게 보이도록 한다.
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        # 손바닥 모양 드래그로 이미지를 끌어서 이동(pan)할 수 있게 한다.
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setBackgroundBrush(Qt.black)
        self.view.setAlignment(Qt.AlignCenter)
        # 일단 확대되면 뷰포트가 휠 이벤트를 '스크롤'로 가로채 버린다 —
        # 그래서 뷰포트에 이벤트 필터를 걸어 휠은 항상 '확대/축소'로 쓰이도록 한다(eventFilter 참고).
        self.view.viewport().installEventFilter(self)
        # 실제 이미지를 담는 픽스맵 아이템. 라벨 박스는 이 아이템의 자식으로 붙인다.
        self.pix_item = QGraphicsPixmapItem()
        self.pix_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(self.pix_item)
        root.addWidget(self.view, 1)

        # 이미지 아래에 표시되는 메타데이터 라벨(검색 점수, 캡션 등). 마우스로 선택/복사 가능.
        self.meta_label = QLabel("")
        self.meta_label.setWordWrap(True)
        self.meta_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.meta_label.setStyleSheet("color:#9fb2c8; padding:2px 4px;")
        # 보여줄 메타데이터가 있을 때만 _refresh_meta에서 표시한다.
        self.meta_label.setVisible(False)
        root.addWidget(self.meta_label)

        # 하단 도구 모음(이전/다음/라벨/확대/축소/맞춤/폴더 열기).
        bar = QHBoxLayout()
        self.prev_btn = QPushButton(icons.prev(), "")
        self.prev_btn.setToolTip("Previous (←)")
        self.prev_btn.clicked.connect(self.prev)
        bar.addWidget(self.prev_btn)

        self.next_btn = QPushButton(icons.nxt(), "")
        self.next_btn.setToolTip("Next (→)")
        self.next_btn.clicked.connect(self.next)
        bar.addWidget(self.next_btn)

        # 라벨 박스 표시 토글 버튼. 누름 상태는 클래스 변수 show_boxes로 세션 간 공유된다.
        self.boxes_btn = QPushButton(icons.tags(), " Labels")
        self.boxes_btn.setCheckable(True)
        self.boxes_btn.setChecked(ImageViewer.show_boxes)
        self.boxes_btn.setToolTip("Show/hide YOLO label boxes (B)")
        self.boxes_btn.toggled.connect(self._on_boxes_toggled)
        bar.addWidget(self.boxes_btn)

        # 좌우 stretch로 가운데에 'n / 전체  —  파일명' 카운터를 배치한다.
        bar.addStretch(1)
        self.counter = QLabel("")
        self.counter.setStyleSheet("color:#cbd5e1;")
        bar.addWidget(self.counter)
        bar.addStretch(1)

        zin = QPushButton(icons.zoom_in(), "")
        zin.setToolTip("Zoom in (+)")
        # 람다로 배율 인자를 넘긴다(1.25배 확대).
        zin.clicked.connect(lambda: self.zoom(1.25))
        bar.addWidget(zin)
        zout = QPushButton(icons.zoom_out(), "")
        zout.setToolTip("Zoom out (-)")
        # 0.8배 = 1/1.25, 확대 버튼과 대칭이 되도록 한 단계 축소.
        zout.clicked.connect(lambda: self.zoom(0.8))
        bar.addWidget(zout)
        fit = QPushButton(icons.fit(), "")
        fit.setToolTip("Fit to window (F)")
        fit.clicked.connect(self.fit)
        bar.addWidget(fit)

        openf = QPushButton(icons.folder(), " Open folder")
        openf.setToolTip("Reveal this image's location in Explorer")
        openf.clicked.connect(self._open_folder)
        bar.addWidget(openf)
        root.addLayout(bar)

        # 위젯 구성이 끝났으니 시작 인덱스의 이미지를 실제로 로드한다.
        self._load()

    # --- 탐색(navigation) ---
    @staticmethod
    def _read_bounded(path: str, max_side: int = 2560) -> QPixmap:
        """긴 변이 max_side를 넘지 않도록 축소 디코딩해 픽스맵을 만든다.

        초대형 이미지를 원본 해상도로 디코딩하면 UI 스레드가 멈추거나 메모리가
        과도하게 쓰일 수 있다. 디코딩 단계에서 미리 크기를 제한해 이를 막는다.
        화면 표시 용도로는 이 정도 해상도면 충분하다.
        """
        if not path:
            return QPixmap()
        reader = QImageReader(path)
        # EXIF 회전 정보를 자동 적용(세로로 찍은 사진이 눕지 않도록).
        reader.setAutoTransform(True)
        size = reader.size()
        # 긴 변이 한계를 넘을 때만 비율을 유지하며 축소 디코딩하도록 설정한다.
        if size.isValid() and max(size.width(), size.height()) > max_side:
            reader.setScaledSize(size.scaled(max_side, max_side, Qt.KeepAspectRatio))
        img = reader.read()
        # 정상 디코딩되면 그 결과를, 실패하면 경로로 직접 로드한 픽스맵을 폴백으로 쓴다.
        return QPixmap.fromImage(img) if not img.isNull() else QPixmap(path)

    def _load(self) -> None:
        """현재 인덱스(self.i)의 이미지를 읽어 화면/카운터/박스/메타를 갱신한다."""
        path = self.paths[self.i]
        pm = self._read_bounded(path)
        self.pix_item.setPixmap(pm)
        # 씬 영역을 이미지 크기에 맞춰 갱신해야 fit/스크롤이 올바르게 동작한다.
        self.scene.setSceneRect(self.pix_item.boundingRect())
        # 새 이미지를 로드할 때마다 항상 '창에 맞춤' 상태로 시작한다.
        self._fit_mode = True
        self.fit()
        self.counter.setText(f"{self.i + 1} / {len(self.paths)}   —   {os.path.basename(path)}")
        # 이미지가 한 장뿐이면 이전/다음 버튼을 비활성화한다.
        self.prev_btn.setEnabled(len(self.paths) > 1)
        self.next_btn.setEnabled(len(self.paths) > 1)
        self._refresh_boxes()
        self._refresh_meta(path)

    def _refresh_meta(self, path: str) -> None:
        """meta_provider 콜백으로 받은 부가정보를 하단 라벨에 표시한다.

        콜백이 없거나 예외가 나도 뷰어 자체는 계속 동작해야 하므로 실패는 조용히 무시한다.
        """
        meta = ""
        if self._meta_provider is not None:
            try:
                meta = self._meta_provider(path) or ""
            except Exception:
                # 메타데이터는 부가정보일 뿐이라 실패해도 빈 문자열로 두고 넘어간다.
                meta = ""
        self.meta_label.setText(meta)
        # 내용이 있을 때만 라벨을 보이게 해 불필요한 빈 줄을 만들지 않는다.
        self.meta_label.setVisible(bool(meta))
        self.meta_label.setToolTip(path)

    # --- YOLO 라벨 박스 오버레이 ---
    def _clear_boxes(self) -> None:
        """이전 이미지에서 그렸던 라벨 박스 아이템들을 씬에서 모두 제거한다."""
        for item in self._box_items:
            self.scene.removeItem(item)
        self._box_items = []

    def _refresh_boxes(self) -> None:
        """현재 이미지에 대응하는 YOLO 라벨을 읽어 박스/이름을 다시 그린다."""
        self._clear_boxes()
        path = self.paths[self.i]
        pm = self.pix_item.pixmap()
        # 이미지 경로로부터 같은 위치의 라벨(.txt) 경로를 추론한다(없으면 None).
        label_path = labels.label_path_for(path) if path else None
        # 라벨 파일이 있을 때만 토글 버튼을 활성화한다.
        self.boxes_btn.setEnabled(label_path is not None)
        # 표시 끔 / 라벨 없음 / 이미지 없음이면 그릴 것이 없으므로 종료.
        if not ImageViewer.show_boxes or label_path is None or pm.isNull():
            return
        w, h = pm.width(), pm.height()
        for cid, cx, cy, bw, bh in labels.parse_yolo(label_path):
            # YOLO는 중심점/너비/높이를 0~1로 정규화해 저장한다.
            # 이를 '현재 표시 중인 픽스맵' 픽셀 좌표로 환산한다. _read_bounded가 축소
            # 디코딩했어도 픽스맵 크기 기준으로 곱하므로 박스 위치는 항상 정확하다.
            rect = QRectF((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h)
            color = _class_color(cid)
            # 박스를 픽스맵 아이템의 자식으로 만들어 이미지와 같은 좌표계를 쓰게 한다.
            box = QGraphicsRectItem(rect, self.pix_item)
            pen = QPen(color, 2)
            pen.setCosmetic(True)  # 줌 배율과 무관하게 화면상 선 두께를 일정하게 유지
            box.setPen(pen)
            # 클래스 이름이 등록돼 있으면 그 이름을, 없으면 '#ID'를 라벨로 쓴다.
            name = str(self._class_names.get(str(cid), f"#{cid}"))
            text = QGraphicsSimpleTextItem(name, box)  # 박스의 자식 → 박스가 지워질 때 함께 제거됨
            text.setBrush(QBrush(color))
            # 줌과 무관하게 글자 크기를 고정(확대해도 글자가 거대해지지 않도록).
            text.setFlag(QGraphicsSimpleTextItem.ItemIgnoresTransformations)
            # 글자를 박스 위쪽에 살짝 띄워 배치(음수 좌표로 잘리지 않게 0으로 하한 처리).
            text.setPos(rect.x(), max(0.0, rect.y() - 2))
            # 자식 텍스트는 박스와 함께 제거되므로 박스만 추적 목록에 넣는다.
            self._box_items.append(box)

    def _on_boxes_toggled(self, checked: bool) -> None:
        """라벨 토글 버튼이 눌리면 세션 공유 상태를 갱신하고 다시 그린다."""
        ImageViewer.show_boxes = bool(checked)
        self._refresh_boxes()

    def show_index(self, i: int) -> None:
        """주어진 인덱스로 이동한다. 모듈러 연산으로 양 끝에서 순환(wrap-around)된다."""
        if not self.paths:
            return
        # i가 음수거나 길이를 넘어도 % 로 0..len-1 범위로 감싸 순환 탐색이 되게 한다.
        self.i = i % len(self.paths)
        self._load()

    def prev(self) -> None:
        """이전 이미지로 이동(첫 장에서 누르면 마지막 장으로 순환)."""
        self.show_index(self.i - 1)

    def next(self) -> None:
        """다음 이미지로 이동(마지막 장에서 누르면 첫 장으로 순환)."""
        self.show_index(self.i + 1)

    # --- 확대/축소(zoom) ---
    def fit(self) -> None:
        """이미지를 창 크기에 비율을 유지하며 꽉 차게 맞춘다(맞춤 모드 진입)."""
        if not self.pix_item.pixmap().isNull():
            # 누적된 확대/축소 변환을 초기화한 뒤 다시 맞춰야 정확히 들어맞는다.
            self.view.resetTransform()
            self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)
            self._fit_mode = True

    def zoom(self, factor: float) -> None:
        """주어진 배율만큼 확대/축소한다(1보다 크면 확대, 작으면 축소)."""
        # 한 번이라도 직접 줌하면 더 이상 자동 맞춤 대상이 아니다(리사이즈 시 유지하기 위함).
        self._fit_mode = False
        self.view.scale(factor, factor)

    def eventFilter(self, obj, event):  # noqa: N802
        """뷰포트의 휠 이벤트를 가로채 스크롤 대신 확대/축소로 처리한다."""
        # 확대된 상태에서도 휠이 '스크롤'이 아니라 항상 '줌'이 되도록 여기서 직접 처리한다.
        if obj is self.view.viewport() and event.type() == QEvent.Wheel:
            # 휠 위로(>0)면 확대, 아래로면 축소. True를 반환해 기본 스크롤 동작을 막는다.
            self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
            return True
        return super().eventFilter(obj, event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        """뷰포트 밖(빈 영역 등)에서의 휠도 확대/축소로 처리한다."""
        self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """창 크기가 바뀔 때 맞춤 모드라면 다시 창에 맞춘다."""
        super().resizeEvent(event)
        # 사용자가 직접 줌한 상태(_fit_mode=False)라면 배율을 건드리지 않는다.
        if self._fit_mode:
            self.fit()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """키보드 단축키 처리: 방향키 탐색, ESC 닫기, +/- 줌, F 맞춤, B 라벨 토글."""
        key = event.key()
        # 오른쪽/스페이스/아래 → 다음 이미지(슬라이드처럼 한 손으로 넘기기 편하게).
        if key in (Qt.Key_Right, Qt.Key_Space, Qt.Key_Down):
            self.next()
        elif key in (Qt.Key_Left, Qt.Key_Up):
            self.prev()
        elif key == Qt.Key_Escape:
            self.accept()
        # '+'와 '='를 함께 받는 이유: '='가 Shift 없이 누르는 '+' 자리에 있는 키이기 때문.
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom(1.25)
        elif key == Qt.Key_Minus:
            self.zoom(0.8)
        elif key == Qt.Key_F:
            self.fit()
        elif key == Qt.Key_B:
            # 버튼을 토글하면 연결된 _on_boxes_toggled가 호출돼 박스가 갱신된다.
            self.boxes_btn.toggle()
        else:
            # 처리하지 않은 키는 기본 동작으로 넘긴다.
            super().keyPressEvent(event)

    def _open_folder(self) -> None:
        """현재 이미지가 들어있는 폴더를 시스템 파일 관리자에서 연다."""
        if self.paths and self.paths[self.i]:
            reveal_in_explorer(self.paths[self.i])
