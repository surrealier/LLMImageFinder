"""실제 백엔드가 mock 모드로 떨어졌을 때 보여주는, 닫을 수 있는 경고 막대.

예: jina-clip/vLLM 로드에 실패하면 앱은 죽지 않고 mock으로 폴백하는데, 그 사실을
사용자가 모르고 결과를 신뢰하지 않도록 이 막대로 분명히 알린다.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QFrame, QPushButton, QSizePolicy

from imgsearch.ui import icons


class MockBanner(QFrame):
    # '설정 열기' 버튼을 누르면 방출되는 신호. 메인 윈도우가 받아 설정 다이얼로그를 연다.
    open_settings = Signal()

    def __init__(self, parent=None) -> None:
        """경고 막대 위젯을 구성한다(처음에는 숨김 상태)."""
        super().__init__(parent)
        # objectName을 지정해 아래 스타일시트의 '#MockBanner' 선택자가 이 위젯에만 적용되게 한다.
        self.setObjectName("MockBanner")
        self.setStyleSheet(
            "#MockBanner { background: #3b2f12; border: 1px solid #a16207;"
            " border-radius: 6px; }"
            " #MockBanner QLabel { color: #fde68a; }"
        )
        # mock으로 떨어진 경우에만 show_reason()이 보이게 하므로 기본은 숨김.
        self.setVisible(False)

        # 한 줄 레이아웃: [경고 아이콘] [메시지(늘어남)] [설정 열기] [닫기].
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 8, 6)

        self._icon = QLabel()
        self._icon.setPixmap(icons.warn().pixmap(18, 18))
        lay.addWidget(self._icon)

        self._msg = QLabel("Running in MOCK mode.")
        self._msg.setWordWrap(True)
        # 메시지가 남는 가로 공간을 모두 차지하도록(stretch=1) 늘어나게 한다.
        self._msg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay.addWidget(self._msg, 1)

        btn = QPushButton("Open Settings")
        # 버튼 클릭을 그대로 open_settings 신호로 전달(별도 핸들러 없이 emit에 직접 연결).
        btn.clicked.connect(self.open_settings.emit)
        lay.addWidget(btn)

        close = QPushButton("✕")
        close.setFixedWidth(28)
        # 닫기 버튼은 막대를 숨기기만 한다(상태를 영구 저장하지는 않음).
        close.clicked.connect(lambda: self.setVisible(False))
        lay.addWidget(close)

    def show_reason(self, reason: str) -> None:
        """mock 폴백 사유를 메시지에 채우고 막대를 보이게 한다.

        reason이 있으면 괄호로 덧붙여 어떤 백엔드가 왜 실패했는지 알 수 있게 한다.
        """
        base = (
            "Running in MOCK mode — only deterministic search results are provided. "
            "To use real models, configure a backend in Settings."
        )
        if reason:
            base += f"  (reason: {reason})"
        self._msg.setText(base)
        self.setVisible(True)
