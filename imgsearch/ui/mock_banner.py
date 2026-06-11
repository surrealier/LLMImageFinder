"""A dismissible warning bar shown when a real backend fell back to mock mode."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QFrame, QPushButton, QSizePolicy

from imgsearch.ui import icons


class MockBanner(QFrame):
    open_settings = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("MockBanner")
        self.setStyleSheet(
            "#MockBanner { background: #3b2f12; border: 1px solid #a16207;"
            " border-radius: 6px; }"
            " #MockBanner QLabel { color: #fde68a; }"
        )
        self.setVisible(False)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 8, 6)

        self._icon = QLabel()
        self._icon.setPixmap(icons.warn().pixmap(18, 18))
        lay.addWidget(self._icon)

        self._msg = QLabel("MOCK 모드로 실행 중입니다.")
        self._msg.setWordWrap(True)
        self._msg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay.addWidget(self._msg, 1)

        btn = QPushButton("설정 열기")
        btn.clicked.connect(self.open_settings.emit)
        lay.addWidget(btn)

        close = QPushButton("✕")
        close.setFixedWidth(28)
        close.clicked.connect(lambda: self.setVisible(False))
        lay.addWidget(close)

    def show_reason(self, reason: str) -> None:
        base = (
            "MOCK 모드로 실행 중입니다 — 결정적(deterministic) 검색 결과만 제공됩니다. "
            "실제 모델을 사용하려면 설정에서 백엔드를 구성하세요."
        )
        if reason:
            base += f"  (사유: {reason})"
        self._msg.setText(base)
        self.setVisible(True)
