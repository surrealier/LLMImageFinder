"""The 'sLLM_' chat panel: transcript + single-line Korean query input."""

from __future__ import annotations

import html

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from imgsearch.ui import icons


class ChatWidget(QWidget):
    submitted = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header = QLabel("sLLM_")
        header.setStyleSheet(
            "font-weight:700; font-size:15px; color:#7dd3fc;"
            " padding:4px 2px; letter-spacing:1px;"
        )
        root.addWidget(header)

        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(False)
        self.transcript.setStyleSheet("QTextBrowser{border:1px solid #2b3342; border-radius:6px;}")
        root.addWidget(self.transcript, 1)

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("찾고 싶은 이미지를 설명하세요 — 예: 밤에 오토바이가 주차되어 있는 이미지")
        self.input.returnPressed.connect(self._on_send)
        row.addWidget(self.input, 1)

        self.send_btn = QPushButton(icons.send(), "")
        self.send_btn.setToolTip("검색")
        self.send_btn.setFixedWidth(42)
        self.send_btn.clicked.connect(self._on_send)
        row.addWidget(self.send_btn)
        root.addLayout(row)

        self.add_system(
            "안녕하세요! 데이터셋에서 찾고 싶은 장면을 자연어로 설명해 주세요. "
            "예) ‘불과 연기가 있는 이미지’, ‘사람이 대로변에 돌아다니는 이미지’"
        )

    # --- input ---
    def _on_send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.submitted.emit(text)

    def set_busy(self, busy: bool) -> None:
        self.input.setEnabled(not busy)
        self.send_btn.setEnabled(not busy)
        if not busy:
            self.input.setFocus()

    # --- transcript ---
    def _append(self, html_fragment: str) -> None:
        self.transcript.append(html_fragment)
        sb = self.transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    def add_user(self, text: str) -> None:
        self._append(
            f'<div style="margin:6px 0;"><span style="color:#7dd3fc;font-weight:600;">나 ▸ </span>'
            f'<span style="color:#e5e7eb;">{html.escape(text)}</span></div>'
        )

    def add_assistant(self, text: str) -> None:
        self._append(
            f'<div style="margin:6px 0;"><span style="color:#34d399;font-weight:600;">sLLM ▸ </span>'
            f'<span style="color:#d1d5db;">{html.escape(text)}</span></div>'
        )

    def add_system(self, text: str) -> None:
        self._append(
            f'<div style="margin:4px 0;color:#94a3b8;font-style:italic;">{html.escape(text)}</div>'
        )
