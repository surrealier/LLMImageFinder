"""The 'sLLM_' chat panel: transcript + single-line Korean query input.

The input keeps a persistent query history (↑/↓ to recall, like a shell);
it is stored as JSON in the app-data dir, never inside the dataset.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, Signal
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

_HISTORY_MAX = 200


class ChatWidget(QWidget):
    submitted = Signal(str)

    def __init__(self, history_file: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self._history_file = history_file
        self._history: list[str] = self._load_history()
        self._hist_pos: int | None = None  # None = not navigating history
        self._draft = ""  # text being typed before history navigation started

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
        self.input.setPlaceholderText("찾고 싶은 이미지를 설명하세요 — ↑/↓ 로 이전 검색어")
        self.input.returnPressed.connect(self._on_send)
        self.input.installEventFilter(self)
        row.addWidget(self.input, 1)

        self.send_btn = QPushButton(icons.send(), "")
        self.send_btn.setToolTip("검색 (Enter)")
        self.send_btn.setFixedWidth(42)
        self.send_btn.clicked.connect(self._on_send)
        row.addWidget(self.send_btn)
        root.addLayout(row)

        self.add_system(
            "안녕하세요! 데이터셋에서 찾고 싶은 장면을 자연어로 설명해 주세요. "
            "↑/↓ 키로 이전 검색어를 다시 불러올 수 있습니다."
        )

    # --- history ---
    def _load_history(self) -> list[str]:
        if self._history_file is None:
            return []
        try:
            data = json.loads(self._history_file.read_text(encoding="utf-8"))
            return [str(x) for x in data][-_HISTORY_MAX:] if isinstance(data, list) else []
        except Exception:
            return []

    def _save_history(self) -> None:
        if self._history_file is None:
            return
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            self._history_file.write_text(
                json.dumps(self._history[-_HISTORY_MAX:], ensure_ascii=False, indent=0),
                encoding="utf-8",
            )
        except Exception:
            pass  # history is a convenience — never fail the search over it

    def _remember(self, text: str) -> None:
        if not self._history or self._history[-1] != text:
            self._history.append(text)
            del self._history[:-_HISTORY_MAX]
            self._save_history()

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self.input and event.type() == QEvent.KeyPress and self._history:
            key = event.key()
            if key == Qt.Key_Up:
                if self._hist_pos is None:
                    self._draft = self.input.text()
                    self._hist_pos = len(self._history) - 1
                elif self._hist_pos > 0:
                    self._hist_pos -= 1
                self.input.setText(self._history[self._hist_pos])
                return True
            if key == Qt.Key_Down and self._hist_pos is not None:
                self._hist_pos += 1
                if self._hist_pos >= len(self._history):
                    self._hist_pos = None
                    self.input.setText(self._draft)
                else:
                    self.input.setText(self._history[self._hist_pos])
                return True
        return super().eventFilter(obj, event)

    # --- input ---
    def _on_send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self._remember(text)
        self._hist_pos = None
        self._draft = ""
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
