"""'sLLM_' 채팅 패널: 대화 기록(transcript) + 한 줄 한국어 검색어 입력창.

입력창은 셸처럼 영속적인 검색어 히스토리를 유지한다(↑/↓ 로 이전 검색어 소환).
히스토리는 앱 데이터 디렉터리에 JSON으로 저장되며, 절대 데이터셋 안에는 두지 않는다
(데이터셋을 깨끗하게 유지하고, 사용자별 기록을 분리하기 위함).
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

_HISTORY_MAX = 200  # 저장/유지할 검색어 히스토리 최대 개수(오래된 것부터 버린다)


class ChatWidget(QWidget):
    """채팅 패널 위젯 — 검색어 입력/제출, 대화 기록 출력, ↑/↓ 히스토리 탐색을 담당."""

    submitted = Signal(str)  # 사용자가 검색어를 제출하면 그 텍스트를 실어 발신(메인 윈도우가 수신)

    def __init__(self, history_file: Path | None = None, parent=None) -> None:
        """히스토리 파일 경로를 받아 위젯을 구성한다(파일이 None이면 히스토리를 저장하지 않음)."""
        super().__init__(parent)
        self._history_file = history_file
        self._history: list[str] = self._load_history()
        self._hist_pos: int | None = None  # None = 히스토리 탐색 중이 아님(현재 입력을 편집 중)
        self._draft = ""  # 히스토리 탐색을 시작하기 직전 입력 중이던 텍스트(↓로 끝까지 가면 복원)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header = QLabel("sLLM_")
        header.setStyleSheet(
            "font-weight:700; font-size:15px; color:#7dd3fc;"
            " padding:4px 2px; letter-spacing:1px;"
        )
        root.addWidget(header)

        # 대화 기록 표시 영역(HTML 렌더). 외부 링크 자동 열기는 끈다(보안/오작동 방지).
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(False)
        self.transcript.setStyleSheet("QTextBrowser{border:1px solid #2b3342; border-radius:6px;}")
        root.addWidget(self.transcript, 1)

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("찾고 싶은 이미지를 설명하세요 — ↑/↓ 로 이전 검색어")
        self.input.returnPressed.connect(self._on_send)  # Enter로 제출
        # ↑/↓ 키를 가로채 히스토리를 탐색하기 위해 이벤트 필터를 설치(아래 eventFilter 참고).
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

    # --- 히스토리 ---
    def _load_history(self) -> list[str]:
        """히스토리 JSON 파일을 읽어 검색어 리스트로 돌려준다(없거나 깨졌으면 빈 리스트)."""
        if self._history_file is None:
            return []
        try:
            data = json.loads(self._history_file.read_text(encoding="utf-8"))
            # 리스트 형태일 때만 받아들이고, 최근 _HISTORY_MAX개로 자른다(형식이 다르면 무시).
            return [str(x) for x in data][-_HISTORY_MAX:] if isinstance(data, list) else []
        except Exception:
            return []  # 파일이 없거나 손상돼도 앱은 정상 시작해야 하므로 빈 히스토리로 폴백

    def _save_history(self) -> None:
        """현재 히스토리(최근 _HISTORY_MAX개)를 UTF-8 JSON으로 저장한다(한글은 그대로 보존)."""
        if self._history_file is None:
            return
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)  # 디렉터리 보장
            self._history_file.write_text(
                # ensure_ascii=False: 한글을 \uXXXX로 이스케이프하지 않고 그대로 저장.
                json.dumps(self._history[-_HISTORY_MAX:], ensure_ascii=False, indent=0),
                encoding="utf-8",
            )
        except Exception:
            pass  # 히스토리는 편의 기능 — 저장 실패가 검색 자체를 망쳐선 안 된다

    def _remember(self, text: str) -> None:
        """검색어를 히스토리에 추가한다 — 직전 항목과 같으면 중복 저장하지 않는다."""
        if not self._history or self._history[-1] != text:
            self._history.append(text)
            del self._history[:-_HISTORY_MAX]  # 앞쪽(오래된) 초과분을 잘라 상한을 유지
            self._save_history()

    def eventFilter(self, obj, event):  # noqa: N802
        """입력창의 ↑/↓ 키를 가로채 셸 스타일 히스토리 탐색을 구현한다.

        ↑: 더 과거로(처음 누르면 현재 입력을 _draft에 백업). ↓: 더 최근으로,
        끝을 넘어가면 _draft를 복원하고 탐색 모드를 빠져나온다. 처리했으면 True를 반환해
        기본 동작(커서 이동 등)을 막는다.
        """
        if obj is self.input and event.type() == QEvent.KeyPress and self._history:
            key = event.key()
            if key == Qt.Key_Up:
                if self._hist_pos is None:
                    # 탐색 시작: 입력 중이던 초안을 백업하고 가장 최근 항목부터 보여 준다.
                    self._draft = self.input.text()
                    self._hist_pos = len(self._history) - 1
                elif self._hist_pos > 0:
                    self._hist_pos -= 1  # 더 과거 항목으로(0에서 멈춤)
                self.input.setText(self._history[self._hist_pos])
                return True
            if key == Qt.Key_Down and self._hist_pos is not None:
                self._hist_pos += 1
                if self._hist_pos >= len(self._history):
                    # 가장 최근을 지나 끝을 넘어가면 탐색을 끝내고 백업해 둔 초안을 되살린다.
                    self._hist_pos = None
                    self.input.setText(self._draft)
                else:
                    self.input.setText(self._history[self._hist_pos])
                return True
        return super().eventFilter(obj, event)

    # --- 입력 ---
    def _on_send(self) -> None:
        """입력창의 텍스트를 다듬어 제출한다 — 빈 문자열이면 무시, 아니면 히스토리에 남기고 발신."""
        text = self.input.text().strip()
        if not text:
            return
        self._remember(text)
        self._hist_pos = None  # 제출 후 히스토리 탐색 상태 초기화
        self._draft = ""
        self.input.clear()
        self.submitted.emit(text)  # 메인 윈도우의 run_query로 전달

    def set_busy(self, busy: bool) -> None:
        """검색 중 입력/전송을 잠근다(busy=True). 풀릴 때는 입력창에 다시 포커스를 준다."""
        self.input.setEnabled(not busy)
        self.send_btn.setEnabled(not busy)
        if not busy:
            self.input.setFocus()

    # --- 대화 기록 출력 ---
    def _append(self, html_fragment: str) -> None:
        """HTML 조각을 기록 영역에 덧붙이고 스크롤을 맨 아래로 내린다(최신 메시지가 보이도록)."""
        self.transcript.append(html_fragment)
        sb = self.transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    def add_user(self, text: str) -> None:
        """사용자 발화를 '나 ▸' 스타일로 추가한다. html.escape로 특수문자 주입을 막는다."""
        self._append(
            f'<div style="margin:6px 0;"><span style="color:#7dd3fc;font-weight:600;">나 ▸ </span>'
            f'<span style="color:#e5e7eb;">{html.escape(text)}</span></div>'
        )

    def add_assistant(self, text: str) -> None:
        """어시스턴트 응답을 'sLLM ▸' 스타일로 추가한다(검색 요약 등)."""
        self._append(
            f'<div style="margin:6px 0;"><span style="color:#34d399;font-weight:600;">sLLM ▸ </span>'
            f'<span style="color:#d1d5db;">{html.escape(text)}</span></div>'
        )

    def add_system(self, text: str) -> None:
        """시스템 안내/오류 메시지를 흐린 이탤릭 스타일로 추가한다(진행 상황·경고 등)."""
        self._append(
            f'<div style="margin:4px 0;color:#94a3b8;font-style:italic;">{html.escape(text)}</div>'
        )
