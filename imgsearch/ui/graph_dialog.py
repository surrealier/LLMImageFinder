"""객체 그래프 — 객체 동시 출현(co-occurrence)을 탐색하고 AND 필터로 이미지를 끌어온다.

화면 구성:
- 왼쪽: 모든 객체 클래스와 그 이미지 수(그리고 AND 필터에 포함할지 체크박스).
- 오른쪽: 왼쪽에서 선택한 클래스와 '함께' 나타나는 클래스 목록.
- 아래: '선택한 객체를 모두 포함한 이미지 보기' 버튼 → 조건을 만족하는 이미지 경로를
  신호(show_images)로 내보내면, 메인 윈도우가 이를 받아 갤러리 행으로 만든다.

모든 조회는 GraphStore 추상화를 통한다(백엔드가 메모리든 kuzu든 동일한 코드로 동작).
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from imgsearch.graph.base import GraphStore


class GraphDialog(QDialog):
    # AND 필터를 만족하는 이미지 경로 목록(list[str])을 메인 윈도우로 전달하는 신호.
    show_images = Signal(list)

    def __init__(self, graph: GraphStore, parent=None) -> None:
        """그래프 저장소를 받아 클래스 목록과 통계로 다이얼로그를 채운다."""
        super().__init__(parent)
        self.graph = graph
        self.setWindowTitle("객체 그래프")
        self.resize(720, 520)

        root = QVBoxLayout(self)
        # 그래프 조회는 백엔드 상태에 따라 실패할 수 있다(예: kuzu 미설치/빈 인덱스).
        # 어떤 경우든 다이얼로그 자체는 떠야 하므로 예외는 빈 값으로 흡수한다.
        counts = {}
        try:
            counts = graph.class_counts()
        except Exception:
            counts = {}
        total = 0
        try:
            total = graph.count()
        except Exception:
            total = 0
        head = QLabel(
            f"라벨이 있는 이미지 {total}개 · 객체 클래스 {len(counts)}종 — "
            "왼쪽에서 객체를 선택하면 함께 나타나는 객체를 보여줍니다."
        )
        head.setWordWrap(True)
        head.setStyleSheet("color:#cbd5e1;")
        root.addWidget(head)

        # 본문은 좌(클래스 목록)/우(동시 출현 표)를 가로로 나눈다.
        body = QHBoxLayout()
        root.addLayout(body, 1)

        # --- 왼쪽: 체크 가능한 객체 클래스 목록 ---
        left = QVBoxLayout()
        left.addWidget(QLabel("객체 (체크 = AND 필터)"))
        self.class_list = QListWidget()
        # 이미지 수 내림차순(-kv[1]), 동수일 땐 이름 오름차순(kv[0])으로 정렬해 많이 쓰인 객체를 위로.
        for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            it = QListWidgetItem(f"{name}  ·  {n}장")
            # 표시 문구에는 개수를 붙이지만, 실제 클래스 이름은 UserRole 데이터로 따로 보관한다.
            it.setData(Qt.UserRole, name)
            # 체크박스를 켤 수 있게 플래그를 추가하고 기본은 해제 상태로 둔다.
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.class_list.addItem(it)
        # 현재 선택(하이라이트)이 바뀌면 오른쪽 동시 출현 표를 갱신한다.
        self.class_list.currentItemChanged.connect(self._on_select_class)
        left.addWidget(self.class_list, 1)
        body.addLayout(left, 1)

        # --- 오른쪽: 선택한 클래스와 함께 나타나는 객체 표 ---
        right = QVBoxLayout()
        self.co_title = QLabel("동시 출현 객체")
        right.addWidget(self.co_title)
        self.co_table = QTableWidget(0, 2)
        self.co_table.setHorizontalHeaderLabels(["객체", "동시 출현"])
        self.co_table.horizontalHeader().setStretchLastSection(True)
        self.co_table.verticalHeader().setVisible(False)
        # 표는 읽기 전용(조회 결과를 보여주기만 한다).
        self.co_table.setEditTriggers(QTableWidget.NoEditTriggers)
        right.addWidget(self.co_table, 1)
        body.addLayout(right, 1)

        # --- 아래: 선택 상태 표시 + AND 필터 실행 버튼 + 닫기 ---
        bottom = QHBoxLayout()
        self.sel_label = QLabel("")
        self.sel_label.setStyleSheet("color:#94a3b8;")
        bottom.addWidget(self.sel_label, 1)
        self.and_btn = QPushButton("선택한 객체를 모두 포함한 이미지 보기")
        self.and_btn.clicked.connect(self._emit_and_filter)
        bottom.addWidget(self.and_btn)
        close = QPushButton("닫기")
        close.clicked.connect(self.reject)
        bottom.addWidget(close)
        root.addLayout(bottom)

        # 체크 상태가 바뀔 때마다(itemChanged) 선택 요약/버튼 활성화를 갱신한다.
        self.class_list.itemChanged.connect(self._refresh_selection)
        # 초기 상태(아무것도 선택되지 않음)를 한 번 반영해 버튼을 비활성화해 둔다.
        self._refresh_selection()

    def _on_select_class(self, current: Optional[QListWidgetItem], _prev) -> None:
        """왼쪽 목록에서 선택이 바뀌면 그 클래스의 동시 출현 객체를 오른쪽 표에 채운다."""
        # 이전 결과를 비운다(선택이 비거나 조회 실패해도 빈 표가 되도록).
        self.co_table.setRowCount(0)
        if current is None:
            return
        # 표시 문구가 아닌 UserRole에 저장한 실제 클래스 이름으로 조회한다.
        name = current.data(Qt.UserRole)
        self.co_title.setText(f"‘{name}’와(과) 함께 나타나는 객체")
        try:
            # 상위 50개까지만 가져온다(목록이 너무 길어지지 않게).
            rows = self.graph.cooccurring(name, top=50)
        except Exception:
            rows = []
        self.co_table.setRowCount(len(rows))
        for r, (other, n) in enumerate(rows):
            # other: 함께 나타난 객체 이름, n: 동시 출현 횟수.
            self.co_table.setItem(r, 0, QTableWidgetItem(str(other)))
            self.co_table.setItem(r, 1, QTableWidgetItem(str(n)))

    def _checked_classes(self) -> list[str]:
        """현재 체크된 클래스들의 (실제) 이름 목록을 반환한다."""
        out = []
        for i in range(self.class_list.count()):
            it = self.class_list.item(i)
            if it.checkState() == Qt.Checked:
                # 표시 문구가 아니라 UserRole의 원래 이름을 모은다.
                out.append(it.data(Qt.UserRole))
        return out

    def _refresh_selection(self) -> None:
        """선택 요약 라벨을 갱신하고, 하나 이상 체크됐을 때만 AND 버튼을 활성화한다."""
        sel = self._checked_classes()
        self.sel_label.setText("선택: " + (", ".join(sel) if sel else "(없음)"))
        self.and_btn.setEnabled(bool(sel))

    def _emit_and_filter(self) -> None:
        """체크된 객체를 '모두' 포함한 이미지 경로를 조회해 신호로 내보내고 닫는다."""
        sel = self._checked_classes()
        if not sel:
            return
        try:
            # 선택된 클래스를 모두 포함(AND)하는 이미지 경로들을 그래프에서 조회.
            paths = self.graph.images_with_all(sel)
        except Exception:
            paths = []
        # 메인 윈도우가 받아 갤러리로 표시하도록 경로 목록을 방출한다.
        self.show_images.emit(paths)
        self.accept()
