"""객체 그래프 — explore object co-occurrence and pull AND-filtered images.

Left: every object class with its image count (and a checkbox to AND-filter on).
Right: classes that co-occur with the selected class.
Bottom: '선택한 객체를 모두 포함한 이미지 보기' → emits image paths → the main window
turns them into gallery rows. All reads go through the GraphStore (memory or kuzu).
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
    show_images = Signal(list)  # list[str] image paths matching the AND filter

    def __init__(self, graph: GraphStore, parent=None) -> None:
        super().__init__(parent)
        self.graph = graph
        self.setWindowTitle("객체 그래프")
        self.resize(720, 520)

        root = QVBoxLayout(self)
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

        body = QHBoxLayout()
        root.addLayout(body, 1)

        left = QVBoxLayout()
        left.addWidget(QLabel("객체 (체크 = AND 필터)"))
        self.class_list = QListWidget()
        for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            it = QListWidgetItem(f"{name}  ·  {n}장")
            it.setData(Qt.UserRole, name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.class_list.addItem(it)
        self.class_list.currentItemChanged.connect(self._on_select_class)
        left.addWidget(self.class_list, 1)
        body.addLayout(left, 1)

        right = QVBoxLayout()
        self.co_title = QLabel("동시 출현 객체")
        right.addWidget(self.co_title)
        self.co_table = QTableWidget(0, 2)
        self.co_table.setHorizontalHeaderLabels(["객체", "동시 출현"])
        self.co_table.horizontalHeader().setStretchLastSection(True)
        self.co_table.verticalHeader().setVisible(False)
        self.co_table.setEditTriggers(QTableWidget.NoEditTriggers)
        right.addWidget(self.co_table, 1)
        body.addLayout(right, 1)

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

        self.class_list.itemChanged.connect(self._refresh_selection)
        self._refresh_selection()

    def _on_select_class(self, current: Optional[QListWidgetItem], _prev) -> None:
        self.co_table.setRowCount(0)
        if current is None:
            return
        name = current.data(Qt.UserRole)
        self.co_title.setText(f"‘{name}’와(과) 함께 나타나는 객체")
        try:
            rows = self.graph.cooccurring(name, top=50)
        except Exception:
            rows = []
        self.co_table.setRowCount(len(rows))
        for r, (other, n) in enumerate(rows):
            self.co_table.setItem(r, 0, QTableWidgetItem(str(other)))
            self.co_table.setItem(r, 1, QTableWidgetItem(str(n)))

    def _checked_classes(self) -> list[str]:
        out = []
        for i in range(self.class_list.count()):
            it = self.class_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def _refresh_selection(self) -> None:
        sel = self._checked_classes()
        self.sel_label.setText("선택: " + (", ".join(sel) if sel else "(없음)"))
        self.and_btn.setEnabled(bool(sel))

    def _emit_and_filter(self) -> None:
        sel = self._checked_classes()
        if not sel:
            return
        try:
            paths = self.graph.images_with_all(sel)
        except Exception:
            paths = []
        self.show_images.emit(paths)
        self.accept()
