"""Editable table of YOLO class id -> display name.

Shown before indexing (when unnamed class ids are detected in the dataset's labels)
and from the toolbar. Names are saved to config.json + class_names.yaml in the
app-data dir — nothing is ever written into the dataset folder.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def _sort_key(cid: str):
    return (0, int(cid)) if cid.isdigit() else (1, cid)


class ClassNamesDialog(QDialog):
    def __init__(
        self,
        class_counts: dict[str, int],
        names: dict[str, str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("클래스 이름 설정")
        self.setMinimumSize(460, 380)

        root = QVBoxLayout(self)
        info = QLabel(
            "데이터셋의 YOLO 라벨에서 발견된 클래스 ID입니다. 캡션과 검색 요약에 사용할 "
            "이름을 입력하세요. 비워두면 ‘#ID’로 표시됩니다.\n"
            "(이름은 앱 설정 폴더의 class_names.yaml로 저장되며, 데이터셋 폴더는 수정하지 않습니다)"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#94a3b8;")
        root.addWidget(info)

        all_ids = sorted(set(class_counts) | set(names), key=_sort_key)
        self.table = QTableWidget(len(all_ids), 3, self)
        self.table.setHorizontalHeaderLabels(["클래스 ID", "박스 수", "이름"])
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        for row, cid in enumerate(all_ids):
            id_item = QTableWidgetItem(cid)
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            cnt_item = QTableWidgetItem(str(class_counts.get(cid, 0)))
            cnt_item.setFlags(cnt_item.flags() & ~Qt.ItemIsEditable)
            cnt_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            name_item = QTableWidgetItem(names.get(cid, ""))
            self.table.setItem(row, 0, id_item)
            self.table.setItem(row, 1, cnt_item)
            self.table.setItem(row, 2, name_item)
        root.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def result_names(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for row in range(self.table.rowCount()):
            cid = self.table.item(row, 0).text().strip()
            name_item = self.table.item(row, 2)
            name = (name_item.text() if name_item else "").strip()
            if cid and name:
                out[cid] = name
        return out
