"""YOLO 클래스 ID -> 표시 이름을 편집하는 표(테이블) 다이얼로그.

인덱싱 전(데이터셋 라벨에서 이름 없는 클래스 ID가 발견됐을 때)과 도구 모음에서
열린다. 입력한 이름은 앱 데이터 폴더의 config.json + class_names.yaml에 저장되며,
데이터셋 폴더에는 절대 아무것도 쓰지 않는다(원본 데이터셋 불변 원칙).
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
    """클래스 ID 정렬 키. 숫자 ID는 숫자 순으로, 비숫자 ID는 그 뒤에 문자열 순으로 둔다.

    튜플의 첫 원소(0/1)가 그룹을 가른다: 숫자(0) 그룹이 먼저, 비숫자(1) 그룹이 나중.
    이렇게 해야 "10"이 "2"보다 뒤에 오는 등 자연스러운 숫자 정렬이 된다(문자열 정렬과 다름).
    """
    return (0, int(cid)) if cid.isdigit() else (1, cid)


class ClassNamesDialog(QDialog):
    """클래스 ID별 표시 이름을 입력받는 표 형식 다이얼로그."""

    def __init__(
        self,
        class_counts: dict[str, int],
        names: dict[str, str],
        parent=None,
    ) -> None:
        """다이얼로그를 구성한다.

        매개변수
        - class_counts: 데이터셋 라벨에서 집계한 {클래스 ID: 박스 개수}.
        - names: 이미 저장돼 있는 {클래스 ID: 이름}(있으면 미리 채워진다).
        """
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

        # 데이터셋에서 발견된 ID와 이미 이름이 있는 ID를 합집합으로 모아 정렬해 빠짐없이 보여준다.
        all_ids = sorted(set(class_counts) | set(names), key=_sort_key)
        # 3열 표: 클래스 ID / 박스 수 / 이름. 행 수는 모든 클래스 개수.
        self.table = QTableWidget(len(all_ids), 3, self)
        self.table.setHorizontalHeaderLabels(["클래스 ID", "박스 수", "이름"])
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        # ID/박스 수 열은 내용에 맞게 좁히고, 이름 열만 남은 공간을 늘려 차지하게 한다.
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        for row, cid in enumerate(all_ids):
            # 클래스 ID 칸: 읽기 전용(편집 플래그를 비트 제거). 사용자가 ID를 바꾸지 못하게 한다.
            id_item = QTableWidgetItem(cid)
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            # 박스 수 칸: 역시 읽기 전용이며 오른쪽 정렬(숫자 가독성).
            cnt_item = QTableWidgetItem(str(class_counts.get(cid, 0)))
            cnt_item.setFlags(cnt_item.flags() & ~Qt.ItemIsEditable)
            cnt_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            # 이름 칸만 편집 가능. 저장된 이름이 있으면 미리 채운다.
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
        """표에서 이름이 입력된 행만 모아 {클래스 ID: 이름} 사전으로 돌려준다.

        ID나 이름이 비어 있는 행은 건너뛴다(빈 이름은 저장하지 않음 → '#ID'로 표시되게 둠).
        """
        out: dict[str, str] = {}
        for row in range(self.table.rowCount()):
            cid = self.table.item(row, 0).text().strip()
            name_item = self.table.item(row, 2)
            # 이름 칸이 아직 한 번도 만들어지지 않았을 수 있어 None 방어 후 strip 한다.
            name = (name_item.text() if name_item else "").strip()
            # ID와 이름이 모두 채워진 행만 결과에 담는다.
            if cid and name:
                out[cid] = name
        return out
