"""현재 인덱스가 '무엇으로, 언제' 만들어졌는지 보여주는 읽기 전용 다이얼로그.

여기서 '전체 재빌드'를 누를 수 있으며, 실제 재빌드는 MainWindow가 수행한다
(이 다이얼로그는 rebuild_clicked 플래그만 세우고 닫힌다).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


def load_index_meta(path: Path) -> dict:
    """인덱스 메타데이터 JSON 파일을 읽어 사전으로 반환한다.

    파일이 없거나 깨졌거나 사전이 아니면 빈 사전을 돌려준다(부가정보이므로 절대 죽지 않게).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # 최상위가 객체가 아니면(예: 리스트/숫자) 신뢰할 수 없으므로 빈 사전으로 취급.
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_index_meta(path: Path, meta: dict) -> None:
    """빌드 메타데이터를 JSON으로 저장한다(없는 상위 폴더는 만들어 둔다).

    ensure_ascii=False로 한글이 깨지지 않게 그대로 쓴다. 저장 실패는 빌드를 막지 않는다.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # 어디까지나 참고용 정보 — 이것 때문에 빌드를 실패시키지 않는다


def _dir_size_mb(path: Path) -> float:
    """디렉터리 안 모든 파일 크기의 합을 MB 단위로 계산한다(저장 공간 표시용)."""
    total = 0
    try:
        # 하위 폴더까지 재귀적으로 훑어 모든 파일 크기를 더한다.
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    # 권한/경합 등으로 특정 파일 크기를 못 읽어도 그 파일만 건너뛴다.
                    continue
    except OSError:
        # 디렉터리 자체에 접근 못 하면 0으로 둔다.
        pass
    return total / (1024 * 1024)


class IndexInfoDialog(QDialog):
    """저장소 통계 + 마지막 빌드 메타데이터를 보여준다.

    사용자가 여기서 전체 재빌드를 요청하면 ``rebuild_clicked``가 True가 된다
    (실제 재빌드 처리는 MainWindow가 담당한다).
    """

    def __init__(self, count: int, signature, meta: dict, current_root: str,
                 chroma_dir: Path, parent=None) -> None:
        """통계/메타를 받아 폼 형태로 표시한다.

        매개변수
        - count: 현재 색인된 레코드 수.
        - signature: (모델 ID, 차원, 색인 단위) 튜플 또는 None(인덱스 없음).
        - meta: load_index_meta가 읽어온 마지막 빌드 메타데이터.
        - current_root: 지금 설정에 잡혀 있는 데이터셋 경로.
        - chroma_dir: 벡터 저장소(ChromaDB) 디렉터리(용량 계산/표시용).
        """
        super().__init__(parent)
        self.setWindowTitle("Index info")
        self.setMinimumWidth(520)
        # 재빌드 요청 여부. 호출자가 다이얼로그가 닫힌 뒤 이 값을 확인한다.
        self.rebuild_clicked = False

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        def row(label: str, value: str, color: str = "") -> None:
            """폼에 '라벨: 값' 한 줄을 추가하는 헬퍼. 값은 선택/복사 가능하게 한다."""
            # 값이 비면 '—'로 대체해 빈 줄처럼 보이지 않게 한다.
            lab = QLabel(value or "—")
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            # 경고 등 강조가 필요할 때만 색을 입힌다(예: 데이터셋 불일치 빨간색).
            if color:
                lab.setStyleSheet(f"color:{color};")
            lab.setWordWrap(True)
            form.addRow(label, lab)

        row("Indexed records", f"{count}")
        # 인덱스가 있을 때만 시그니처(모델/차원/단위)를 표시한다.
        if signature:
            model_id, dim, gran = signature
            # 내부 키("image"/"folder")를 사람이 읽을 한글로 바꾼다(없으면 원문/기록 없음).
            gran_txt = {"image": "Per image", "folder": "Per folder"}.get(gran or "", gran or "No record")
            row("Embedding model", f"{model_id} ({dim} dimensions)")
            row("Index unit", gran_txt)
        built_root = str(meta.get("dataset_root", ""))
        # 메타가 있으면 마지막 빌드 시각/소요시간/처리량/대상 데이터셋을 보여준다.
        if meta:
            row("Last build", str(meta.get("built_at", "—")))
            dur = meta.get("duration_s")
            # 숫자일 때만 '초' 단위로 포맷(누락/문자열인 경우 줄 자체를 생략).
            if isinstance(dur, (int, float)):
                row("Duration", f"{dur:.1f}s")
            row("Written / skipped / pruned", (
                f"{meta.get('n_written', '?')} written · "
                f"{meta.get('skipped', 0)} skipped · {meta.get('pruned', 0)} pruned"
            ))
            row("Dataset at build time", built_root)
        # 빌드 당시 경로와 현재 경로가 모두 있고 서로 다르면 '불일치'로 본다.
        # (인덱스가 다른 데이터셋으로 만들어진 상태 → 검색 결과가 엉뚱할 수 있음)
        mismatch = bool(built_root) and bool(current_root) and built_root != current_root
        row(
            "Current dataset",
            current_root or "(not set)",
            # 불일치면 빨간색으로 경고 강조.
            color="#ef4444" if mismatch else "",
        )
        if mismatch:
            row("Warning", "The index was built from a different dataset — a full rebuild is recommended.", "#ef4444")
        row("Storage", f"{_dir_size_mb(chroma_dir):.1f} MB ({chroma_dir})")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        # 데이터셋 불일치거나 색인된 레코드가 있을 때만 '전체 재빌드' 버튼을 노출한다
        # (인덱스가 비어 있으면 재빌드할 것도 없으므로 굳이 보여주지 않음).
        if mismatch or count > 0:
            rebuild_btn = QPushButton("Full rebuild…")
            rebuild_btn.clicked.connect(self._on_rebuild)
            buttons.addButton(rebuild_btn, QDialogButtonBox.ActionRole)
        root.addWidget(buttons)

    def _on_rebuild(self) -> None:
        """'전체 재빌드' 버튼: 플래그만 세우고 닫는다(실제 작업은 MainWindow가 수행)."""
        self.rebuild_clicked = True
        self.accept()
