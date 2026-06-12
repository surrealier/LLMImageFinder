"""Read-only dialog showing what the current index was built from and when."""

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
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_index_meta(path: Path, meta: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # informational only — never fail a build over it


def _dir_size_mb(path: Path) -> float:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    continue
    except OSError:
        pass
    return total / (1024 * 1024)


class IndexInfoDialog(QDialog):
    """Shows store stats + last-build metadata. ``rebuild_clicked`` is True when
    the user asked for a full rebuild from here (handled by MainWindow)."""

    def __init__(self, count: int, signature, meta: dict, current_root: str,
                 chroma_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("인덱스 정보")
        self.setMinimumWidth(520)
        self.rebuild_clicked = False

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        def row(label: str, value: str, color: str = "") -> None:
            lab = QLabel(value or "—")
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if color:
                lab.setStyleSheet(f"color:{color};")
            lab.setWordWrap(True)
            form.addRow(label, lab)

        row("색인된 레코드", f"{count}개")
        if signature:
            model_id, dim, gran = signature
            gran_txt = {"image": "이미지별", "folder": "폴더별"}.get(gran or "", gran or "기록 없음")
            row("임베딩 모델", f"{model_id} ({dim}차원)")
            row("색인 단위", gran_txt)
        built_root = str(meta.get("dataset_root", ""))
        if meta:
            row("마지막 빌드", str(meta.get("built_at", "—")))
            dur = meta.get("duration_s")
            if isinstance(dur, (int, float)):
                row("소요 시간", f"{dur:.1f}초")
            row("기록/건너뜀/정리", (
                f"{meta.get('n_written', '?')}개 기록 · "
                f"{meta.get('skipped', 0)}건 건너뜀 · {meta.get('pruned', 0)}개 정리"
            ))
            row("빌드 당시 데이터셋", built_root)
        mismatch = bool(built_root) and bool(current_root) and built_root != current_root
        row(
            "현재 데이터셋",
            current_root or "(미설정)",
            color="#ef4444" if mismatch else "",
        )
        if mismatch:
            row("주의", "인덱스가 다른 데이터셋으로 만들어졌습니다 — 전체 재빌드를 권장합니다.", "#ef4444")
        row("저장 공간", f"{_dir_size_mb(chroma_dir):.1f} MB ({chroma_dir})")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        if mismatch or count > 0:
            rebuild_btn = QPushButton("전체 재빌드…")
            rebuild_btn.clicked.connect(self._on_rebuild)
            buttons.addButton(rebuild_btn, QDialogButtonBox.ActionRole)
        root.addWidget(buttons)

    def _on_rebuild(self) -> None:
        self.rebuild_clicked = True
        self.accept()
