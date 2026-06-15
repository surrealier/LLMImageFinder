"""qtawesome icon helpers (FontAwesome 5 solid)."""

from __future__ import annotations

import qtawesome as qta
from PySide6.QtGui import QIcon

_MUTED = "#9aa4b2"


def _i(name: str, color: str = _MUTED, **kw) -> QIcon:
    try:
        return qta.icon(name, color=color, **kw)
    except Exception:
        return QIcon()


def send() -> QIcon:
    return _i("fa5s.paper-plane", "#3b82f6")


def folder() -> QIcon:
    return _i("fa5s.folder-open", "#eab308")


def index() -> QIcon:
    return _i("fa5s.sync-alt", "#22c55e")


def rebuild() -> QIcon:
    # warning color: this action wipes the existing index
    return _i("fa5s.redo", "#f59e0b")


def search() -> QIcon:
    return _i("fa5s.search")


def settings() -> QIcon:
    return _i("fa5s.cog")


def images() -> QIcon:
    return _i("fa5s.images")


def zoom_in() -> QIcon:
    return _i("fa5s.search-plus")


def zoom_out() -> QIcon:
    return _i("fa5s.search-minus")


def fit() -> QIcon:
    return _i("fa5s.expand")


def prev() -> QIcon:
    return _i("fa5s.chevron-left")


def nxt() -> QIcon:
    return _i("fa5s.chevron-right")


def tags() -> QIcon:
    return _i("fa5s.tags", "#a78bfa")


def warn() -> QIcon:
    return _i("fa5s.exclamation-triangle", "#f59e0b")


def cancel() -> QIcon:
    return _i("fa5s.times", "#ef4444")


def copy() -> QIcon:
    return _i("fa5s.copy")


def export() -> QIcon:
    return _i("fa5s.file-export", "#34d399")


def info() -> QIcon:
    return _i("fa5s.info-circle", "#60a5fa")


def similar() -> QIcon:
    return _i("fa5s.clone", "#7dd3fc")


def eye() -> QIcon:
    return _i("fa5s.eye")


def graph() -> QIcon:
    return _i("fa5s.project-diagram", "#a78bfa")
