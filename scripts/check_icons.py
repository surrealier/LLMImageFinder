"""Confirm every qtawesome icon name we use actually resolves (no blank fallbacks)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import qtawesome as qta
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

NAMES = [
    "fa5s.paper-plane", "fa5s.folder-open", "fa5s.sync-alt", "fa5s.redo",
    "fa5s.search", "fa5s.cog", "fa5s.images", "fa5s.search-plus",
    "fa5s.search-minus", "fa5s.expand", "fa5s.chevron-left", "fa5s.chevron-right",
    "fa5s.exclamation-triangle", "fa5s.times",
]
bad = []
for n in NAMES:
    try:
        ic = qta.icon(n)
        if ic.isNull():
            bad.append((n, "null"))
    except Exception as e:
        bad.append((n, repr(e)))

print("checked:", len(NAMES), "bad:", bad)
assert not bad, f"unresolved icons: {bad}"
print("ICONS OK")
