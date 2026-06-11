"""OS integration: reveal a file/folder in the system file manager."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices


def reveal_in_explorer(path: str) -> None:
    """On Windows, open Explorer with the file selected; else open the folder."""
    p = Path(path)
    if sys.platform.startswith("win"):
        try:
            if p.exists():
                # /select, highlights the file; explorer returns 1 even on success.
                subprocess.Popen(["explorer", "/select,", os.path.normpath(str(p))])
                return
        except Exception:
            pass
    folder = p if p.is_dir() else p.parent
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
