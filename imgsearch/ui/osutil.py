"""OS 연동: 파일/폴더를 시스템 파일 관리자(탐색기/Finder 등)에서 열어 보여준다."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices


def reveal_in_explorer(path: str) -> None:
    """파일/폴더를 시스템 파일 관리자에서 연다.

    Windows에서는 가능하면 탐색기를 열되 해당 '파일을 선택(하이라이트)'한 상태로 띄운다.
    그 외 플랫폼이나 Windows에서 실패한 경우에는 단순히 '폴더'를 연다.
    """
    p = Path(path)
    if sys.platform.startswith("win"):
        try:
            if p.exists():
                # '/select,'는 폴더를 열면서 그 안의 파일을 강조 선택해 준다.
                # 주의: explorer는 성공해도 종료코드 1을 반환하므로, 반환값으로
                # 성공/실패를 판단하지 말 것. 그래서 Popen으로 띄우고 곧장 반환한다.
                # 또한 경로 구분자를 OS 표준(역슬래시)으로 정규화해야 탐색기가 인식한다.
                subprocess.Popen(["explorer", "/select,", os.path.normpath(str(p))])
                return
        except Exception:
            # 탐색기 실행 실패 시 아래의 공통 폴더 열기 폴백으로 넘어간다.
            pass
    # 폴백: 경로가 폴더면 그대로, 파일이면 그 상위 폴더를 OS 기본 방식으로 연다.
    folder = p if p.is_dir() else p.parent
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
