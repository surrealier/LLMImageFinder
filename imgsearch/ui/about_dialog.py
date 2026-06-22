"""'정보'(About) 및 '키보드 단축키' 다이얼로그.

앱 이름/버전/설명/링크를 한곳에서 보여 주고, 흩어져 있던 단축키를 한 화면에 모아
사용자가 기능을 발견하기 쉽게 만든다. 버전은 ``imgsearch.__version__`` 단일 출처에서 읽는다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from imgsearch import __version__
from imgsearch.ui.appicon import app_icon

# 저장소/문서 링크 — About 다이얼로그와 README가 같은 곳을 가리키도록 한 곳에 모아 둔다.
REPO_URL = "https://github.com/surrealier/LLMImageFinder"
DOCS_URL = "https://github.com/surrealier/LLMImageFinder/blob/master/docs/ARCHITECTURE.md"

# (단축키, 설명) 목록 — 코드 곳곳에 흩어진 실제 바인딩을 한 곳에 모아 문서화한다.
# main_window.py(F5/Ctrl+L), results_gallery.py(Enter/Ctrl+C/Ctrl+E), chat_widget.py(↑/↓),
# image_viewer.py(←/→/Space, +/-, F, B, Esc)와 일치해야 한다.
_SHORTCUTS: list[tuple[str, str]] = [
    ("F5", "Build / update index"),
    ("Ctrl+L", "Focus the chat input"),
    ("Enter", "Search (in the chat) · open the selected result (in the gallery)"),
    ("↑ / ↓", "Recall previous searches (in the chat)"),
    ("Ctrl+C", "Copy the selected image path (in the gallery)"),
    ("Ctrl+E", "Reveal the selected image in Explorer (in the gallery)"),
    ("← / →", "Previous / next image (in the viewer)"),
    ("+ / −", "Zoom in / out (in the viewer)"),
    ("F", "Fit the image to the window (in the viewer)"),
    ("B", "Show / hide YOLO label boxes for labeled images (in the viewer)"),
    ("Esc", "Close the viewer"),
]


def build_about_dialog(parent: QWidget | None = None) -> QDialog:
    """앱 정보 다이얼로그를 만들어 반환한다(exec는 호출부/테스트가 결정).

    구성과 표시를 분리해 둔 이유: 오프스크린 스모크/단위 테스트가 exec()로 블로킹되지 않고
    다이얼로그를 검사할 수 있게 하기 위함이다.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("About LLMImageFinder")
    dlg.setWindowIcon(app_icon())
    root = QVBoxLayout(dlg)
    root.setContentsMargins(20, 18, 20, 14)
    root.setSpacing(10)

    # 상단: 아이콘 + 제목/버전
    top = QHBoxLayout()
    icon_label = QLabel()
    icon_label.setPixmap(app_icon().pixmap(56, 56))
    top.addWidget(icon_label)
    head = QLabel(
        f"<div style='font-size:18px;font-weight:700;color:#e5e7eb;'>🔎 LLMImageFinder</div>"
        f"<div style='color:#94a3b8;'>version {__version__}</div>"
    )
    head.setTextFormat(Qt.RichText)
    top.addWidget(head, 1)
    root.addLayout(top)

    # 본문: 설명 + mock-first 안내 + 링크/라이선스. 링크는 클릭 시 외부 브라우저로 열리게 한다.
    body = QLabel(
        "<p style='color:#cbd5e1;'>Find an image in your folder by <i>describing</i> it — "
        "a local, offline-first desktop app fusing multilingual CLIP vector search, BM25 keyword "
        "search, an embedded object-graph DB, and a multi-agent RAG pipeline.</p>"
        "<p style='color:#94a3b8;'>Runs fully in a deterministic <b>mock mode</b> with no ML "
        "dependencies; real models (jina-clip, vLLM, kùzu) activate from Settings.</p>"
        f"<p><a style='color:#60a5fa;' href='{REPO_URL}'>GitHub repository</a> &nbsp;·&nbsp; "
        f"<a style='color:#60a5fa;' href='{DOCS_URL}'>Architecture &amp; design</a></p>"
        "<p style='color:#6b7689;'>MIT License · Python 3.12 · PySide6 / Qt</p>"
    )
    body.setTextFormat(Qt.RichText)
    body.setWordWrap(True)
    body.setOpenExternalLinks(True)  # 링크 클릭 시 OS 기본 브라우저로 연다
    body.setMinimumWidth(440)
    root.addWidget(body)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok)
    buttons.accepted.connect(dlg.accept)
    root.addWidget(buttons)
    return dlg


def show_about(parent: QWidget | None = None) -> None:
    """앱 정보 다이얼로그를 모달로 띄운다(이름·버전·설명·mock-first 안내·링크·라이선스)."""
    build_about_dialog(parent).exec()


def build_shortcuts_dialog(parent: QWidget | None = None) -> QDialog:
    """키보드 단축키 목록 다이얼로그를 만들어 반환한다(exec는 호출부/테스트가 결정)."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("Keyboard shortcuts")
    dlg.setWindowIcon(app_icon())
    root = QVBoxLayout(dlg)
    root.setContentsMargins(20, 16, 20, 12)
    root.setSpacing(8)

    title = QLabel("<div style='font-size:15px;font-weight:700;color:#e5e7eb;'>Keyboard shortcuts</div>")
    title.setTextFormat(Qt.RichText)
    root.addWidget(title)

    # 단축키 표를 간단한 HTML 테이블로 렌더(키는 굵게, 설명은 흐리게).
    rows = "".join(
        f"<tr><td style='padding:3px 16px 3px 0;color:#7dd3fc;font-weight:600;"
        f"white-space:nowrap;'>{key}</td>"
        f"<td style='padding:3px 0;color:#cbd5e1;'>{desc}</td></tr>"
        for key, desc in _SHORTCUTS
    )
    table = QLabel(f"<table style='border-spacing:0;'>{rows}</table>")
    table.setTextFormat(Qt.RichText)
    root.addWidget(table)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok)
    buttons.accepted.connect(dlg.accept)
    root.addWidget(buttons)
    return dlg


def show_shortcuts(parent: QWidget | None = None) -> None:
    """키보드 단축키 목록 다이얼로그를 모달로 띄운다(앱 전역 + 갤러리 + 뷰어)."""
    build_shortcuts_dialog(parent).exec()
