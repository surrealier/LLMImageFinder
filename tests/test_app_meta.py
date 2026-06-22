"""v0.3.1: 패키지 메타데이터(버전 단일 출처)와 새 UI 표면(About/단축키/빈 상태) 테스트."""

from __future__ import annotations

import os

import pytest

import imgsearch

# Qt 위젯 테스트는 오프스크린 플랫폼에서 돌려야 헤드리스 CI에서도 안전하다.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_version_is_semver_like() -> None:
    """__version__은 'X.Y.Z' 형태의 비어 있지 않은 문자열이어야 한다."""
    v = imgsearch.__version__
    assert isinstance(v, str) and v
    parts = v.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts), v


def test_installed_metadata_matches_dunder_version() -> None:
    """pyproject의 동적 버전(설치 메타데이터)이 __init__.py의 __version__과 일치해야 한다.

    이 테스트가 'pyproject가 imgsearch/__init__.py에서 버전을 읽는다'는 단일 출처 배선을 지킨다.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("imgsearch")
    except PackageNotFoundError:
        pytest.skip("imgsearch is not installed as distribution metadata")
    assert installed == imgsearch.__version__


@pytest.fixture(scope="module")
def qapp():
    """프로세스당 하나의 QApplication(위젯 생성에 필요)."""
    from imgsearch.app import build_app

    return build_app()


def test_about_dialog_builds_and_shows_version(qapp) -> None:
    """About 다이얼로그가 예외 없이 생성되고 현재 버전 문자열을 포함해야 한다."""
    from imgsearch.ui.about_dialog import REPO_URL, build_about_dialog

    dlg = build_about_dialog()
    try:
        assert "About" in dlg.windowTitle()
        # 다이얼로그의 어떤 라벨에든 현재 버전이 노출되어야 한다.
        from PySide6.QtWidgets import QLabel

        texts = " ".join(lbl.text() for lbl in dlg.findChildren(QLabel))
        assert imgsearch.__version__ in texts
        assert "github.com/surrealier/LLMImageFinder" in REPO_URL
    finally:
        dlg.deleteLater()


def test_shortcuts_dialog_lists_known_keys(qapp) -> None:
    """단축키 다이얼로그가 생성되고 대표 단축키(F5/Ctrl+L)를 담아야 한다."""
    from imgsearch.ui.about_dialog import _SHORTCUTS, build_shortcuts_dialog

    keys = {k for k, _ in _SHORTCUTS}
    assert "F5" in keys and "Ctrl+L" in keys
    dlg = build_shortcuts_dialog()
    try:
        assert "shortcut" in dlg.windowTitle().lower()
    finally:
        dlg.deleteLater()


def test_gallery_empty_message_and_count(qapp) -> None:
    """빈 갤러리는 result_count()==0이고, 안내 문구 설정이 예외 없이 동작해야 한다."""
    from imgsearch.ui.results_gallery import ResultsGallery

    g = ResultsGallery()
    try:
        assert g.result_count() == 0
        g.set_empty_message("No index yet.")
        assert g.result_count() == 0  # 안내 문구는 결과 수를 바꾸지 않는다
    finally:
        g.deleteLater()


def test_app_icon_is_constructible(qapp) -> None:
    """코드 생성 앱 아이콘이 예외 없이 만들어져야 한다(글꼴이 있으면 비어 있지 않음)."""
    from imgsearch.ui.appicon import app_icon

    icon = app_icon()
    # qtawesome 글꼴 누락 시 빈 아이콘으로 폴백하므로 isNull 단언은 하지 않는다 — 생성만 검증.
    assert icon is not None
