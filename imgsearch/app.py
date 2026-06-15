"""애플리케이션 진입점: 테마 적용, 설정 로드, 메인 윈도우 생성."""

from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from imgsearch.config import AppConfig
from imgsearch.logging_setup import setup_logging
from imgsearch.paths import app_paths, ensure_dirs

# 다크 테마 스타일시트(QSS). 색상/여백 등 앱 전반의 외형을 한 곳에서 정의한다.
# 주의: 이 문자열은 런타임에 그대로 사용되는 스타일 정의이므로 내용을 바꾸지 말 것.
_QSS = """
QWidget { font-size: 13px; }
QMainWindow, QWidget { background: #0f141c; color: #e5e7eb; }
QLineEdit, QTextBrowser, QComboBox, QSpinBox {
    background: #161c26; border: 1px solid #2b3342; border-radius: 6px; padding: 5px;
    selection-background-color: #2563eb;
}
QPushButton {
    background: #1f2735; border: 1px solid #2b3342; border-radius: 6px; padding: 6px 10px;
}
QPushButton:hover { background: #283242; }
QPushButton:disabled { color: #5b6675; }
QGroupBox { border: 1px solid #2b3342; border-radius: 8px; margin-top: 10px; padding-top: 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9aa4b2; }
QToolBar { background: #131822; border-bottom: 1px solid #232b38; spacing: 4px; padding: 4px; }
QStatusBar { background: #131822; }
QProgressBar { border: 1px solid #2b3342; border-radius: 6px; text-align: center; }
QProgressBar::chunk { background: #2563eb; border-radius: 5px; }
QListView { background: #11161f; border: 1px solid #232b38; border-radius: 8px; }
QSplitter::handle { background: #232b38; }
"""


def _apply_theme(app: QApplication) -> None:
    """Fusion 스타일 + 다크 팔레트 + QSS를 적용해 일관된 다크 테마를 구성한다."""
    # 플랫폼별 기본 스타일 대신 Fusion을 쓰면 모든 OS에서 동일한 룩을 얻을 수 있고
    # 팔레트 색상이 위젯에 더 일관되게 반영된다.
    app.setStyle("Fusion")
    # 팔레트는 표준 위젯의 기본 색을, QSS는 세부 외형을 담당한다(둘을 함께 적용).
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#0f141c"))
    pal.setColor(QPalette.WindowText, QColor("#e5e7eb"))
    pal.setColor(QPalette.Base, QColor("#161c26"))
    pal.setColor(QPalette.AlternateBase, QColor("#1b212c"))
    pal.setColor(QPalette.Text, QColor("#e5e7eb"))
    pal.setColor(QPalette.Button, QColor("#1f2735"))
    pal.setColor(QPalette.ButtonText, QColor("#e5e7eb"))
    pal.setColor(QPalette.Highlight, QColor("#2563eb"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor("#1f2735"))
    pal.setColor(QPalette.ToolTipText, QColor("#e5e7eb"))
    pal.setColor(QPalette.PlaceholderText, QColor("#6b7689"))
    app.setPalette(pal)
    app.setStyleSheet(_QSS)


def build_app() -> QApplication:
    """QApplication 인스턴스를 가져오거나 새로 만들고, 메타데이터와 테마를 설정한다."""
    # 이미 인스턴스가 있으면 재사용(테스트 등 한 프로세스에서 중복 생성 방지).
    app = QApplication.instance() or QApplication(sys.argv)
    # 앱/조직 이름은 QSettings 등이 사용자 설정 경로를 결정할 때 활용된다.
    app.setApplicationName("ImgSearch")
    app.setOrganizationName("MarkAny")
    _apply_theme(app)
    return app


def main() -> int:
    """앱 부트스트랩: CLI 처리 → 경로/로깅/설정 준비 → 메인 윈도우 실행."""
    # 간단한 CLI 진입: `python -m imgsearch --make-sample <dir>`
    # GUI를 띄우지 않고 샘플 데이터셋만 만들고 종료한다(데모/테스트 준비용).
    if "--make-sample" in sys.argv:
        from imgsearch.sample_data import generate_sample_dataset

        i = sys.argv.index("--make-sample")
        # 인자가 더 있으면 그 경로를, 없으면 기본 디렉터리명을 출력 위치로 쓴다.
        out = sys.argv[i + 1] if len(sys.argv) > i + 1 else "sample_dataset"
        path = generate_sample_dataset(out)
        print(f"Sample dataset created at: {path}")
        return 0

    # 사용자별 앱 데이터 디렉터리들을 확정하고 실제로 생성해 둔다.
    paths = ensure_dirs(app_paths())
    setup_logging(paths.logs_dir)
    cfg = AppConfig.load(paths.config_file)

    # class_names.yaml(앱 데이터, 사용자가 직접 편집 가능)이 클래스명의 단일 진실 원천이다:
    # 수동 편집한 yaml이 config.json보다 우선하고, config에만 이름이 있으면 yaml로 구체화한다.
    from imgsearch.config import load_class_names_yaml, save_class_names_yaml

    yaml_names = load_class_names_yaml(paths.class_names_file)
    if yaml_names and yaml_names != cfg.class_names:
        # 사용자가 yaml을 손봤다 → 그 값을 config에 반영하고 저장(yaml이 승리).
        cfg.class_names = yaml_names
        cfg.save(paths.config_file)
    elif cfg.class_names and not paths.class_names_file.exists():
        # config에만 이름이 있고 yaml 파일이 아직 없다 → 편집 가능한 yaml로 만들어 준다.
        save_class_names_yaml(paths.class_names_file, cfg.class_names)

    app = build_app()
    # MainWindow는 무거운 임포트라 GUI 초기화 직전까지 미뤄 시작 비용을 줄인다.
    from imgsearch.ui.main_window import MainWindow

    win = MainWindow(cfg, paths)
    win.show()
    # Qt 이벤트 루프 진입. 창이 닫히면 반환되는 종료 코드를 그대로 돌려준다.
    return app.exec()


if __name__ == "__main__":
    # main()의 반환값(종료 코드)을 프로세스 종료 코드로 전달한다.
    raise SystemExit(main())
