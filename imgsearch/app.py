"""Application entrypoint: theme, config load, main window."""

from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from imgsearch.config import AppConfig
from imgsearch.logging_setup import setup_logging
from imgsearch.paths import app_paths, ensure_dirs

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
    app.setStyle("Fusion")
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
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ImgSearch")
    app.setOrganizationName("MarkAny")
    _apply_theme(app)
    return app


def main() -> int:
    # quick CLI: `python -m imgsearch --make-sample <dir>`
    if "--make-sample" in sys.argv:
        from imgsearch.sample_data import generate_sample_dataset

        i = sys.argv.index("--make-sample")
        out = sys.argv[i + 1] if len(sys.argv) > i + 1 else "sample_dataset"
        path = generate_sample_dataset(out)
        print(f"Sample dataset created at: {path}")
        return 0

    paths = ensure_dirs(app_paths())
    setup_logging(paths.logs_dir)
    cfg = AppConfig.load(paths.config_file)

    # class_names.yaml (app-data, user-editable) is the source of truth for names:
    # manual edits win over config.json; if only config has names, materialize the yaml.
    from imgsearch.config import load_class_names_yaml, save_class_names_yaml

    yaml_names = load_class_names_yaml(paths.class_names_file)
    if yaml_names and yaml_names != cfg.class_names:
        cfg.class_names = yaml_names
        cfg.save(paths.config_file)
    elif cfg.class_names and not paths.class_names_file.exists():
        save_class_names_yaml(paths.class_names_file, cfg.class_names)

    app = build_app()
    from imgsearch.ui.main_window import MainWindow

    win = MainWindow(cfg, paths)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
