"""Render README screenshots offscreen (no real display needed).

Builds the app in MOCK mode against small generated datasets, drives a query, and
grabs the window/dialogs to PNGs under docs/screenshots/. Deterministic + dependency
-free, so anyone can regenerate the shots with:  uv run python scripts/make_screenshots.py
"""

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw  # noqa: E402
from PySide6.QtCore import QElapsedTimer, QEventLoop  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "docs", "screenshots")
os.makedirs(OUT, exist_ok=True)


def _pump(app, ms):
    t = QElapsedTimer()
    t.start()
    while t.elapsed() < ms:
        app.processEvents(QEventLoop.AllEvents, 30)


def _wait_query(app, win, ms=8000):
    t = QElapsedTimer()
    t.start()
    while win._query_runner is not None and t.elapsed() < ms:
        app.processEvents(QEventLoop.AllEvents, 30)


_CLASSES = {"0": "사람", "1": "오토바이", "2": "자동차", "3": "신호등"}
_COLORS = [(229, 115, 115), (100, 181, 246), (129, 199, 132), (255, 213, 79)]


def _make_labeled_dataset(root, n=12):
    """A small YOLO-labeled dataset (drawn shapes) so graph/agentic have content."""
    os.makedirs(root, exist_ok=True)
    combos = [
        ["0", "1"], ["0", "2"], ["0", "1", "2"], ["1"], ["0", "3"], ["2", "3"],
        ["0", "1", "3"], ["0"], ["1", "2"], ["0", "2", "3"], ["3"], ["0", "1", "2", "3"],
    ]
    for i in range(n):
        classes = combos[i % len(combos)]
        bg = _COLORS[i % len(_COLORS)]
        im = Image.new("RGB", (320, 240), (bg[0] // 3, bg[1] // 3, bg[2] // 3))
        d = ImageDraw.Draw(im)
        lines = []
        for j, c in enumerate(classes):
            col = _COLORS[int(c)]
            cx, cy = 0.25 + 0.2 * (j % 3), 0.3 + 0.2 * (j // 3)
            w, h = 0.18, 0.22
            x0, y0 = (cx - w / 2) * 320, (cy - h / 2) * 240
            x1, y1 = (cx + w / 2) * 320, (cy + h / 2) * 240
            d.rectangle([x0, y0, x1, y1], fill=col, outline=(250, 250, 250), width=2)
            lines.append(f"{c} {cx:.3f} {cy:.3f} {w:.3f} {h:.3f}")
        p = os.path.join(root, f"frame_{i:02d}.jpg")
        im.save(p, "JPEG", quality=90)
        with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def main() -> None:
    from imgsearch.app import build_app

    app = build_app()

    # offscreen Qt ships no fonts and won't enumerate Windows fonts, so Korean glyphs
    # get dropped. Load Malgun Gothic (the Windows Korean UI font) explicitly so the
    # screenshots render Hangul correctly.
    from PySide6.QtGui import QFont, QFontDatabase

    for fp in (r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunsl.ttf"):
        if os.path.exists(fp):
            fid = QFontDatabase.addApplicationFont(fp)
            fams = QFontDatabase.applicationFontFamilies(fid)
            if fams:
                app.setFont(QFont(fams[0], 10))
                print("font:", fams[0])
                break

    # ---------------- shot 1: hero (sample scenes, hybrid search) ----------------
    home1 = tempfile.mkdtemp(prefix="shot_home1_")
    os.environ["IMGSEARCH_HOME"] = home1
    from imgsearch.config import AppConfig
    from imgsearch.paths import app_paths, ensure_dirs
    from imgsearch.sample_data import generate_sample_dataset

    paths = ensure_dirs(app_paths())
    ds1 = generate_sample_dataset(os.path.join(home1, "data"), per_scene=4)
    cfg = AppConfig(dataset_root=str(ds1), search_mode="hybrid")
    cfg.save(paths.config_file)

    from imgsearch.ui.main_window import MainWindow

    win = MainWindow(cfg, paths)
    win.resize(1280, 820)
    win.show()
    win.indexer.build(str(ds1), full_rebuild=True)
    win._refresh_status()
    win.chat_widget.input.setText("불과 연기가 있는 이미지")
    win.chat_widget._on_send()
    _wait_query(app, win)
    _pump(app, 1500)  # let thumbnails decode
    win.grab().save(os.path.join(OUT, "01_search.png"))
    print("saved 01_search.png")
    win.close()

    # ------------- shots 2-4: labeled dataset (agentic, graph, viewer) -------------
    home2 = tempfile.mkdtemp(prefix="shot_home2_")
    os.environ["IMGSEARCH_HOME"] = home2
    paths2 = ensure_dirs(app_paths())
    ds2 = os.path.join(home2, "data", "scene")
    _make_labeled_dataset(ds2)
    root2 = os.path.join(home2, "data")
    cfg2 = AppConfig(
        dataset_root=root2, index_granularity="image", class_names=_CLASSES,
        search_mode="hybrid", agentic_enabled=True,
    )
    cfg2.save(paths2.config_file)

    win2 = MainWindow(cfg2, paths2)
    win2.resize(1280, 820)
    win2.show()
    win2.indexer.build(root2, full_rebuild=True)
    win2._refresh_status()
    from imgsearch.graph.builder import build_records

    win2.graph.build(build_records(root2, cfg2))

    # agentic query -> chat shows the A2A trace, gallery shows provenance badges
    win2.chat_widget.input.setText("사람과 오토바이가 함께 있는 이미지")
    win2.chat_widget._on_send()
    _wait_query(app, win2)
    _pump(app, 1500)
    win2.grab().save(os.path.join(OUT, "02_agentic.png"))
    print("saved 02_agentic.png")

    # graph dialog
    from imgsearch.ui.graph_dialog import GraphDialog

    gdlg = GraphDialog(win2.graph, win2)
    gdlg.resize(760, 540)
    gdlg.show()
    if gdlg.class_list.count():
        gdlg.class_list.setCurrentRow(0)  # select first class -> co-occurrence table
    _pump(app, 400)
    gdlg.grab().save(os.path.join(OUT, "03_graph.png"))
    print("saved 03_graph.png")
    gdlg.close()

    # viewer with YOLO bbox overlay
    from imgsearch.ui.image_viewer import ImageViewer

    imgs = sorted(
        os.path.join(ds2, f) for f in os.listdir(ds2) if f.endswith(".jpg")
    )
    v = ImageViewer(imgs, 2, win2, class_names=_CLASSES,
                    meta_provider=lambda p: f"결과 3/12 · 점수 0.41 · 탐지 객체: 사람, 오토바이, 자동차")
    v.resize(820, 660)
    v.show()
    _pump(app, 500)
    v.grab().save(os.path.join(OUT, "04_viewer.png"))
    print("saved 04_viewer.png")
    v.close()
    win2.close()

    print("\nSCREENSHOTS DONE ->", OUT)


if __name__ == "__main__":
    main()
