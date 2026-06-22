"""Offscreen smoke test of the GUI: construct everything + run threaded query path."""

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Korean output on cp949 consoles

from PySide6.QtCore import QElapsedTimer, QEventLoop, QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402

from imgsearch.app import build_app  # noqa: E402
from imgsearch.config import AppConfig  # noqa: E402
from imgsearch.paths import AppPaths  # noqa: E402
from imgsearch.sample_data import generate_sample_dataset  # noqa: E402


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="imgsearch_ui_")
    os.environ["IMGSEARCH_HOME"] = tmp
    ds = generate_sample_dataset(os.path.join(tmp, "data"), per_scene=3)

    from imgsearch.paths import app_paths, ensure_dirs

    paths: AppPaths = ensure_dirs(app_paths())
    cfg = AppConfig(dataset_root=str(ds))

    app = build_app()
    from imgsearch.ui.main_window import MainWindow

    win = MainWindow(cfg, paths)
    win.show()
    print("window constructed:", win.windowTitle())
    assert win._backends_ready, "mock backends must be ready synchronously"

    # synchronous index to populate the store
    report = win.indexer.build(str(ds), full_rebuild=True)
    win._refresh_status()
    print("indexed:", report.n_written, "count:", win.store.count())
    assert not report.skipped, report.skipped

    # threaded query path — submitted through the chat input so history records
    win.chat_widget.input.setText("불과 연기가 있는 이미지")
    win.chat_widget._on_send()
    timer = QElapsedTimer()
    timer.start()
    while win._query_runner is not None and timer.elapsed() < 8000:
        app.processEvents(QEventLoop.AllEvents, 50)
    rows = win.gallery._model.rowCount()
    print("gallery rows after query:", rows)
    assert rows > 0, "gallery should have results"

    # query history: Up arrow must recall the query we just sent
    ev = QKeyEvent(QEvent.KeyPress, Qt.Key_Up, Qt.NoModifier)
    app.sendEvent(win.chat_widget.input, ev)
    assert win.chat_widget.input.text() == "불과 연기가 있는 이미지", win.chat_widget.input.text()
    win.chat_widget.input.clear()
    win.chat_widget._hist_pos = None
    print("history recall: OK")

    # score-threshold filter narrows the displayed set without re-querying
    n_all = len(win._display_hits)
    win.thr_spin.setValue(0.99)
    n_filtered = len(win._display_hits)
    print(f"threshold filter: {n_all} -> {n_filtered}")
    assert n_filtered <= n_all
    win.thr_spin.setValue(0.0)
    assert len(win._display_hits) == n_all

    # export: clipboard path list
    win.copy_result_paths()
    from PySide6.QtGui import QGuiApplication

    clip = QGuiApplication.clipboard().text()
    assert clip.splitlines(), "clipboard should hold result paths"
    print("clipboard export lines:", len(clip.splitlines()))

    # similar-image search reuses the stored embedding (synchronous)
    hit = win.gallery.current_hit()
    assert hit is not None
    win.run_similar(hit)
    print("similar-search rows:", win.gallery._model.rowCount())
    assert win.gallery._model.rowCount() > 0

    # let thumbnail tasks run
    t2 = QElapsedTimer()
    t2.start()
    while t2.elapsed() < 1500:
        app.processEvents(QEventLoop.AllEvents, 50)

    # construct viewer with class names + meta provider (no exec)
    from imgsearch.ui.image_viewer import ImageViewer
    from imgsearch.ui.settings_dialog import SettingsDialog

    members = win.service.member_images(hit.folder)
    v = ImageViewer(members, 0, win, class_names={"0": "사람"}, meta_provider=lambda p: f"점수 0.42 · {p}")
    assert not v.meta_label.isHidden(), "meta line should be shown when a provider returns text"
    v.boxes_btn.toggle()  # exercise the overlay toggle path
    v.next()
    print("viewer members:", len(members), "boxes toggled:", ImageViewer.show_boxes)
    v.close()

    dlg = SettingsDialog(cfg, win)
    rc = dlg.result_config()
    print("settings roundtrip backend:", rc.embedder_backend, "granularity:", rc.index_granularity)
    assert rc.embed_dim <= 1024
    dlg.close()

    from imgsearch.ui.class_names_dialog import ClassNamesDialog

    cdlg = ClassNamesDialog({"0": 12, "1": 5}, {"0": "사람"}, win)
    names = cdlg.result_names()
    assert names == {"0": "사람"}, names
    print("class-names dialog roundtrip:", names)
    cdlg.close()

    from imgsearch.ui.index_info_dialog import IndexInfoDialog, load_index_meta

    idlg = IndexInfoDialog(
        win.store.count(), win.store.stored_signature(),
        load_index_meta(paths.index_meta_file), cfg.dataset_root, paths.chroma_dir, win,
    )
    idlg.close()
    print("index info dialog constructed")

    # v0.3.0: search-mode switch (vector/hybrid/keyword) via the header combo
    win.mode_combo.setCurrentIndex(2)  # keyword
    assert win.cfg.search_mode == "keyword", win.cfg.search_mode
    win.mode_combo.setCurrentIndex(0)  # hybrid
    assert win.cfg.search_mode == "hybrid"
    win.chat_widget.input.setText("불과 연기")
    win.chat_widget._on_send()
    timer = QElapsedTimer(); timer.start()
    while win._query_runner is not None and timer.elapsed() < 8000:
        app.processEvents(QEventLoop.AllEvents, 50)
    print("hybrid query rows:", win.gallery._model.rowCount())
    assert win.gallery._model.rowCount() > 0

    # v0.3.0: agentic search path (mock planner + memory graph + hybrid)
    win.agent_check.setChecked(True)
    assert win.cfg.agentic_enabled
    # let the graph build worker finish
    t3 = QElapsedTimer(); t3.start()
    while win._graph_runner is not None and t3.elapsed() < 8000:
        app.processEvents(QEventLoop.AllEvents, 50)
    win.chat_widget.input.setText("불과 연기")
    win.chat_widget._on_send()
    timer = QElapsedTimer(); timer.start()
    while win._query_runner is not None and timer.elapsed() < 8000:
        app.processEvents(QEventLoop.AllEvents, 50)
    print("agentic query rows:", win.gallery._model.rowCount())
    win.agent_check.setChecked(False)

    # v0.3.0: object graph dialog
    from imgsearch.ui.graph_dialog import GraphDialog

    if win.graph.count() == 0:
        from imgsearch.graph.builder import build_records
        win.graph.build(build_records(str(ds), cfg))
    gdlg = GraphDialog(win.graph, win)
    print("graph dialog classes:", gdlg.class_list.count())
    gdlg.close()

    # v0.3.1: About + Keyboard-shortcuts dialogs construct and show the current version
    from imgsearch import __version__
    from imgsearch.ui.about_dialog import build_about_dialog, build_shortcuts_dialog

    adlg = build_about_dialog(win)
    assert "About" in adlg.windowTitle(), adlg.windowTitle()
    adlg.close()
    sdlg = build_shortcuts_dialog(win)
    assert "shortcuts" in sdlg.windowTitle().lower(), sdlg.windowTitle()
    sdlg.close()
    print("about/shortcuts dialogs constructed; version:", __version__)

    # v0.3.1: empty-state overlay text is driven by store/search state
    from imgsearch.ui.results_gallery import ResultsGallery

    g = ResultsGallery()
    g.set_empty_message("nothing here")
    assert g.result_count() == 0
    g.close()
    print("empty-state gallery OK")

    win.close()
    print("\nUI SMOKE OK")


if __name__ == "__main__":
    main()
