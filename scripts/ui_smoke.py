"""Offscreen smoke test of the GUI: construct everything + run threaded query path."""

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QElapsedTimer, QEventLoop  # noqa: E402

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

    # synchronous index to populate the store
    n = win.indexer.build(str(ds), full_rebuild=True)
    win._refresh_status()
    print("indexed:", n, "count:", win.store.count())

    # threaded query path
    win.run_query("불과 연기가 있는 이미지")
    timer = QElapsedTimer()
    timer.start()
    while win._query_runner is not None and timer.elapsed() < 8000:
        app.processEvents(QEventLoop.AllEvents, 50)
    rows = win.gallery._model.rowCount()
    print("gallery rows after query:", rows)
    assert rows > 0, "gallery should have results"

    # let thumbnail tasks run
    t2 = QElapsedTimer()
    t2.start()
    while t2.elapsed() < 1500:
        app.processEvents(QEventLoop.AllEvents, 50)

    # construct viewer + settings (no exec)
    from imgsearch.ui.image_viewer import ImageViewer
    from imgsearch.ui.settings_dialog import SettingsDialog

    hit = win.gallery.current_hit()
    members = win.service.member_images(hit.folder)
    v = ImageViewer(members, 0, win)
    print("viewer members:", len(members))
    v.close()

    dlg = SettingsDialog(cfg, win)
    rc = dlg.result_config()
    print("settings roundtrip backend:", rc.embedder_backend, "granularity:", rc.index_granularity)
    dlg.close()

    from imgsearch.ui.class_names_dialog import ClassNamesDialog

    cdlg = ClassNamesDialog({"0": 12, "1": 5}, {"0": "사람"}, win)
    names = cdlg.result_names()
    assert names == {"0": "사람"}, names
    print("class-names dialog roundtrip:", names)
    cdlg.close()

    win.close()
    print("\nUI SMOKE OK")


if __name__ == "__main__":
    main()
