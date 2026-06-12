"""The main window: chat (left) + results gallery (right), threaded index/query.

Heavy backends (jina-clip GPU weights) load on a background thread after the
window is shown; until then queries/indexing are politely refused. The gallery
shows a display-side filtered view (score threshold) of the last query's hits.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime

from PySide6.QtCore import QRunnable, Qt, QThreadPool
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from imgsearch.backends.mock import MockChatLLM, MockEmbedder
from imgsearch.config import AppConfig, save_class_names_yaml
from imgsearch.core.registry import build_backends
from imgsearch.core.services import SearchService
from imgsearch.index import labels
from imgsearch.index.indexer import Indexer
from imgsearch.paths import AppPaths
from imgsearch.store.chroma_store import ChromaStore
from imgsearch.thumbs import ensure_thumb
from imgsearch.ui import icons
from imgsearch.ui.chat_widget import ChatWidget
from imgsearch.ui.class_names_dialog import ClassNamesDialog
from imgsearch.ui.image_viewer import ImageViewer
from imgsearch.ui.index_info_dialog import IndexInfoDialog, load_index_meta, save_index_meta
from imgsearch.ui.mock_banner import MockBanner
from imgsearch.ui.osutil import reveal_in_explorer
from imgsearch.ui.results_gallery import ResultsGallery
from imgsearch.ui.settings_dialog import SettingsDialog
from imgsearch.workers.index_worker import IndexWorker
from imgsearch.workers.preload_worker import PreloadWorker
from imgsearch.workers.qworker import ThreadRunner
from imgsearch.workers.query_worker import QueryWorker


def _index_signature(cfg: AppConfig) -> tuple:
    return (cfg.embedder_backend, cfg.embedder_model, cfg.embed_dim, cfg.index_granularity)


def _backend_signature(cfg: AppConfig) -> tuple:
    """Everything whose change requires reconstructing backend objects. Cheap
    settings (top_k, thumb_size, dataset_root, …) deliberately excluded so a
    settings save doesn't reload ~GB of model weights."""
    return (
        cfg.embedder_backend, cfg.embedder_model, cfg.embedder_device, cfg.embed_dim,
        cfg.caption_enabled, cfg.vlm_backend, cfg.vlm_base_url, cfg.vlm_model, cfg.vlm_api_key,
        cfg.chat_backend, cfg.chat_base_url, cfg.chat_model, cfg.chat_api_key,
    )


class _PrewarmTask(QRunnable):
    """Generate one thumbnail into the on-disk cache (index-time pre-warm)."""

    def __init__(self, path: str, size: int) -> None:
        super().__init__()
        self._path, self._size = path, size
        self.setAutoDelete(True)

    def run(self) -> None:
        ensure_thumb(self._path, self._size)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, paths: AppPaths) -> None:
        super().__init__()
        self.cfg = config
        self.paths = paths
        self._index_runner: ThreadRunner | None = None
        self._query_runner: ThreadRunner | None = None
        self._preload_runner: ThreadRunner | None = None
        self._index_mode = "build"
        self._refresh_after_build = False
        self._backends_ready = False
        self._build_t0 = 0.0
        self._last_hits: list = []  # raw hits of the last query
        self._display_hits: list = []  # after score-threshold filter

        self.setWindowTitle("데이터셋 검색 — sLLM 이미지 탐색")
        self.resize(1240, 820)

        # the store never depends on backend choice — create it once
        self.store = ChromaStore(self.paths.chroma_dir)
        # index-time thumbnail pre-warm runs here instead of on the index thread
        self._prewarm_pool = QThreadPool(self)
        self._prewarm_pool.setMaxThreadCount(2)

        self._build_ui()
        self._setup_backends()
        self._post_init()

    # ---------------------------------------------------------------- services
    def _wire_services(self) -> None:
        self.service = SearchService(self.embedder, self.store, self.chat, self.cfg)
        self.indexer = Indexer(self.embedder, self.captioner, self.store, self.cfg)

    def _setup_backends(self) -> None:
        """Mock backends build instantly in-place; real ones load off-thread so
        the window paints immediately instead of freezing for the model load."""
        if self.cfg.embedder_backend == "mock":
            self.embedder, self.captioner, self.chat = build_backends(self.cfg)
            self._wire_services()
            self._backends_ready = True
            self._update_model_chip()
            self._apply_banner()
            return
        # instant placeholders keep the UI alive; queries are gated until ready
        self.embedder = MockEmbedder(dim=self.cfg.embed_dim)
        self.captioner = None
        self.chat = MockChatLLM()
        self._wire_services()
        self._backends_ready = False
        self._start_preload()

    def _start_preload(self) -> None:
        self._update_model_chip()
        self.status_msg.setText("임베딩 모델 로딩 중…")
        self.act_index.setEnabled(False)
        self.act_rebuild.setEnabled(False)
        worker = PreloadWorker(self.cfg)
        worker.finished.connect(self._on_preload_done)
        worker.error.connect(self._on_preload_error)
        self._preload_runner = ThreadRunner(worker, self)
        self._preload_runner.start()

    def _on_preload_done(self, backends) -> None:
        self.embedder, self.captioner, self.chat = backends
        self._wire_services()
        self._backends_ready = True
        self._preload_runner = None
        self.act_index.setEnabled(True)
        self.act_rebuild.setEnabled(True)
        self._update_model_chip()
        self._apply_banner()
        self._refresh_status()
        name = getattr(self.embedder, "name", "?")
        self.chat_widget.add_system(f"모델 로딩 완료 ({name}) — 검색을 시작할 수 있습니다.")

    def _on_preload_error(self, msg: str) -> None:
        # build_backends falls back to mock internally, so this is unexpected —
        # stay on the placeholder mock backends and tell the user why
        self._backends_ready = True
        self._preload_runner = None
        self.act_index.setEnabled(True)
        self.act_rebuild.setEnabled(True)
        self.cfg.mark_degraded(f"백엔드 로드 실패: {msg}")
        self._update_model_chip()
        self._apply_banner()
        self._refresh_status()

    def _apply_banner(self) -> None:
        self.banner.setVisible(False)
        if self.cfg.degraded:
            self.banner.show_reason(self.cfg.degraded_reason)
        elif self.cfg.embedder_backend == "mock":
            self.banner.show_reason("")

    def _update_model_chip(self) -> None:
        if not self._backends_ready:
            self.model_chip.setText("모델 로딩 중…")
            self.model_chip.setStyleSheet("color:#f59e0b; padding:0 8px;")
            return
        name = getattr(self.embedder, "name", "?")
        device = getattr(self.embedder, "device", "")
        self.model_chip.setText(f"임베딩: {name}" + (f" ({device})" if device else ""))
        self.model_chip.setStyleSheet("color:#34d399; padding:0 8px;")

    # ---------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        tb = self.addToolBar("main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.act_index = tb.addAction(icons.index(), "인덱스 빌드/업데이트", self.start_index_update)
        self.act_index.setShortcut(QKeySequence("F5"))
        self.act_index.setToolTip("변경된 파일만 색인하고 삭제된 레코드를 정리합니다 (F5)")
        self.act_rebuild = tb.addAction(icons.rebuild(), "전체 재빌드", self.start_full_rebuild)
        self.act_rebuild.setToolTip("기존 인덱스를 모두 지우고 처음부터 다시 색인합니다")
        tb.addSeparator()
        tb.addAction(icons.tags(), "클래스 이름", self.edit_class_names)
        tb.addAction(icons.info(), "인덱스 정보", self.show_index_info)
        tb.addSeparator()
        tb.addAction(icons.images(), "샘플 데이터셋 생성", self.make_sample)
        tb.addSeparator()
        tb.addAction(icons.settings(), "설정", self.open_settings)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        self.banner = MockBanner()
        self.banner.open_settings.connect(self.open_settings)
        outer.addWidget(self.banner)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)

        self.chat_widget = ChatWidget(history_file=self.paths.history_file)
        self.chat_widget.submitted.connect(self.run_query)
        splitter.addWidget(self.chat_widget)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(4, 4, 4, 4)
        header = QHBoxLayout()
        title = QLabel("검색 결과")
        title.setStyleSheet("font-weight:600; color:#cbd5e1;")
        header.addWidget(title)
        header.addStretch(1)

        header.addWidget(QLabel("결과 수"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 200)
        self.k_spin.setValue(int(self.cfg.top_k))
        self.k_spin.setToolTip("다음 검색부터 가져올 결과 개수 (top-k)")
        self.k_spin.valueChanged.connect(self._on_k_changed)
        header.addWidget(self.k_spin)

        header.addWidget(QLabel("점수 ≥"))
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0.0, 1.0)
        self.thr_spin.setSingleStep(0.05)
        self.thr_spin.setDecimals(2)
        self.thr_spin.setValue(float(self.cfg.score_threshold))
        self.thr_spin.setToolTip("이 유사도 점수 미만의 결과를 숨깁니다 (재검색 없이 즉시 적용)")
        self.thr_spin.valueChanged.connect(self._on_threshold_changed)
        header.addWidget(self.thr_spin)

        self.export_btn = QToolButton()
        self.export_btn.setIcon(icons.export())
        self.export_btn.setText(" 내보내기")
        self.export_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.export_btn.setPopupMode(QToolButton.InstantPopup)
        emenu = QMenu(self.export_btn)
        emenu.addAction("CSV로 저장…", self.export_csv)
        emenu.addAction("경로 목록 복사", self.copy_result_paths)
        self.export_btn.setMenu(emenu)
        self.export_btn.setEnabled(False)
        header.addWidget(self.export_btn)

        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color:#94a3b8;")
        header.addWidget(self.count_label)
        self.open_folder_btn = QPushButton(icons.folder(), " 폴더 열기")
        self.open_folder_btn.setToolTip("선택한 이미지를 탐색기에서 열기 (Ctrl+E)")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self.open_selected_folder)
        header.addWidget(self.open_folder_btn)
        rlay.addLayout(header)

        self.gallery = ResultsGallery(thumb_size=self.cfg.thumb_size)
        self.gallery.activated_hit.connect(self.open_viewer)
        self.gallery.selected_hit.connect(self._on_select)
        self.gallery.similar_requested.connect(self.run_similar)
        rlay.addWidget(self.gallery, 1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([440, 800])

        # status bar: message + model chip + progress + cancel
        self.status_msg = QLabel("준비됨")
        self.statusBar().addWidget(self.status_msg, 1)
        self.model_chip = QLabel("")
        self.statusBar().addPermanentWidget(self.model_chip)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(320)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.cancel_btn = QPushButton(icons.cancel(), " 취소")
        self.cancel_btn.setVisible(False)
        self.statusBar().addPermanentWidget(self.cancel_btn)

        QShortcut(QKeySequence("Ctrl+L"), self, activated=self._focus_search)

    def _focus_search(self) -> None:
        self.chat_widget.input.setFocus()
        self.chat_widget.input.selectAll()

    def _post_init(self) -> None:
        self._refresh_status()
        if not self.cfg.dataset_root:
            self.chat_widget.add_system(
                "데이터셋 루트가 설정되지 않았습니다. 툴바의 ‘설정’에서 폴더를 지정하거나, "
                "‘샘플 데이터셋 생성’으로 예시 데이터를 만든 뒤 ‘인덱스 빌드’를 실행하세요."
            )
        # warn when the stored index was built from a DIFFERENT dataset root
        meta = load_index_meta(self.paths.index_meta_file)
        built_root = str(meta.get("dataset_root", ""))
        if (
            built_root
            and self.cfg.dataset_root
            and built_root != self.cfg.dataset_root
            and self.store.count() > 0
        ):
            self.chat_widget.add_system(
                "주의: 현재 인덱스는 다른 데이터셋 경로에서 생성되었습니다. "
                "툴바의 ‘인덱스 정보’를 확인하고 필요하면 전체 재빌드를 실행하세요."
            )

    def _unit_label(self) -> str:
        return "이미지" if self.cfg.index_granularity == "image" else "폴더"

    def _refresh_status(self) -> None:
        n = self.store.count()
        self.count_label.setText(f"색인된 {self._unit_label()}: {n}개")
        self.status_msg.setText(f"준비됨 · 색인 {n}개")
        self._update_model_chip()

    # ---------------------------------------------------------------- query
    def _ready_for_query(self) -> bool:
        if self._index_runner is not None and self._index_runner.is_running():
            self.chat_widget.add_system("인덱싱이 진행 중입니다. 완료 후 다시 검색해 주세요.")
            return False
        if not self._backends_ready:
            self.chat_widget.add_system("임베딩 모델을 로딩하는 중입니다 — 잠시 후 다시 시도해 주세요.")
            return False
        if self.store.count() == 0:
            self.chat_widget.add_system("인덱스가 비어 있습니다. 먼저 ‘인덱스 빌드/업데이트’를 실행하세요.")
            return False
        return True

    def run_query(self, text: str) -> None:
        if self._query_runner is not None:
            return
        self.chat_widget.add_user(text)
        if not self._ready_for_query():
            return
        self.chat_widget.set_busy(True)
        self.status_msg.setText("검색 중…")

        worker = QueryWorker(self.service, text, self.cfg.top_k)
        worker.finished.connect(self._on_query_done)
        worker.error.connect(self._on_query_error)
        self._query_runner = ThreadRunner(worker, self)
        self._query_runner.start()

    def run_similar(self, hit) -> None:
        """Query-by-example: reuses the stored embedding, so it's fast enough
        to run synchronously (no model call in the common path)."""
        if hit is None or self._query_runner is not None:
            return
        if not self._ready_for_query():
            return
        name = os.path.basename(hit.image_path or hit.folder)
        self.chat_widget.add_user(f"(유사 이미지 검색) {name}")
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = self.service.query_by_example(hit, self.cfg.top_k)
        except Exception as e:
            self.chat_widget.add_system(f"유사 검색 오류: {type(e).__name__}: {e}")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()
        self._last_hits = list(result.hits)
        self._apply_display_filter()
        self.chat_widget.add_assistant(
            f"'{name}'와(과) 유사한 결과 {len(result.hits)}개를 찾았습니다."
        )

    def _apply_display_filter(self) -> None:
        thr = float(self.cfg.score_threshold)
        hits = [h for h in self._last_hits if h.score >= thr]
        self._display_hits = hits
        self.gallery.set_results(hits)
        self.export_btn.setEnabled(bool(hits))
        if self._last_hits and len(hits) != len(self._last_hits):
            self.count_label.setText(f"표시 {len(hits)} / 검색 {len(self._last_hits)}개")
        else:
            self.count_label.setText(f"검색 결과: {len(hits)}개")

    def _on_query_done(self, result) -> None:
        self._last_hits = list(result.hits)
        self._apply_display_filter()
        if result.hits:
            self.chat_widget.add_assistant(
                result.summary or f"{len(result.hits)}개의 관련 결과를 찾았습니다."
            )
        else:
            self.chat_widget.add_system("일치하는 결과가 없습니다. 다른 표현으로 검색해 보세요.")
        self.chat_widget.set_busy(False)
        self.status_msg.setText("준비됨")
        self._query_runner = None

    def _on_query_error(self, msg: str) -> None:
        self.chat_widget.add_system(f"검색 오류: {msg}")
        self.chat_widget.set_busy(False)
        self.status_msg.setText("오류")
        self._query_runner = None

    def _on_k_changed(self, value: int) -> None:
        self.cfg.top_k = int(value)
        self.cfg.save(self.paths.config_file)

    def _on_threshold_changed(self, value: float) -> None:
        self.cfg.score_threshold = float(value)
        self.cfg.save(self.paths.config_file)
        if self._last_hits:
            self._apply_display_filter()

    # ---------------------------------------------------------------- export
    def export_csv(self) -> None:
        hits = self._display_hits
        if not hits:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "검색 결과를 CSV로 저장", "search_results.csv", "CSV 파일 (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["score", "caption", "image_path", "folder", "member_count"])
                for h in hits:
                    w.writerow([f"{h.score:.4f}", h.caption, h.image_path, h.folder, h.member_count])
        except OSError as e:
            QMessageBox.warning(self, "저장 실패", str(e))
            return
        self.chat_widget.add_system(f"CSV 저장 완료 — {len(hits)}행: {path}")

    def copy_result_paths(self) -> None:
        hits = self._display_hits
        if not hits:
            return
        QGuiApplication.clipboard().setText("\n".join(h.image_path for h in hits))
        self.chat_widget.add_system(f"이미지 경로 {len(hits)}개를 클립보드에 복사했습니다.")

    # ---------------------------------------------------------------- indexing
    def start_index_update(self) -> None:
        self._start_index(full_rebuild=False)

    def start_full_rebuild(self) -> None:
        if self.store.count() > 0:
            ok = QMessageBox.question(
                self, "전체 재빌드",
                "기존 인덱스를 모두 지우고 처음부터 다시 색인합니다. 계속할까요?",
            )
            if ok != QMessageBox.Yes:
                return
        self._start_index(full_rebuild=True)

    def _scan_class_ids(self, root: str) -> dict[str, int]:
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            counts = labels.scan_class_ids(root, self.cfg.image_exts)
        finally:
            QGuiApplication.restoreOverrideCursor()
        return {str(k): int(v) for k, v in counts.items()}

    def _prompt_class_names(self, class_counts: dict[str, int]) -> bool:
        """Open the editor; persist to config + class_names.yaml. True if changed."""
        dlg = ClassNamesDialog(class_counts, dict(self.cfg.class_names), self)
        if dlg.exec() != ClassNamesDialog.Accepted:
            return False
        new_names = dlg.result_names()
        if new_names == self.cfg.class_names:
            return False
        self.cfg.class_names = new_names
        self.cfg.save(self.paths.config_file)
        save_class_names_yaml(self.paths.class_names_file, new_names)
        self.chat_widget.add_system(
            f"클래스 이름 {len(new_names)}개 저장됨 → {self.paths.class_names_file}"
        )
        return True

    def edit_class_names(self) -> None:
        if self._index_runner is not None and self._index_runner.is_running():
            QMessageBox.information(self, "인덱싱 진행 중", "인덱싱이 끝난 뒤 변경하세요.")
            return
        if self._query_runner is not None and self._query_runner.is_running():
            QMessageBox.information(self, "검색 진행 중", "검색이 끝난 뒤 변경하세요.")
            return
        root = self.cfg.dataset_root
        counts: dict[str, int] = {}
        if root and os.path.isdir(root):
            counts = self._scan_class_ids(root)
        if not counts and not self.cfg.class_names:
            QMessageBox.information(
                self, "클래스 없음",
                "데이터셋에서 YOLO 라벨 파일을 찾지 못했습니다. (이름은 라벨이 있는 데이터셋에서만 사용됩니다)",
            )
            return
        changed = self._prompt_class_names(counts)
        # apply to the existing index quickly (caption-only, no re-embedding) —
        # only when the stored records really are image-granularity
        sig = self.store.stored_signature()
        gran_ok = sig is None or sig[2] in (None, "image")
        if (
            changed
            and self.store.count() > 0
            and self.cfg.index_granularity == "image"
            and gran_ok
        ):
            ok = QMessageBox.question(
                self, "캡션 갱신",
                "저장된 인덱스의 캡션에 새 이름을 반영할까요?\n(재임베딩 없이 라벨 캡션만 빠르게 갱신합니다)",
            )
            if ok == QMessageBox.Yes:
                self._start_index(full_rebuild=False, refresh=True)

    def _queue_prewarm(self, path: str) -> None:
        # called from the INDEX thread via thumb_cb — QThreadPool.start is
        # thread-safe, and moving the decode here keeps the index thread on
        # captioning/embedding instead of JPEG work
        self._prewarm_pool.start(_PrewarmTask(path, self.cfg.thumb_size))

    def _start_index(self, full_rebuild: bool, refresh: bool = False) -> None:
        if self._index_runner is not None:
            self.chat_widget.add_system("이미 인덱싱이 진행 중입니다 — 완료 후 다시 실행하세요.")
            return
        if self._query_runner is not None and self._query_runner.is_running():
            QMessageBox.information(self, "검색 진행 중", "검색이 끝난 뒤 실행하세요.")
            return
        if not self._backends_ready:
            self.chat_widget.add_system("임베딩 모델을 로딩하는 중입니다 — 완료 후 다시 시도해 주세요.")
            return
        root = self.cfg.dataset_root
        if not root or not os.path.isdir(root):
            QMessageBox.warning(self, "데이터셋 없음", "유효한 데이터셋 루트를 먼저 설정하세요.")
            self.open_settings()
            return

        # first build of a labeled dataset: offer to name the classes up front.
        # Scanned only while no names exist yet — avoids a full dataset walk on
        # the UI thread for every subsequent build.
        names_changed = False
        if not refresh and self.cfg.index_granularity == "image" and not self.cfg.class_names:
            counts = self._scan_class_ids(root)
            if counts:
                ok = QMessageBox.question(
                    self, "클래스 이름 입력",
                    f"YOLO 라벨에서 클래스 {len(counts)}종이 발견되었습니다.\n"
                    "캡션/요약에 사용할 이름을 지금 입력할까요?",
                )
                if ok == QMessageBox.Yes:
                    names_changed = self._prompt_class_names(counts)
        # names changed right before an incremental build: already-indexed,
        # unchanged records would keep old captions — chain a refresh afterwards
        self._refresh_after_build = (
            names_changed and not full_rebuild and self.store.count() > 0
        )
        self._index_mode = "refresh" if refresh else "build"
        self._build_t0 = time.time()

        self.act_index.setEnabled(False)
        self.act_rebuild.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.cancel_btn.setVisible(True)

        worker = IndexWorker(
            self.indexer, root, full_rebuild=full_rebuild,
            thumb_cb=self._queue_prewarm,
            mode="refresh" if refresh else "build",
        )
        worker.progress.connect(self._on_index_progress)
        worker.log.connect(self.status_msg.setText)
        worker.finished.connect(self._on_index_done)
        worker.error.connect(self._on_index_error)
        # DirectConnection: the worker thread is blocked inside run() (no event
        # loop), so a queued slot would only fire AFTER the run finishes. Direct
        # invocation sets the bool from the GUI thread — GIL-atomic and polled.
        self.cancel_btn.clicked.connect(worker.cancel, Qt.DirectConnection)
        self._index_runner = ThreadRunner(worker, self)
        self._index_runner.start()
        self.chat_widget.add_system("캡션 갱신을 시작합니다…" if refresh else "인덱싱을 시작합니다…")

    def _on_index_progress(self, p) -> None:
        if p.phase == "scan":
            self.progress.setRange(0, 0)
            self.status_msg.setText(f"스캔 중… {p.current} 폴더 발견")
        elif p.phase in ("index", "caption"):
            if p.total:
                self.progress.setRange(0, p.total)
                self.progress.setValue(p.current)
            verb = "캡션 갱신" if p.phase == "caption" else "인덱싱"
            self.status_msg.setText(f"{verb} {p.current}/{p.total} — {os.path.basename(p.folder)}")

    def _finish_index_ui(self) -> None:
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        try:
            self.cancel_btn.clicked.disconnect()  # drop this run's worker.cancel wiring
        except (RuntimeError, TypeError):
            pass
        self.act_index.setEnabled(True)
        self.act_rebuild.setEnabled(True)
        self._index_runner = None

    def _write_index_meta(self, report) -> None:
        save_index_meta(self.paths.index_meta_file, {
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self._build_t0, 1),
            "dataset_root": self.cfg.dataset_root,
            "granularity": self.cfg.index_granularity,
            "model_id": getattr(self.embedder, "name", "?"),
            "embed_dim": int(getattr(self.embedder, "dim", 0)),
            "n_written": report.n_written,
            "skipped": len(report.skipped),
            "pruned": report.pruned,
        })

    def _on_index_done(self, result) -> None:
        was_refresh = self._index_mode == "refresh"
        self._finish_index_ui()
        self._refresh_status()
        if was_refresh:
            n = int(result)
            if n > 0:
                self.chat_widget.add_system(f"캡션 갱신 완료 — {n}개 레코드에 새 이름이 반영되었습니다.")
            else:
                self.chat_widget.add_system(
                    "갱신된 레코드가 없습니다 — 인덱스와 색인 단위가 일치하는지 확인하거나 전체 재빌드를 실행하세요."
                )
            return

        report = result  # BuildReport
        if report.cancelled:
            msg = f"인덱싱이 취소되었습니다 — 현재 {self.store.count()}개 레코드가 저장되어 있습니다."
        else:
            msg = f"인덱싱 완료 — 총 {self.store.count()}개 {self._unit_label()}가 색인되었습니다."
        if report.pruned:
            msg += f" (삭제된 파일 레코드 {report.pruned}개 정리)"
        self.chat_widget.add_system(msg)
        if not report.cancelled:
            self._write_index_meta(report)
        if report.skipped:
            self.chat_widget.add_system(f"{len(report.skipped)}개 항목은 오류로 건너뛰었습니다 (자세한 내용 표시됨).")
            box = QMessageBox(
                QMessageBox.Warning, "일부 항목 건너뜀",
                f"{len(report.skipped)}개 항목이 오류로 색인되지 않았습니다.\n"
                "자세히 보기에서 전체 목록을 확인할 수 있습니다.",
                QMessageBox.Ok, self,
            )
            box.setDetailedText("\n".join(f"{p} — {e}" for p, e in report.skipped[:500]))
            box.exec()
        if self._refresh_after_build:
            self._refresh_after_build = False
            self.chat_widget.add_system("기존 레코드의 캡션에 새 클래스 이름을 반영합니다…")
            self._start_index(full_rebuild=False, refresh=True)

    def _on_index_error(self, msg: str) -> None:
        self._finish_index_ui()
        QMessageBox.critical(self, "인덱싱 오류", msg)
        self.status_msg.setText("오류")

    # ---------------------------------------------------------------- viewer / folder
    def open_viewer(self, hit) -> None:
        # image granularity: navigate the RESULTS in ranking order, with score +
        # caption shown; folder granularity keeps folder-member navigation
        if self.cfg.index_granularity == "image" and self._display_hits:
            paths = [h.image_path for h in self._display_hits]
            try:
                start = paths.index(hit.image_path)
            except ValueError:
                paths, start = [hit.image_path], 0
            by_path = {h.image_path: h for h in self._display_hits}

            def meta(p: str) -> str:
                h = by_path.get(p)
                if h is None:
                    return ""
                rank = paths.index(p) + 1 if p in by_path else 0
                return f"결과 {rank}/{len(paths)} · 점수 {h.score:.2f} · {h.caption}" if h.caption \
                    else f"결과 {rank}/{len(paths)} · 점수 {h.score:.2f}"

            ImageViewer(
                paths, start, self, class_names=self.cfg.class_names, meta_provider=meta
            ).exec()
            return
        members = self.service.member_images(hit.folder) or [hit.image_path]
        try:
            start = members.index(hit.image_path)
        except ValueError:
            start = 0
        ImageViewer(members, start, self, class_names=self.cfg.class_names).exec()

    def _on_select(self, hit) -> None:
        self.open_folder_btn.setEnabled(hit is not None and bool(hit.image_path))

    def open_selected_folder(self) -> None:
        hit = self.gallery.current_hit()
        if hit and hit.image_path:
            reveal_in_explorer(hit.image_path)

    # ---------------------------------------------------------------- info / settings / sample
    def show_index_info(self) -> None:
        meta = load_index_meta(self.paths.index_meta_file)
        dlg = IndexInfoDialog(
            self.store.count(), self.store.stored_signature(), meta,
            self.cfg.dataset_root, self.paths.chroma_dir, self,
        )
        dlg.exec()
        if dlg.rebuild_clicked:
            self.start_full_rebuild()

    def open_settings(self) -> None:
        if self._index_runner is not None and self._index_runner.is_running():
            QMessageBox.information(self, "인덱싱 진행 중", "인덱싱이 끝난 뒤 설정을 변경하세요.")
            return
        if self._query_runner is not None and self._query_runner.is_running():
            QMessageBox.information(self, "검색 진행 중", "검색이 끝난 뒤 설정을 변경하세요.")
            return
        before_index = _index_signature(self.cfg)
        before_backend = _backend_signature(self.cfg)
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() != SettingsDialog.Accepted:
            return
        self.cfg = dlg.result_config()
        self.cfg.save(self.paths.config_file)
        if _backend_signature(self.cfg) != before_backend:
            self._setup_backends()  # may load real weights on a worker thread
        else:
            # cheap settings only: re-point the live objects at the new config
            self.service.cfg = self.cfg
            self.indexer.cfg = self.cfg
            self._update_model_chip()
            self._apply_banner()
        self.gallery.set_thumb_size(self.cfg.thumb_size)
        self.k_spin.setValue(int(self.cfg.top_k))
        self._refresh_status()
        if _index_signature(self.cfg) != before_index and self.store.count() > 0:
            ok = QMessageBox.question(
                self, "색인 설정 변경",
                "임베딩 모델/차원 또는 색인 단위가 변경되었습니다. 기존 인덱스와 호환되지 않습니다.\n"
                "지금 전체 재빌드를 실행할까요?",
            )
            if ok == QMessageBox.Yes:
                self._start_index(full_rebuild=True)

    def make_sample(self) -> None:
        if self._index_runner is not None and self._index_runner.is_running():
            QMessageBox.information(self, "인덱싱 진행 중", "인덱싱이 끝난 뒤 실행하세요.")
            return
        target = QFileDialog.getExistingDirectory(self, "샘플 데이터셋을 생성할 폴더 선택")
        if not target:
            return
        from imgsearch.sample_data import generate_sample_dataset

        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ds = generate_sample_dataset(os.path.join(target, "sample_dataset"))
        finally:
            QGuiApplication.restoreOverrideCursor()
        self.cfg.dataset_root = str(ds)
        self.cfg.save(self.paths.config_file)
        self._refresh_status()
        self.chat_widget.add_system(f"샘플 데이터셋을 생성했습니다: {ds}")
        if QMessageBox.question(self, "인덱스 빌드", "지금 인덱스를 빌드할까요?") == QMessageBox.Yes:
            self._start_index(full_rebuild=True)

    # ---------------------------------------------------------------- lifecycle
    def closeEvent(self, event) -> None:  # noqa: N802
        if self._index_runner is not None and self._index_runner.is_running():
            self._index_runner.worker.cancel()
            self._index_runner.wait(8000)
        if self._query_runner is not None and self._query_runner.is_running():
            self._query_runner.wait(8000)
        if self._preload_runner is not None and self._preload_runner.is_running():
            # a torch weight load can't be aborted — wait it out (rarely hit)
            QGuiApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._preload_runner.wait(60000)
            finally:
                QGuiApplication.restoreOverrideCursor()
        self._prewarm_pool.clear()
        if not self._prewarm_pool.waitForDone(2000):
            self._prewarm_pool.setParent(None)  # don't let a wedged decode block exit
        self.gallery.shutdown()  # drain thumbnail tasks before teardown
        super().closeEvent(event)
