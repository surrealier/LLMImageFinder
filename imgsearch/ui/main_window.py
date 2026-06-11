"""The main window: chat (left) + results gallery (right), threaded index/query."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

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
from imgsearch.ui.mock_banner import MockBanner
from imgsearch.ui.osutil import reveal_in_explorer
from imgsearch.ui.results_gallery import ResultsGallery
from imgsearch.ui.settings_dialog import SettingsDialog
from imgsearch.workers.index_worker import IndexWorker
from imgsearch.workers.qworker import ThreadRunner
from imgsearch.workers.query_worker import QueryWorker


def _index_signature(cfg: AppConfig) -> tuple:
    return (cfg.embedder_backend, cfg.embedder_model, cfg.embed_dim, cfg.index_granularity)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, paths: AppPaths) -> None:
        super().__init__()
        self.cfg = config
        self.paths = paths
        self._index_runner: ThreadRunner | None = None
        self._query_runner: ThreadRunner | None = None
        self._index_mode = "build"
        self._refresh_after_build = False

        self.setWindowTitle("데이터셋 검색 — sLLM 이미지 탐색")
        self.resize(1240, 820)

        self._build_services()
        self._build_ui()
        self._post_init()

    # ---------------------------------------------------------------- services
    def _build_services(self) -> None:
        self.embedder, self.captioner, self.chat = build_backends(self.cfg)
        self.store = ChromaStore(self.paths.chroma_dir)
        self.service = SearchService(self.embedder, self.store, self.chat, self.cfg)
        self.indexer = Indexer(self.embedder, self.captioner, self.store, self.cfg)

    # ---------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        tb = self.addToolBar("main")
        tb.setMovable(False)
        self.act_index = tb.addAction(icons.index(), "인덱스 빌드/업데이트", self.start_index_update)
        self.act_rebuild = tb.addAction(icons.rebuild(), "전체 재빌드", self.start_full_rebuild)
        tb.addSeparator()
        tb.addAction(icons.tags(), "클래스 이름", self.edit_class_names)
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

        self.chat_widget = ChatWidget()
        self.chat_widget.submitted.connect(self.run_query)
        splitter.addWidget(self.chat_widget)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(4, 4, 4, 4)
        header = QHBoxLayout()
        title = QLabel("검색 결과 — 폴더별 대표 이미지")
        title.setStyleSheet("font-weight:600; color:#cbd5e1;")
        header.addWidget(title)
        header.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color:#94a3b8;")
        header.addWidget(self.count_label)
        self.open_folder_btn = QPushButton(icons.folder(), " 폴더 열기")
        self.open_folder_btn.setToolTip("선택한 이미지를 탐색기에서 열기")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self.open_selected_folder)
        header.addWidget(self.open_folder_btn)
        rlay.addLayout(header)

        self.gallery = ResultsGallery(thumb_size=self.cfg.thumb_size)
        self.gallery.activated_hit.connect(self.open_viewer)
        self.gallery.selected_hit.connect(self._on_select)
        rlay.addWidget(self.gallery, 1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([440, 800])

        # status bar: message + progress + cancel
        self.status_msg = QLabel("준비됨")
        self.statusBar().addWidget(self.status_msg, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(320)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.cancel_btn = QPushButton(icons.cancel(), " 취소")
        self.cancel_btn.setVisible(False)
        self.statusBar().addPermanentWidget(self.cancel_btn)

    def _post_init(self) -> None:
        if self.cfg.degraded:
            self.banner.show_reason(self.cfg.degraded_reason)
        elif self.cfg.embedder_backend == "mock":
            self.banner.show_reason("")
        self._refresh_status()
        if not self.cfg.dataset_root:
            self.chat_widget.add_system(
                "데이터셋 루트가 설정되지 않았습니다. 툴바의 ‘설정’에서 폴더를 지정하거나, "
                "‘샘플 데이터셋 생성’으로 예시 데이터를 만든 뒤 ‘인덱스 빌드’를 실행하세요."
            )

    def _unit_label(self) -> str:
        return "이미지" if self.cfg.index_granularity == "image" else "폴더"

    def _refresh_status(self) -> None:
        n = self.store.count()
        mode = self.cfg.embedder_backend
        self.count_label.setText(f"색인된 {self._unit_label()}: {n}개")
        self.status_msg.setText(f"준비됨 · 임베딩={mode} · 색인 {n}개")

    # ---------------------------------------------------------------- query
    def run_query(self, text: str) -> None:
        if self._query_runner is not None:
            return
        self.chat_widget.add_user(text)
        if self._index_runner is not None and self._index_runner.is_running():
            self.chat_widget.add_system("인덱싱이 진행 중입니다. 완료 후 다시 검색해 주세요.")
            return
        if self.store.count() == 0:
            self.chat_widget.add_system("인덱스가 비어 있습니다. 먼저 ‘인덱스 빌드/업데이트’를 실행하세요.")
            return
        self.chat_widget.set_busy(True)
        self.status_msg.setText("검색 중…")

        worker = QueryWorker(self.service, text, self.cfg.top_k)
        worker.finished.connect(self._on_query_done)
        worker.error.connect(self._on_query_error)
        self._query_runner = ThreadRunner(worker, self)
        self._query_runner.start()

    def _on_query_done(self, result) -> None:
        self.gallery.set_results(result.hits)
        self.count_label.setText(f"검색 결과: {len(result.hits)}개 폴더")
        if result.hits:
            self.chat_widget.add_assistant(
                result.summary or f"{len(result.hits)}개의 관련 폴더를 찾았습니다."
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

    def _start_index(self, full_rebuild: bool, refresh: bool = False) -> None:
        if self._index_runner is not None:
            return
        if self._query_runner is not None and self._query_runner.is_running():
            QMessageBox.information(self, "검색 진행 중", "검색이 끝난 뒤 실행하세요.")
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

        self.act_index.setEnabled(False)
        self.act_rebuild.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.cancel_btn.setVisible(True)

        size = self.cfg.thumb_size
        worker = IndexWorker(
            self.indexer, root, full_rebuild=full_rebuild,
            thumb_cb=lambda p: ensure_thumb(p, size),
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

    def _on_index_done(self, n: int) -> None:
        was_refresh = self._index_mode == "refresh"
        self._finish_index_ui()
        self._refresh_status()
        if was_refresh:
            if n > 0:
                self.chat_widget.add_system(f"캡션 갱신 완료 — {n}개 레코드에 새 이름이 반영되었습니다.")
            else:
                self.chat_widget.add_system(
                    "갱신된 레코드가 없습니다 — 인덱스와 색인 단위가 일치하는지 확인하거나 전체 재빌드를 실행하세요."
                )
        else:
            self.chat_widget.add_system(
                f"인덱싱 완료 — 총 {self.store.count()}개 {self._unit_label()}가 색인되었습니다."
            )
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
        members = self.service.member_images(hit.folder) or [hit.image_path]
        try:
            start = members.index(hit.image_path)
        except ValueError:
            start = 0
        ImageViewer(members, start, self).exec()

    def _on_select(self, hit) -> None:
        self.open_folder_btn.setEnabled(hit is not None and bool(hit.image_path))

    def open_selected_folder(self) -> None:
        hit = self.gallery.current_hit()
        if hit and hit.image_path:
            reveal_in_explorer(hit.image_path)

    # ---------------------------------------------------------------- settings / sample
    def open_settings(self) -> None:
        if self._index_runner is not None and self._index_runner.is_running():
            QMessageBox.information(self, "인덱싱 진행 중", "인덱싱이 끝난 뒤 설정을 변경하세요.")
            return
        before = _index_signature(self.cfg)
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() != SettingsDialog.Accepted:
            return
        self.cfg = dlg.result_config()
        self.cfg.save(self.paths.config_file)
        self._build_services()  # rebuild backends with new config
        self.gallery._thumb_size = self.cfg.thumb_size
        self.banner.setVisible(False)
        if self.cfg.degraded:
            self.banner.show_reason(self.cfg.degraded_reason)
        elif self.cfg.embedder_backend == "mock":
            self.banner.show_reason("")
        self._refresh_status()
        if _index_signature(self.cfg) != before and self.store.count() > 0:
            ok = QMessageBox.question(
                self, "색인 설정 변경",
                "임베딩 모델/차원 또는 색인 단위가 변경되었습니다. 기존 인덱스와 호환되지 않습니다.\n"
                "지금 전체 재빌드를 실행할까요?",
            )
            if ok == QMessageBox.Yes:
                self._start_index(full_rebuild=True)

    def make_sample(self) -> None:
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
        self.gallery.shutdown()  # drain thumbnail tasks before teardown
        super().closeEvent(event)
