"""메인 윈도우: 좌측 채팅 + 우측 결과 갤러리, 색인/검색을 스레드로 처리.

무거운 백엔드(jina-clip GPU 가중치)는 윈도우가 표시된 뒤 백그라운드 스레드에서
로딩된다. 로딩이 끝나기 전까지는 검색/색인 요청을 정중히 거부한다(UI가 먼저 그려지도록).
갤러리는 마지막 검색 결과(hits)에 대해 표시 단계 필터(점수 임계값)를 적용한
뷰를 보여 준다 — 즉 재검색 없이 화면에서만 걸러낸다.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime

from PySide6.QtCore import QRunnable, Qt, QThreadPool
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from imgsearch.core.agentic import AgenticSearch
from imgsearch.core.registry import build_backends
from imgsearch.core.services import SearchService
from imgsearch.graph.registry import build_graph_store
from imgsearch.index import labels
from imgsearch.index.indexer import Indexer, image_id
from imgsearch.paths import AppPaths
from imgsearch.store.chroma_store import ChromaStore
from imgsearch.thumbs import ensure_thumb
from imgsearch.ui import icons
from imgsearch.ui.chat_widget import ChatWidget
from imgsearch.ui.class_names_dialog import ClassNamesDialog
from imgsearch.ui.graph_dialog import GraphDialog
from imgsearch.ui.image_viewer import ImageViewer
from imgsearch.ui.index_info_dialog import IndexInfoDialog, load_index_meta, save_index_meta
from imgsearch.ui.mock_banner import MockBanner
from imgsearch.ui.osutil import reveal_in_explorer
from imgsearch.ui.results_gallery import ResultsGallery
from imgsearch.ui.settings_dialog import SettingsDialog
from imgsearch.workers.graph_worker import GraphBuildWorker
from imgsearch.workers.index_worker import IndexWorker
from imgsearch.workers.preload_worker import PreloadWorker
from imgsearch.workers.qworker import ThreadRunner
from imgsearch.workers.query_worker import QueryWorker

# 검색 방식 콤보박스용 (표시 라벨, 내부 값) 쌍. 표시 문자열은 한국어, 값은 코드에서 쓰는 식별자.
_MODE_LABELS = [("Hybrid", "hybrid"), ("Vector", "vector"), ("Keyword", "keyword")]


def _index_signature(cfg: AppConfig) -> tuple:
    """기존 인덱스와의 호환성을 좌우하는 설정들의 서명(signature).

    이 튜플이 바뀌면 저장된 인덱스는 더 이상 호환되지 않으므로 전체 재빌드가 필요하다.
    (임베딩 백엔드/모델/차원이 바뀌면 벡터 자체가 달라지고, 색인 단위가 바뀌면
    레코드 구성이 달라지기 때문)
    """
    return (cfg.embedder_backend, cfg.embedder_model, cfg.embed_dim, cfg.index_granularity)


def _backend_signature(cfg: AppConfig) -> tuple:
    """백엔드 객체를 새로 만들어야 하는 변경 항목 전체.

    top_k, thumb_size, dataset_root 같은 '가벼운' 설정은 일부러 제외한다 —
    그래야 단순 설정 저장만으로 수 GB짜리 모델 가중치를 다시 로드하지 않는다.
    (이 서명이 그대로면 백엔드를 재구성하지 않고 살아 있는 객체에 새 cfg만 꽂는다)"""
    return (
        cfg.embedder_backend, cfg.embedder_model, cfg.embedder_device, cfg.embed_dim,
        cfg.caption_enabled, cfg.vlm_backend, cfg.vlm_base_url, cfg.vlm_model, cfg.vlm_api_key,
        cfg.chat_backend, cfg.chat_base_url, cfg.chat_model, cfg.chat_api_key,
    )


class _PrewarmTask(QRunnable):
    """썸네일 하나를 디스크 캐시에 미리 생성하는 작업(색인 시점의 사전 워밍업).

    색인이 끝난 직후 갤러리에서 썸네일을 즉시 보여 줄 수 있도록, 색인 중에
    미리 디코딩/리사이즈해 캐시에 저장해 둔다. QRunnable이라 스레드풀에서 실행된다.
    """

    def __init__(self, path: str, size: int) -> None:
        """워밍업할 이미지 경로와 목표 썸네일 크기를 받는다.

        setAutoDelete(True): 실행이 끝나면 QThreadPool이 이 객체를 자동 삭제하므로
        호출 측에서 수명을 따로 관리할 필요가 없다.
        """
        super().__init__()
        self._path, self._size = path, size
        self.setAutoDelete(True)

    def run(self) -> None:
        """스레드풀 워커에서 호출 — 썸네일을 캐시에 생성(이미 있으면 ensure_thumb가 건너뜀)."""
        ensure_thumb(self._path, self._size)


class MainWindow(QMainWindow):
    """앱 전체를 묶는 메인 윈도우 — 백엔드/서비스 생성, 색인·검색 스레드 조율, UI 이벤트 처리."""

    def __init__(self, config: AppConfig, paths: AppPaths) -> None:
        """설정/경로를 받아 스토어·그래프·스레드풀을 만들고 UI와 백엔드를 초기화한다.

        무거운 백엔드 로딩은 _setup_backends 안에서 (mock이 아니면) 백그라운드로 미뤄지므로
        생성자는 빠르게 반환되어 윈도우가 즉시 그려진다.
        """
        super().__init__()
        self.cfg = config
        self.paths = paths
        # 동시에 두 번 돌면 안 되는 백그라운드 작업들의 핸들. None이면 해당 작업이 없다는 뜻.
        self._index_runner: ThreadRunner | None = None
        self._query_runner: ThreadRunner | None = None
        self._preload_runner: ThreadRunner | None = None
        self._index_mode = "build"  # 직전 색인이 "build"였는지 "refresh"(캡션만 갱신)였는지 기억
        self._refresh_after_build = False  # 증분 빌드 직후 캡션 갱신을 이어서 돌릴지 여부
        self._backends_ready = False  # 실제 임베딩 모델이 로딩 완료됐는지 — 검색/색인의 게이트
        self._build_t0 = 0.0  # 색인 시작 시각(소요 시간 기록용)
        self._last_hits: list = []  # 마지막 검색의 원본 결과(필터 전)
        self._display_hits: list = []  # 점수 임계값 필터를 거친 화면 표시용 결과
        self._graph_runner: ThreadRunner | None = None

        self.setWindowTitle("LLMImageFinder — sLLM image search")
        self.resize(1240, 820)

        # 벡터 스토어는 백엔드 선택과 무관하므로 한 번만 만든다(설정이 바뀌어도 재생성하지 않음).
        self.store = ChromaStore(self.paths.chroma_dir)
        # 객체 그래프(메모리 또는 kuzu) — 데이터셋 라벨로부터 백그라운드에서 지연 빌드된다.
        self.graph = build_graph_store(self.cfg, self.paths.graph_dir)
        # 색인 시점 썸네일 사전 워밍업을 색인 스레드가 아니라 이 별도 풀에서 돌린다.
        self._prewarm_pool = QThreadPool(self)
        self._prewarm_pool.setMaxThreadCount(2)  # JPEG 디코딩이 색인을 방해하지 않도록 소규모로 제한

        self._build_ui()
        self._setup_backends()
        self._post_init()

    # ---------------------------------------------------------------- 서비스 계층
    def _wire_services(self) -> None:
        """현재 백엔드(embedder/captioner/chat)를 기반으로 서비스 객체들을 (재)구성한다.

        백엔드가 placeholder mock -> 실제 모델로 교체된 직후에도 이 메서드를 다시 불러
        서비스들이 새 백엔드를 가리키게 한다. store/graph/cfg는 그대로 재사용한다.
        """
        self.service = SearchService(self.embedder, self.store, self.chat, self.cfg)
        self.indexer = Indexer(self.embedder, self.captioner, self.store, self.cfg)
        self.agent = AgenticSearch(self.service, self.graph, self.chat, self.cfg)

    def _setup_backends(self) -> None:
        """백엔드를 준비한다 — mock은 즉시, 실제 모델은 백그라운드로 로딩.

        mock 백엔드는 즉석에서 만들어지지만, 실제 모델은 로딩에 수 초~수십 초가 걸리므로
        UI가 멈추지 않도록 별도 스레드에서 로드한다(그 사이에는 placeholder mock으로 버틴다)."""
        if self.cfg.embedder_backend == "mock":
            self.embedder, self.captioner, self.chat = build_backends(self.cfg)
            self._wire_services()
            self._backends_ready = True
            self._update_model_chip()
            self._apply_banner()
            return
        # 즉석 placeholder로 UI를 살려 둔다 — 실제 모델이 준비될 때까지 검색은 게이트로 막힌다.
        self.embedder = MockEmbedder(dim=self.cfg.embed_dim)
        self.captioner = None
        self.chat = MockChatLLM()
        self._wire_services()
        self._backends_ready = False
        self._start_preload()

    def _start_preload(self) -> None:
        """실제 백엔드 가중치를 백그라운드 스레드에서 로딩 시작 — 그동안 색인 버튼은 비활성화."""
        self._update_model_chip()
        self.status_msg.setText("Loading embedding model…")
        self.act_index.setEnabled(False)
        self.act_rebuild.setEnabled(False)
        worker = PreloadWorker(self.cfg)
        worker.finished.connect(self._on_preload_done)
        worker.error.connect(self._on_preload_error)
        self._preload_runner = ThreadRunner(worker, self)
        self._preload_runner.start()

    def _on_preload_done(self, backends) -> None:
        """프리로드 성공 콜백(GUI 스레드) — placeholder를 실제 백엔드로 교체하고 게이트를 연다."""
        self.embedder, self.captioner, self.chat = backends
        self._wire_services()  # 서비스들이 새(실제) 백엔드를 가리키도록 재구성
        self._backends_ready = True
        self._preload_runner = None
        self.act_index.setEnabled(True)
        self.act_rebuild.setEnabled(True)
        self._update_model_chip()
        self._apply_banner()
        self._refresh_status()
        name = getattr(self.embedder, "name", "?")
        self.chat_widget.add_system(f"Model loaded ({name}) — ready to search.")

    def _on_preload_error(self, msg: str) -> None:
        """프리로드 실패 콜백(GUI 스레드) — placeholder mock에 머무르며 사용자에게 사유 안내."""
        # build_backends는 내부적으로 mock으로 폴백하므로 여기까지 오는 건 예외적인 상황이다.
        # 따라서 placeholder mock 백엔드를 그대로 쓰되, 왜 그런지 사용자에게 알려 준다.
        self._backends_ready = True
        self._preload_runner = None
        self.act_index.setEnabled(True)
        self.act_rebuild.setEnabled(True)
        self.cfg.mark_degraded(f"Backend load failed: {msg}")  # 배너에 표시할 degraded 사유 기록
        self._update_model_chip()
        self._apply_banner()
        self._refresh_status()

    def _apply_banner(self) -> None:
        """상단 mock/degraded 배너의 표시 여부와 사유를 현재 cfg 상태에 맞춰 갱신한다."""
        self.banner.setVisible(False)
        if self.cfg.degraded:
            self.banner.show_reason(self.cfg.degraded_reason)
        elif self.cfg.embedder_backend == "mock":
            # 의도적으로 mock을 쓰는 경우 — 사유 문자열은 비워 두고 mock 안내 배너만 띄운다.
            self.banner.show_reason("")

    def _update_model_chip(self) -> None:
        """상태바의 모델 칩(현재 임베딩 모델/디바이스 표시)을 로딩 상태에 맞춰 갱신한다."""
        if not self._backends_ready:
            self.model_chip.setText("Loading model…")
            self.model_chip.setStyleSheet("color:#f59e0b; padding:0 8px;")
            return
        # 백엔드마다 name/device 속성이 없을 수 있어 getattr로 안전하게 읽는다.
        name = getattr(self.embedder, "name", "?")
        device = getattr(self.embedder, "device", "")
        self.model_chip.setText(f"Embedding: {name}" + (f" ({device})" if device else ""))
        self.model_chip.setStyleSheet("color:#34d399; padding:0 8px;")

    # ---------------------------------------------------------------- UI 구성
    def _build_ui(self) -> None:
        """툴바·중앙 스플리터(채팅/갤러리)·상태바를 만들고 시그널을 연결한다."""
        tb = self.addToolBar("main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.act_index = tb.addAction(icons.index(), "Build / update index", self.start_index_update)
        self.act_index.setShortcut(QKeySequence("F5"))
        self.act_index.setToolTip("Index only changed files and prune deleted records (F5)")
        self.act_rebuild = tb.addAction(icons.rebuild(), "Full rebuild", self.start_full_rebuild)
        self.act_rebuild.setToolTip("Clear the whole index and re-index from scratch")
        tb.addSeparator()
        tb.addAction(icons.tags(), "Class names", self.edit_class_names)
        tb.addAction(icons.graph(), "Object graph", self.show_object_graph)
        tb.addAction(icons.info(), "Index info", self.show_index_info)
        tb.addSeparator()
        tb.addAction(icons.images(), "Generate sample dataset", self.make_sample)
        tb.addSeparator()
        tb.addAction(icons.settings(), "Settings", self.open_settings)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        # mock/degraded 안내 배너 — '설정 열기' 링크를 메인 윈도우의 설정 다이얼로그로 연결.
        self.banner = MockBanner()
        self.banner.open_settings.connect(self.open_settings)
        outer.addWidget(self.banner)

        # 좌:채팅 / 우:갤러리 를 가로로 나누는 스플리터.
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)

        # 채팅 위젯 — 사용자가 검색어를 제출하면 submitted 시그널로 run_query가 호출된다.
        self.chat_widget = ChatWidget(history_file=self.paths.history_file)
        self.chat_widget.submitted.connect(self.run_query)
        splitter.addWidget(self.chat_widget)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(4, 4, 4, 4)
        header = QHBoxLayout()
        title = QLabel("Search results")
        title.setStyleSheet("font-weight:600; color:#cbd5e1;")
        header.addWidget(title)
        header.addStretch(1)

        # 검색 방식 콤보 — 각 항목의 userData에 내부 식별자("hybrid" 등)를 함께 저장한다.
        self.mode_combo = QComboBox()
        for label, data in _MODE_LABELS:
            self.mode_combo.addItem(label, data)
        # 저장된 cfg.search_mode에 해당하는 항목을 선택. 못 찾으면 0번(하이브리드)으로 폴백.
        cur = next((i for i, (_, d) in enumerate(_MODE_LABELS) if d == self.cfg.search_mode), 0)
        self.mode_combo.setCurrentIndex(cur)
        self.mode_combo.setToolTip("Search mode — Hybrid (vector+keyword), Vector (semantic), Keyword (BM25)")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        header.addWidget(self.mode_combo)

        self.agent_check = QCheckBox("Agent")
        self.agent_check.setChecked(bool(self.cfg.agentic_enabled))
        self.agent_check.setToolTip(
            "Agentic search — plan → hybrid search → object-graph filter → summarize"
        )
        self.agent_check.toggled.connect(self._on_agentic_toggled)
        header.addWidget(self.agent_check)

        header.addWidget(QLabel("Results"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 200)
        self.k_spin.setValue(int(self.cfg.top_k))
        self.k_spin.setToolTip("Number of results to fetch on the next search (top-k)")
        self.k_spin.valueChanged.connect(self._on_k_changed)
        header.addWidget(self.k_spin)

        header.addWidget(QLabel("Score ≥"))
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0.0, 1.0)
        self.thr_spin.setSingleStep(0.05)
        self.thr_spin.setDecimals(2)
        self.thr_spin.setValue(float(self.cfg.score_threshold))
        self.thr_spin.setToolTip("Hide results below this similarity score (applied instantly, no re-search)")
        self.thr_spin.valueChanged.connect(self._on_threshold_changed)
        header.addWidget(self.thr_spin)

        # 내보내기 드롭다운(CSV 저장 / 경로 복사) — 결과가 있을 때만 활성화된다(_apply_display_filter).
        self.export_btn = QToolButton()
        self.export_btn.setIcon(icons.export())
        self.export_btn.setText(" Export")
        self.export_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.export_btn.setPopupMode(QToolButton.InstantPopup)  # 클릭 즉시 메뉴 펼침
        emenu = QMenu(self.export_btn)
        emenu.addAction("Save as CSV…", self.export_csv)
        emenu.addAction("Copy path list", self.copy_result_paths)
        self.export_btn.setMenu(emenu)
        self.export_btn.setEnabled(False)  # 검색 결과가 생기기 전까지 비활성
        header.addWidget(self.export_btn)

        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color:#94a3b8;")
        header.addWidget(self.count_label)
        self.open_folder_btn = QPushButton(icons.folder(), " Open folder")
        self.open_folder_btn.setToolTip("Reveal the selected image in Explorer (Ctrl+E)")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self.open_selected_folder)
        header.addWidget(self.open_folder_btn)
        rlay.addLayout(header)

        # 결과 갤러리 — 더블클릭/Enter는 뷰어 열기, 선택 변경은 폴더 버튼 토글, 우클릭은 유사 검색.
        self.gallery = ResultsGallery(thumb_size=self.cfg.thumb_size)
        self.gallery.activated_hit.connect(self.open_viewer)
        self.gallery.selected_hit.connect(self._on_select)
        self.gallery.similar_requested.connect(self.run_similar)
        rlay.addWidget(self.gallery, 1)
        splitter.addWidget(right)

        # 초기 가로 비율 — 갤러리 쪽을 더 넓게(채팅 3 : 갤러리 5), 픽셀 기준 초기 크기도 지정.
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([440, 800])

        # 상태바: 메시지 + 모델 칩 + 진행 막대 + 취소 버튼
        self.status_msg = QLabel("Ready")
        self.statusBar().addWidget(self.status_msg, 1)
        self.model_chip = QLabel("")
        self.statusBar().addPermanentWidget(self.model_chip)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(320)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.cancel_btn = QPushButton(icons.cancel(), " Cancel")
        self.cancel_btn.setVisible(False)  # 색인이 진행 중일 때만 노출
        self.statusBar().addPermanentWidget(self.cancel_btn)

        # Ctrl+L: 브라우저 주소창처럼 검색 입력창에 포커스를 주고 기존 텍스트를 선택.
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self._focus_search)

    def _focus_search(self) -> None:
        """검색 입력창에 포커스를 주고 전체 선택 — 바로 새 검색어를 덮어쓸 수 있게 한다."""
        self.chat_widget.input.setFocus()
        self.chat_widget.input.selectAll()

    def _post_init(self) -> None:
        """UI 구성 후 초기 안내/경고를 띄우고, 필요하면 객체 그래프를 미리 빌드한다."""
        self._refresh_status()
        if not self.cfg.dataset_root:
            self.chat_widget.add_system(
                "No dataset root is set. Choose a folder in the toolbar 'Settings', or "
                "use 'Generate sample dataset' to create example data, then run 'Build index'."
            )
        # 저장된 인덱스가 '다른' 데이터셋 경로에서 만들어졌다면 경고(불일치한 결과 방지).
        meta = load_index_meta(self.paths.index_meta_file)
        built_root = str(meta.get("dataset_root", ""))
        if (
            built_root
            and self.cfg.dataset_root
            and built_root != self.cfg.dataset_root
            and self.store.count() > 0
        ):
            self.chat_widget.add_system(
                "Note: the current index was built from a different dataset path. "
                "Check 'Index info' in the toolbar and run a full rebuild if needed."
            )
        # 에이전트 모드가 이미 켜져 있으면, 그것이 의존하는 객체 그래프를 미리 빌드해 둔다.
        if self.cfg.agentic_enabled and self.store.count() > 0:
            self._ensure_graph_async()

    def _unit_label(self) -> str:
        """색인 단위에 맞는 한국어 단위 명사("이미지" 또는 "폴더")를 돌려준다(메시지 표기용)."""
        return "image" if self.cfg.index_granularity == "image" else "folder"

    def _refresh_status(self) -> None:
        """현재 색인 개수를 상태바·결과 카운트 라벨·모델 칩에 반영한다."""
        n = self.store.count()
        self.count_label.setText(f"Indexed {self._unit_label()}s: {n}")
        self.status_msg.setText(f"Ready · {n} indexed")
        self._update_model_chip()

    # ---------------------------------------------------------------- 검색
    def _ready_for_query(self) -> bool:
        """검색을 실행해도 되는 상태인지 검사 — 불가하면 채팅에 사유를 남기고 False 반환.

        세 가지 게이트: (1) 색인 진행 중 아님, (2) 백엔드 로딩 완료, (3) 인덱스가 비어 있지 않음.
        """
        if self._index_runner is not None and self._index_runner.is_running():
            self.chat_widget.add_system("Indexing is in progress. Please search again when it finishes.")
            return False
        if not self._backends_ready:
            self.chat_widget.add_system("The embedding model is loading — please try again in a moment.")
            return False
        if self.store.count() == 0:
            self.chat_widget.add_system("The index is empty. Run 'Build / update index' first.")
            return False
        return True

    def run_query(self, text: str) -> None:
        """채팅 입력으로 들어온 자연어 검색어를 백그라운드 QueryWorker로 실행한다.

        에이전트 모드면 agent를 함께 넘겨 계획→검색→그래프 필터→요약 파이프라인을 태운다.
        """
        if self._query_runner is not None:
            self.chat_widget.add_system("A previous search is still running — please try again when it finishes.")
            return
        self.chat_widget.add_user(text)
        if not self._ready_for_query():
            return
        self.chat_widget.set_busy(True)  # 입력창/전송 버튼 잠금
        agentic = bool(self.cfg.agentic_enabled)
        self.status_msg.setText("Agentic search…" if agentic else "Searching…")

        agent = self.agent if agentic else None
        worker = QueryWorker(self.service, text, self.cfg.top_k, agent=agent)
        worker.finished.connect(self._on_query_done)
        worker.error.connect(self._on_query_error)
        # 큐 연결(DirectConnection 아님): 워커 스레드는 단계를 만들어 내느라 바쁘고,
        # 이 슬롯은 채팅 위젯을 건드리므로 반드시 GUI 스레드에서 실행되어야 한다.
        worker.trace.connect(self._on_agent_step)
        self._query_runner = ThreadRunner(worker, self)
        self._query_runner.start()

    def _on_agent_step(self, line: str) -> None:
        """에이전트가 보내온 진행 단계 문자열을 채팅에 시스템 메시지로 출력(GUI 스레드)."""
        self.chat_widget.add_system(line)

    def run_similar(self, hit) -> None:
        """예시 기반 검색(query-by-example): 이미 저장된 임베딩을 재사용한다.

        모델 호출이 없는 일반 경로라 충분히 빨라 동기 실행한다(검색 워커 스레드를 띄우지 않음).
        """
        if hit is None or self._query_runner is not None:
            return
        if not self._ready_for_query():
            return
        name = os.path.basename(hit.image_path or hit.folder)
        self.chat_widget.add_user(f"(find similar) {name}")
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)  # 동기 작업 동안 모래시계 커서
        try:
            result = self.service.query_by_example(hit, self.cfg.top_k)
        except Exception as e:
            self.chat_widget.add_system(f"Similar-search error: {type(e).__name__}: {e}")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()  # 예외가 나도 커서는 반드시 복구
        self._last_hits = list(result.hits)
        self._apply_display_filter()
        self.chat_widget.add_assistant(
            f"Found {len(result.hits)} results similar to '{name}'."
        )

    def _apply_display_filter(self) -> None:
        """원본 결과(_last_hits)에 점수 임계값 필터를 적용해 갤러리에 표시한다(재검색 없음).

        점수 임계값을 바꾸거나 새 결과가 들어올 때마다 호출되어 화면을 갱신한다.
        """
        thr = float(self.cfg.score_threshold)
        # 그래프 멤버십 결과(객체 그래프 AND 필터)는 코사인 점수로 랭킹된 게 아니므로,
        # 점수 임계값으로 숨기면 안 된다 — match=="graph"이면 임계값과 무관하게 유지한다.
        hits = [h for h in self._last_hits if h.score >= thr or h.match == "graph"]
        self._display_hits = hits
        self.gallery.set_results(hits)
        self.export_btn.setEnabled(bool(hits))  # 표시할 결과가 있을 때만 내보내기 허용
        # 필터로 일부가 가려졌으면 "표시 N / 검색 M"으로, 아니면 "검색 결과: N개"로 카운트 표기.
        if self._last_hits and len(hits) != len(self._last_hits):
            self.count_label.setText(f"Showing {len(hits)} / {len(self._last_hits)} found")
        else:
            self.count_label.setText(f"Search results: {len(hits)}")

    def _on_query_done(self, result) -> None:
        """검색 워커 완료 콜백 — 결과를 저장/표시하고 요약을 채팅에 출력, busy 해제."""
        self._last_hits = list(result.hits)
        self._apply_display_filter()
        if result.hits:
            self.chat_widget.add_assistant(
                result.summary or f"Found {len(result.hits)} related results."
            )
        else:
            self.chat_widget.add_system("No matching results. Try different wording.")
        self.chat_widget.set_busy(False)  # 입력창/전송 버튼 다시 활성화
        self.status_msg.setText("Ready")
        self._query_runner = None  # 다음 검색을 허용

    def _on_query_error(self, msg: str) -> None:
        """검색 워커 오류 콜백 — 채팅에 오류를 알리고 busy/runner 상태를 정리한다."""
        self.chat_widget.add_system(f"Search error: {msg}")
        self.chat_widget.set_busy(False)
        self.status_msg.setText("Error")
        self._query_runner = None

    def _on_k_changed(self, value: int) -> None:
        """top-k 스핀박스 변경 — 다음 검색부터 적용되도록 cfg에 저장(현재 결과는 그대로)."""
        self.cfg.top_k = int(value)
        self.cfg.save(self.paths.config_file)

    def _on_threshold_changed(self, value: float) -> None:
        """점수 임계값 변경 — cfg에 저장하고, 결과가 있으면 즉시 표시 필터를 다시 적용(재검색 없음)."""
        self.cfg.score_threshold = float(value)
        self.cfg.save(self.paths.config_file)
        if self._last_hits:
            self._apply_display_filter()

    def _on_mode_changed(self, _index: int) -> None:
        """검색 방식 콤보 변경 — userData(없으면 hybrid)를 저장하고 어휘 인덱스를 무효화."""
        self.cfg.search_mode = self.mode_combo.currentData() or "hybrid"
        self.cfg.save(self.paths.config_file)
        # 키워드/하이브리드 동작이 달라지므로 캐시된 BM25 어휘 인덱스를 폐기해 다음 검색 때 재구성.
        self.service.invalidate_lexical()

    def _on_agentic_toggled(self, on: bool) -> None:
        """에이전트 모드 체크박스 토글 — 켜졌고 인덱스가 있으면 그래프를 미리 빌드한다."""
        self.cfg.agentic_enabled = bool(on)
        self.cfg.save(self.paths.config_file)
        if on and self.store.count() > 0:
            self._ensure_graph_async()  # 에이전트의 그래프 필터가 객체 그래프를 필요로 함

    # ---------------------------------------------------------------- 객체 그래프
    def _ensure_graph_async(self, force: bool = False) -> None:
        """객체 그래프를 백그라운드에서 빌드한다.

        force=False: 그래프가 비어 있을 때만 빌드(지연 빌드).
        force=True: 색인/캡션이 바뀐 뒤 무조건 다시 빌드(라벨 변경 반영).
        """
        if self._graph_runner is not None and self._graph_runner.is_running():
            return  # 이미 빌드 중이면 중복 실행 방지
        root = self.cfg.dataset_root
        if not root or not os.path.isdir(root):
            return  # 유효한 데이터셋 경로가 없으면 그래프를 만들 수 없음
        if not force:
            try:
                if self.graph.count() > 0:
                    return  # 지연 빌드: 이미 채워져 있으면 다시 만들지 않는다
            except Exception:
                pass  # count 실패는 무시하고 그냥 빌드 진행
        worker = GraphBuildWorker(self.graph, root, self.cfg)
        worker.finished.connect(self._on_graph_built)
        worker.error.connect(lambda m: self.chat_widget.add_system(f"Object-graph build error: {m}"))
        self._graph_runner = ThreadRunner(worker, self)
        self._graph_runner.start()

    def _on_graph_built(self, n: int) -> None:
        """그래프 빌드 완료 콜백 — runner를 비우고, 라벨 이미지가 있으면 상태바에 개수 표시."""
        self._graph_runner = None
        if n > 0:
            self.status_msg.setText(f"Object graph ready — {n} labeled images")

    def show_object_graph(self) -> None:
        """객체 그래프 다이얼로그를 연다. 비어 있으면 (사용자 요청이므로) 동기로 즉시 빌드한다."""
        if self._graph_runner is not None and self._graph_runner.is_running():
            QMessageBox.information(self, "Building graph", "The object graph is being built. Please reopen it in a moment.")
            return
        try:
            empty = self.graph.count() == 0
        except Exception:
            empty = True  # count 실패 시 안전하게 '비어 있음'으로 간주해 빌드를 유도
        if empty:
            if not self.cfg.dataset_root or not os.path.isdir(self.cfg.dataset_root):
                QMessageBox.information(
                    self, "No graph",
                    "Available after indexing a dataset that has YOLO labels.",
                )
                return
            # 사용자가 명시적으로 그래프를 요청했으므로 여기서는 동기로 빌드한다(백그라운드 X).
            QGuiApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                # 지연 임포트: 그래프를 실제로 만들 때만 builder를 들여온다(시작 비용 절감).
                from imgsearch.graph.builder import build_records

                self.graph.build(build_records(self.cfg.dataset_root, self.cfg))
            finally:
                QGuiApplication.restoreOverrideCursor()
        dlg = GraphDialog(self.graph, self)
        dlg.show_images.connect(self._show_graph_images)
        dlg.exec()

    def _show_graph_images(self, paths: list) -> None:
        """그래프 다이얼로그에서 고른 객체 조합에 해당하는 이미지들을 결과 갤러리에 표시한다."""
        if not paths:
            self.chat_widget.add_system("No images contain all of the selected objects.")
            return
        # image_id로 조인한다(그래프는 원본 경로를 저장하지만 인덱스는 image_id를 키로 쓴다).
        hit_map = self.store.fetch([image_id(p) for p in paths])
        hits = list(hit_map.values())
        if not hits:
            self.chat_widget.add_system(
                "The graph doesn't match the current index — a full rebuild is recommended."
            )
            return
        for h in hits:
            h.match = "graph"  # 출처 배지를 'G'(그래프)로 표시 + 점수 필터에서 제외되도록 표시
        self._last_hits = hits
        self._apply_display_filter()
        if len(hits) < len(paths):
            self.chat_widget.add_system(
                f"Only {len(hits)} of {len(paths)} graph images are in the current index — a full rebuild is recommended."
            )
        self.chat_widget.add_assistant(f"Showing {len(hits)} results from the object-graph filter.")

    # ---------------------------------------------------------------- 내보내기
    def export_csv(self) -> None:
        """현재 표시 중인 결과(_display_hits)를 CSV로 저장한다(Excel 호환 UTF-8 BOM)."""
        hits = self._display_hits
        if not hits:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save search results as CSV", "search_results.csv", "CSV files (*.csv)"
        )
        if not path:
            return  # 사용자가 저장 대화상자를 취소함
        try:
            # utf-8-sig: BOM을 붙여 Excel에서 한글이 깨지지 않게 한다. newline=""는 csv 모듈 권장.
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["score", "caption", "image_path", "folder", "member_count"])
                for h in hits:
                    w.writerow([f"{h.score:.4f}", h.caption, h.image_path, h.folder, h.member_count])
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        self.chat_widget.add_system(f"CSV saved — {len(hits)} rows: {path}")

    def copy_result_paths(self) -> None:
        """표시 중인 결과의 이미지 경로들을 줄바꿈으로 이어 클립보드에 복사한다."""
        hits = self._display_hits
        if not hits:
            return
        QGuiApplication.clipboard().setText("\n".join(h.image_path for h in hits))
        self.chat_widget.add_system(f"Copied {len(hits)} image paths to the clipboard.")

    # ---------------------------------------------------------------- 색인(인덱싱)
    def start_index_update(self) -> None:
        """툴바 '인덱스 빌드/업데이트' — 변경분만 증분 색인(full_rebuild=False)."""
        self._start_index(full_rebuild=False)

    def start_full_rebuild(self) -> None:
        """툴바 '전체 재빌드' — 기존 인덱스를 모두 지우고 처음부터 다시 색인(확인 후)."""
        if self.store.count() > 0:
            ok = QMessageBox.question(
                self, "Full rebuild",
                "This clears the whole index and re-indexes from scratch. Continue?",
            )
            if ok != QMessageBox.Yes:
                return
        self._start_index(full_rebuild=True)

    def _scan_class_ids(self, root: str) -> dict[str, int]:
        """데이터셋의 YOLO 라벨을 훑어 {클래스 ID 문자열: 등장 횟수} 맵을 만든다(동기, 대기 커서)."""
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            counts = labels.scan_class_ids(root, self.cfg.image_exts)
        finally:
            QGuiApplication.restoreOverrideCursor()
        # 다이얼로그/설정에서 다루기 쉽도록 키는 str, 값은 int로 정규화한다.
        return {str(k): int(v) for k, v in counts.items()}

    def _prompt_class_names(self, class_counts: dict[str, int]) -> bool:
        """클래스 이름 편집기를 열어 config + class_names.yaml에 저장한다. 변경됐으면 True."""
        dlg = ClassNamesDialog(class_counts, dict(self.cfg.class_names), self)
        if dlg.exec() != ClassNamesDialog.Accepted:
            return False  # 사용자가 취소
        new_names = dlg.result_names()
        if new_names == self.cfg.class_names:
            return False  # 내용이 그대로면 저장도 변경 표시도 하지 않음
        self.cfg.class_names = new_names
        self.cfg.save(self.paths.config_file)
        # 다른 도구(예: 학습 파이프라인)와 공유할 수 있도록 표준 YAML로도 함께 기록.
        save_class_names_yaml(self.paths.class_names_file, new_names)
        self.chat_widget.add_system(
            f"Saved {len(new_names)} class names → {self.paths.class_names_file}"
        )
        return True

    def edit_class_names(self) -> None:
        """툴바 '클래스 이름' — 라벨을 스캔해 이름을 편집하고, 가능하면 캡션만 빠르게 갱신한다."""
        if self._index_runner is not None and self._index_runner.is_running():
            QMessageBox.information(self, "Indexing in progress", "Change it after indexing finishes.")
            return
        if self._query_runner is not None and self._query_runner.is_running():
            QMessageBox.information(self, "Search in progress", "Change it after the search finishes.")
            return
        root = self.cfg.dataset_root
        counts: dict[str, int] = {}
        if root and os.path.isdir(root):
            counts = self._scan_class_ids(root)
        if not counts and not self.cfg.class_names:
            QMessageBox.information(
                self, "No classes",
                "No YOLO label files found in the dataset. (Names are only used for labeled datasets.)",
            )
            return
        changed = self._prompt_class_names(counts)
        # 기존 인덱스에 빠르게 반영(재임베딩 없이 캡션만) — 단, 저장된 레코드가 정말
        # 이미지 단위일 때만. 폴더 단위 인덱스에는 라벨 캡션 갱신이 의미가 없다.
        sig = self.store.stored_signature()
        gran_ok = sig is None or sig[2] in (None, "image")  # sig[2]=색인 단위(미상이면 허용)
        if (
            changed
            and self.store.count() > 0
            and self.cfg.index_granularity == "image"
            and gran_ok
        ):
            ok = QMessageBox.question(
                self, "Refresh captions",
                "Apply the new names to the stored index captions?\n(Refreshes label captions only, with no re-embedding.)",
            )
            if ok == QMessageBox.Yes:
                self._start_index(full_rebuild=False, refresh=True)

    def _queue_prewarm(self, path: str) -> None:
        """색인 스레드의 thumb_cb로 호출되어 썸네일 워밍업을 별도 풀에 넣는다.

        QThreadPool.start는 스레드 안전하다. JPEG 디코딩을 여기로 옮겨 두면
        색인 스레드는 캡셔닝/임베딩에만 집중할 수 있다(디코딩 부하를 분리).
        """
        self._prewarm_pool.start(_PrewarmTask(path, self.cfg.thumb_size))

    def _start_index(self, full_rebuild: bool, refresh: bool = False) -> None:
        """색인을 백그라운드에서 시작하는 공통 진입점(게이트 검사 → 워커 기동 → UI 전환).

        full_rebuild=True면 전체 재빌드, refresh=True면 재임베딩 없이 캡션만 갱신.
        """
        if self._index_runner is not None:
            self.chat_widget.add_system("Indexing is already in progress — run again when it finishes.")
            return
        if self._query_runner is not None and self._query_runner.is_running():
            QMessageBox.information(self, "Search in progress", "Run it after the search finishes.")
            return
        if not self._backends_ready:
            self.chat_widget.add_system("The embedding model is loading — please try again when it finishes.")
            return
        root = self.cfg.dataset_root
        if not root or not os.path.isdir(root):
            QMessageBox.warning(self, "No dataset", "Set a valid dataset root first.")
            self.open_settings()
            return

        # 라벨이 있는 데이터셋을 처음 빌드할 때: 클래스 이름을 먼저 입력받을지 제안한다.
        # 아직 이름이 하나도 없을 때만 스캔한다 — 이후 빌드마다 UI 스레드에서 전체
        # 데이터셋을 다시 훑는 비용을 피하기 위함.
        names_changed = False
        if not refresh and self.cfg.index_granularity == "image" and not self.cfg.class_names:
            counts = self._scan_class_ids(root)
            if counts:
                ok = QMessageBox.question(
                    self, "Enter class names",
                    f"Found {len(counts)} classes in the YOLO labels.\n"
                    "Enter names to use in captions/summaries now?",
                )
                if ok == QMessageBox.Yes:
                    names_changed = self._prompt_class_names(counts)
        # 증분 빌드 직전에 이름이 바뀐 경우: 이미 색인되어 변경이 없는 레코드는 옛 캡션을
        # 그대로 유지하므로, 빌드가 끝나면 캡션 갱신(refresh)을 이어서 돌리도록 예약한다.
        self._refresh_after_build = (
            names_changed and not full_rebuild and self.store.count() > 0
        )
        self._index_mode = "refresh" if refresh else "build"
        self._build_t0 = time.time()  # 소요 시간 측정 시작

        # 색인 중에는 색인/재빌드 버튼을 잠그고 진행 막대와 취소 버튼을 노출한다.
        self.act_index.setEnabled(False)
        self.act_rebuild.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # 총량을 알기 전까지는 불확정(busy) 막대
        self.cancel_btn.setVisible(True)

        worker = IndexWorker(
            self.indexer, root, full_rebuild=full_rebuild,
            thumb_cb=self._queue_prewarm,  # 색인 스레드 → 썸네일 워밍업 풀로 위임하는 콜백
            mode="refresh" if refresh else "build",
        )
        worker.progress.connect(self._on_index_progress)
        worker.log.connect(self.status_msg.setText)
        worker.finished.connect(self._on_index_done)
        worker.error.connect(self._on_index_error)
        # DirectConnection: 워커 스레드는 run() 안에서 막혀 있어(이벤트 루프 없음) 큐 연결
        # 슬롯은 run()이 끝난 뒤에야 실행된다. 직접 호출하면 GUI 스레드에서 취소 플래그(bool)를
        # 바로 세팅한다 — GIL 단위로 원자적이며 워커가 폴링으로 확인한다.
        self.cancel_btn.clicked.connect(worker.cancel, Qt.DirectConnection)
        self._index_runner = ThreadRunner(worker, self)
        self._index_runner.start()
        self.chat_widget.add_system("Starting caption refresh…" if refresh else "Starting indexing…")

    def _on_index_progress(self, p) -> None:
        """색인 진행 시그널 처리 — 단계(scan/index/caption)에 맞춰 진행 막대와 상태 문구를 갱신."""
        if p.phase == "scan":
            self.progress.setRange(0, 0)  # 스캔 단계는 총량 미정 → 불확정 막대
            self.status_msg.setText(f"Scanning… {p.current} folders found")
        elif p.phase in ("index", "caption"):
            if p.total:
                self.progress.setRange(0, p.total)
                self.progress.setValue(p.current)
            verb = "Caption refresh" if p.phase == "caption" else "Indexing"
            self.status_msg.setText(f"{verb} {p.current}/{p.total} — {os.path.basename(p.folder)}")

    def _finish_index_ui(self) -> None:
        """색인 종료(완료/오류 공통) 시 UI를 원상복구 — 막대/취소 버튼 숨김, 버튼 재활성, runner 비움."""
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        try:
            self.cancel_btn.clicked.disconnect()  # 이번 실행에서 연결한 worker.cancel 배선 제거
        except (RuntimeError, TypeError):
            pass  # 연결된 게 없으면 disconnect가 예외를 던지므로 무시
        self.act_index.setEnabled(True)
        self.act_rebuild.setEnabled(True)
        self._index_runner = None

    def _write_index_meta(self, report) -> None:
        """이번 색인의 메타데이터(생성 시각/소요/경로/모델/건수 등)를 index_meta 파일에 기록한다."""
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
        """색인 완료 콜백 — refresh/build 두 모드를 분기 처리하고 후속 작업(그래프 재빌드 등)을 잇는다."""
        was_refresh = self._index_mode == "refresh"
        self._finish_index_ui()
        self._refresh_status()
        # 문서/라벨이 바뀌었으므로 BM25 어휘 인덱스와 객체 그래프는 이제 낡았다 → 무효화.
        self.service.invalidate_lexical()
        if was_refresh:
            # refresh 모드: 워커가 갱신한 레코드 수(int)를 돌려준다.
            n = int(result)
            if n > 0:
                self.chat_widget.add_system(f"Caption refresh complete — new names applied to {n} records.")
            else:
                self.chat_widget.add_system(
                    "No records were refreshed — check the index granularity matches, or run a full rebuild."
                )
            self._ensure_graph_async(force=True)  # 이름이 바뀌었으니 그래프 재빌드
            return

        report = result  # build 모드: 워커가 BuildReport를 돌려준다
        if report.cancelled:
            msg = f"Indexing was cancelled — {self.store.count()} records are currently stored."
        else:
            msg = f"Indexing complete — {self.store.count()} {self._unit_label()}s indexed in total."
        if report.pruned:
            msg += f" (pruned {report.pruned} deleted-file records)"
        self.chat_widget.add_system(msg)
        # 취소된 빌드는 메타를 신뢰할 수 없으므로 기록/그래프 재빌드를 건너뛴다.
        if not report.cancelled:
            self._write_index_meta(report)
            self._ensure_graph_async(force=True)
        if report.skipped:
            self.chat_widget.add_system(f"{len(report.skipped)} items were skipped due to errors (details shown).")
            box = QMessageBox(
                QMessageBox.Warning, "Some items skipped",
                f"{len(report.skipped)} items were not indexed due to errors.\n"
                "See the full list under Details.",
                QMessageBox.Ok, self,
            )
            # 목록이 너무 길어 대화상자가 비대해지지 않도록 앞 500개로 잘라서 보여 준다.
            box.setDetailedText("\n".join(f"{p} — {e}" for p, e in report.skipped[:500]))
            box.exec()
        # _start_index에서 예약해 둔 후속 캡션 갱신 — 한 번만 돌도록 플래그를 먼저 끈다.
        if self._refresh_after_build:
            self._refresh_after_build = False
            self.chat_widget.add_system("Applying the new class names to existing record captions…")
            self._start_index(full_rebuild=False, refresh=True)

    def _on_index_error(self, msg: str) -> None:
        """색인 오류 콜백 — UI를 복구하고 치명 오류 대화상자를 띄운다."""
        self._finish_index_ui()
        QMessageBox.critical(self, "Indexing error", msg)
        self.status_msg.setText("Error")

    # ---------------------------------------------------------------- 뷰어 / 폴더
    def open_viewer(self, hit) -> None:
        """결과를 더블클릭/Enter했을 때 이미지 뷰어를 연다.

        이미지 단위: 결과 전체를 랭킹 순서로 넘기며 보고, 각 이미지에 순위/점수/캡션을 표시한다.
        폴더 단위: 해당 폴더의 멤버 이미지들 사이를 넘기는 기존 동작을 유지한다.
        """
        # 이미지 단위: 결과를 랭킹 순서로 탐색하며 점수+캡션을 표시.
        if self.cfg.index_granularity == "image" and self._display_hits:
            paths = [h.image_path for h in self._display_hits]
            try:
                start = paths.index(hit.image_path)  # 더블클릭한 이미지를 시작 위치로
            except ValueError:
                # 표시 목록에 없으면(드물게) 그 한 장만 보여 준다.
                paths, start = [hit.image_path], 0
            by_path = {h.image_path: h for h in self._display_hits}  # 경로→hit 역참조

            def meta(p: str) -> str:
                """뷰어가 현재 보고 있는 경로 p에 대해 표시할 메타 문구(순위/점수/캡션)를 만든다."""
                h = by_path.get(p)
                if h is None:
                    return ""
                rank = paths.index(p) + 1 if p in by_path else 0
                return f"Result {rank}/{len(paths)} · score {h.score:.2f} · {h.caption}" if h.caption \
                    else f"Result {rank}/{len(paths)} · score {h.score:.2f}"

            ImageViewer(
                paths, start, self, class_names=self.cfg.class_names, meta_provider=meta
            ).exec()
            return
        # 폴더 단위: 해당 폴더의 멤버 이미지를 가져온다(없으면 클릭한 이미지 한 장).
        members = self.service.member_images(hit.folder) or [hit.image_path]
        try:
            start = members.index(hit.image_path)
        except ValueError:
            start = 0
        ImageViewer(members, start, self, class_names=self.cfg.class_names).exec()

    def _on_select(self, hit) -> None:
        """갤러리 선택 변경 — 선택된 hit에 이미지 경로가 있을 때만 '폴더 열기' 버튼 활성화."""
        self.open_folder_btn.setEnabled(hit is not None and bool(hit.image_path))

    def open_selected_folder(self) -> None:
        """선택한 결과의 이미지를 탐색기에서(해당 위치를 강조해) 연다."""
        hit = self.gallery.current_hit()
        if hit and hit.image_path:
            reveal_in_explorer(hit.image_path)

    # ---------------------------------------------------------------- 정보 / 설정 / 샘플
    def show_index_info(self) -> None:
        """툴바 '인덱스 정보' — 현재 인덱스 통계/메타를 보여 주고, 재빌드 버튼을 누르면 전체 재빌드."""
        meta = load_index_meta(self.paths.index_meta_file)
        dlg = IndexInfoDialog(
            self.store.count(), self.store.stored_signature(), meta,
            self.cfg.dataset_root, self.paths.chroma_dir, self,
        )
        dlg.exec()
        if dlg.rebuild_clicked:
            self.start_full_rebuild()

    def open_settings(self) -> None:
        """설정 다이얼로그를 연다. 저장된 변경의 '서명'을 비교해 꼭 필요한 재구성만 수행한다.

        백엔드 서명이 바뀌면 백엔드 재로딩, 그래프 백엔드가 바뀌면 그래프 재생성,
        색인 서명이 바뀌면 전체 재빌드를 제안한다. 그 외(가벼운 설정)는 살아 있는 객체에
        새 cfg만 다시 연결한다 — 불필요한 모델 재로딩을 피하기 위한 핵심 최적화.
        """
        # 진행 중인 백그라운드 작업이 있으면 설정 변경을 막는다(객체가 교체되면 작업이 깨질 수 있음).
        if self._index_runner is not None and self._index_runner.is_running():
            QMessageBox.information(self, "Indexing in progress", "Change settings after indexing finishes.")
            return
        if self._query_runner is not None and self._query_runner.is_running():
            QMessageBox.information(self, "Search in progress", "Change settings after the search finishes.")
            return
        if self._graph_runner is not None and self._graph_runner.is_running():
            QMessageBox.information(self, "Building graph", "Change settings after the object-graph build finishes.")
            return
        # 변경 전 서명을 찍어 두고, 다이얼로그 종료 후 무엇이 바뀌었는지 비교한다.
        before_index = _index_signature(self.cfg)
        before_backend = _backend_signature(self.cfg)
        before_graph = self.cfg.graph_backend
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() != SettingsDialog.Accepted:
            return  # 취소 시 아무것도 적용하지 않음
        self.cfg = dlg.result_config()
        self.cfg.save(self.paths.config_file)
        if _backend_signature(self.cfg) != before_backend:
            self._setup_backends()  # 실제 가중치를 워커 스레드에서 다시 로드할 수 있음
        else:
            # 가벼운 설정만 바뀐 경우: 살아 있는 객체들이 새 cfg를 가리키도록만 갱신(재로딩 X).
            self.service.cfg = self.cfg
            self.indexer.cfg = self.cfg
            self.agent.cfg = self.cfg
            self._update_model_chip()
            self._apply_banner()
        if self.cfg.graph_backend != before_graph:
            # 그래프 백엔드가 바뀌면 기존 그래프를 닫고(특히 kuzu 디렉터리 락 해제) 새로 만든다.
            try:
                self.graph.close()
            except Exception:
                pass
            self.graph = build_graph_store(self.cfg, self.paths.graph_dir)
            self.agent.graph = self.graph  # 에이전트도 새 그래프를 가리키게 한다
            if self.store.count() > 0:
                self._ensure_graph_async(force=True)
        self.gallery.set_thumb_size(self.cfg.thumb_size)
        self.k_spin.setValue(int(self.cfg.top_k))  # 설정에서 바꾼 top-k를 스핀박스에도 반영
        self._refresh_status()
        # 색인 서명이 바뀌었고 기존 인덱스가 있으면 호환되지 않으므로 전체 재빌드를 제안.
        if _index_signature(self.cfg) != before_index and self.store.count() > 0:
            ok = QMessageBox.question(
                self, "Index settings changed",
                "The embedding model/dimension or index granularity changed and is incompatible "
                "with the existing index.\nRun a full rebuild now?",
            )
            if ok == QMessageBox.Yes:
                self._start_index(full_rebuild=True)

    def make_sample(self) -> None:
        """툴바 '샘플 데이터셋 생성' — 예시 데이터를 만들어 데이터셋 루트로 지정하고 빌드를 제안."""
        if self._index_runner is not None and self._index_runner.is_running():
            QMessageBox.information(self, "Indexing in progress", "Run it after indexing finishes.")
            return
        target = QFileDialog.getExistingDirectory(self, "Choose a folder to generate the sample dataset in")
        if not target:
            return  # 폴더 선택 취소
        # 지연 임포트: 샘플 생성 경로에서만 필요한 무거운 의존성을 시작 시점에 끌어오지 않는다.
        from imgsearch.sample_data import generate_sample_dataset

        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ds = generate_sample_dataset(os.path.join(target, "sample_dataset"))
        finally:
            QGuiApplication.restoreOverrideCursor()
        self.cfg.dataset_root = str(ds)  # 생성된 샘플을 곧바로 데이터셋 루트로 설정
        self.cfg.save(self.paths.config_file)
        self._refresh_status()
        self.chat_widget.add_system(f"Generated sample dataset: {ds}")
        if QMessageBox.question(self, "Build index", "Build the index now?") == QMessageBox.Yes:
            self._start_index(full_rebuild=True)

    # ---------------------------------------------------------------- 수명 주기
    def closeEvent(self, event) -> None:  # noqa: N802
        """윈도우 종료 시 백그라운드 작업을 정리한다 — 워커 종료 대기, 풀 비우기, 그래프 락 해제.

        깔끔한 종료를 위해 각 워커를 (취소 후) 일정 시간 대기한다. 어떤 작업은 중단할 수
        없어(예: torch 가중치 로드) 끝날 때까지 기다리거나, 막힌 경우 스레드를 떼어 내(leak)
        앱이 영영 안 닫히는 상황을 막는다.
        """
        if self._index_runner is not None and self._index_runner.is_running():
            self._index_runner.worker.cancel()  # 색인은 취소 플래그로 안전하게 멈출 수 있음
            self._index_runner.wait(8000)
        if self._query_runner is not None and self._query_runner.is_running():
            self._query_runner.wait(8000)
        if self._graph_runner is not None and self._graph_runner.is_running():
            self._graph_runner.wait(8000)
        if self._preload_runner is not None and self._preload_runner.is_running():
            # torch 가중치 로드는 중단할 수 없으므로 끝날 때까지 기다린다(드물게 발생).
            QGuiApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._preload_runner.wait(60000)
            finally:
                QGuiApplication.restoreOverrideCursor()
        self._prewarm_pool.clear()  # 아직 시작 안 한 워밍업 작업을 큐에서 비움
        if not self._prewarm_pool.waitForDone(2000):
            self._prewarm_pool.setParent(None)  # 막힌 디코딩이 종료를 막지 못하도록 풀을 떼어 냄
        self.gallery.shutdown()  # 갤러리의 썸네일 작업을 모두 비우고 정리
        try:
            self.graph.close()  # 다음 세션을 위해 kuzu 디렉터리 락을 해제
        except Exception:
            pass
        super().closeEvent(event)
