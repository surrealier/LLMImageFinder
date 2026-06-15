"""Settings: dataset root, backend selection, model ids/endpoints, retrieval knobs."""

from __future__ import annotations

import copy

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from imgsearch.config import AppConfig


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.setMinimumWidth(560)
        self._cfg = copy.deepcopy(config)

        root = QVBoxLayout(self)

        # --- dataset ---
        ds_box = QGroupBox("데이터셋")
        ds_form = QFormLayout(ds_box)
        row = QHBoxLayout()
        self.dataset_root = QLineEdit(self._cfg.dataset_root)
        browse = QPushButton("찾아보기…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.dataset_root, 1)
        row.addWidget(browse)
        ds_form.addRow("데이터셋 루트", row)
        self.granularity = QComboBox()
        self.granularity.addItem("폴더당 대표 1장 (유사 장면 폴더)", "folder")
        self.granularity.addItem("이미지별 개별 색인 (다양한 사진)", "image")
        self.granularity.setCurrentIndex(1 if self._cfg.index_granularity == "image" else 0)
        self.granularity.setToolTip(
            "변경하면 기존 인덱스와 호환되지 않아 전체 재빌드가 필요합니다."
        )
        ds_form.addRow("색인 단위", self.granularity)
        root.addWidget(ds_box)

        # --- embedder ---
        emb_box = QGroupBox("임베딩 (검색 핵심)")
        emb_form = QFormLayout(emb_box)
        self.embedder_backend = QComboBox()
        self.embedder_backend.addItems(["mock", "jina-clip"])
        self.embedder_backend.setCurrentText(self._cfg.embedder_backend)
        self.embedder_model = QLineEdit(self._cfg.embedder_model)
        self.embedder_device = QComboBox()
        self.embedder_device.addItems(["auto", "cpu", "cuda"])
        self.embedder_device.setCurrentText(self._cfg.embedder_device)
        self.embed_dim = QSpinBox()
        # jina-clip-v2's Matryoshka range is 64..1024 — larger values would fail
        # the model load at next startup and silently drop the app to mock mode
        self.embed_dim.setRange(64, 1024)
        self.embed_dim.setSingleStep(64)
        self.embed_dim.setValue(min(int(self._cfg.embed_dim), 1024))
        self.embed_dim.setToolTip(
            "jina-clip-v2는 64~1024 차원을 지원합니다. 변경 시 전체 재빌드가 필요합니다."
        )
        emb_form.addRow("백엔드", self.embedder_backend)
        emb_form.addRow("모델 ID", self.embedder_model)
        emb_form.addRow("디바이스", self.embedder_device)
        emb_form.addRow("임베딩 차원", self.embed_dim)
        root.addWidget(emb_box)

        # --- captioner (VLM) ---
        vlm_box = QGroupBox("캡션 (VLM)")
        vlm_form = QFormLayout(vlm_box)
        self.caption_enabled = QCheckBox("인덱싱 시 대표 이미지 캡션 생성")
        self.caption_enabled.setChecked(self._cfg.caption_enabled)
        self.vlm_backend = QComboBox()
        self.vlm_backend.addItems(["mock", "vllm"])
        self.vlm_backend.setCurrentText(self._cfg.vlm_backend)
        self.vlm_base_url = QLineEdit(self._cfg.vlm_base_url)
        self.vlm_base_url.setToolTip("vLLM 등 OpenAI 호환 서버 주소 (예: http://localhost:8000/v1)")
        self.vlm_model = QLineEdit(self._cfg.vlm_model)
        self.vlm_api_key = QLineEdit(self._cfg.vlm_api_key)
        self.vlm_api_key.setEchoMode(QLineEdit.Password)
        vlm_form.addRow(self.caption_enabled)
        vlm_form.addRow("백엔드", self.vlm_backend)
        vlm_form.addRow("Base URL", self.vlm_base_url)
        vlm_form.addRow("모델", self.vlm_model)
        vlm_form.addRow("API Key", self.vlm_api_key)
        root.addWidget(vlm_box)

        # --- chat sLLM ---
        chat_box = QGroupBox("채팅 sLLM")
        chat_form = QFormLayout(chat_box)
        self.summarize_enabled = QCheckBox("검색 결과 요약 생성")
        self.summarize_enabled.setChecked(self._cfg.summarize_enabled)
        self.chat_backend = QComboBox()
        self.chat_backend.addItems(["mock", "vllm"])
        self.chat_backend.setCurrentText(self._cfg.chat_backend)
        self.chat_base_url = QLineEdit(self._cfg.chat_base_url)
        self.chat_base_url.setToolTip("vLLM 등 OpenAI 호환 서버 주소 (예: http://localhost:8000/v1)")
        self.chat_model = QLineEdit(self._cfg.chat_model)
        self.chat_api_key = QLineEdit(self._cfg.chat_api_key)
        self.chat_api_key.setEchoMode(QLineEdit.Password)
        chat_form.addRow(self.summarize_enabled)
        chat_form.addRow("백엔드", self.chat_backend)
        chat_form.addRow("Base URL", self.chat_base_url)
        chat_form.addRow("모델", self.chat_model)
        chat_form.addRow("API Key", self.chat_api_key)
        root.addWidget(chat_box)

        # --- retrieval ---
        ret_box = QGroupBox("검색/표시")
        ret_form = QFormLayout(ret_box)
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 200)
        self.top_k.setValue(self._cfg.top_k)
        self.thumb_size = QSpinBox()
        self.thumb_size.setRange(96, 512)
        self.thumb_size.setSingleStep(32)
        self.thumb_size.setValue(self._cfg.thumb_size)
        ret_form.addRow("결과 개수 (top-k)", self.top_k)
        ret_form.addRow("썸네일 크기(px)", self.thumb_size)
        root.addWidget(ret_box)

        # --- object graph ---
        graph_box = QGroupBox("객체 그래프 (GraphDB)")
        graph_form = QFormLayout(graph_box)
        self.graph_backend = QComboBox()
        self.graph_backend.addItem("메모리 (내장, 의존성 없음)", "memory")
        self.graph_backend.addItem("kuzu (임베디드 GraphDB, [graph] 설치 필요)", "kuzu")
        self.graph_backend.setCurrentIndex(1 if self._cfg.graph_backend == "kuzu" else 0)
        self.graph_backend.setToolTip(
            "kuzu는 'uv sync --extra graph' 설치가 필요하며, 없으면 메모리 그래프로 자동 대체됩니다."
        )
        graph_form.addRow("그래프 백엔드", self.graph_backend)
        root.addWidget(graph_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _browse(self) -> None:
        start = self.dataset_root.text() or ""
        chosen = QFileDialog.getExistingDirectory(self, "데이터셋 폴더 선택", start)
        if chosen:
            self.dataset_root.setText(chosen)

    def result_config(self) -> AppConfig:
        c = self._cfg
        c.dataset_root = self.dataset_root.text().strip()
        c.index_granularity = self.granularity.currentData()
        c.embedder_backend = self.embedder_backend.currentText()
        c.embedder_model = self.embedder_model.text().strip()
        c.embedder_device = self.embedder_device.currentText()
        c.embed_dim = int(self.embed_dim.value())
        c.caption_enabled = self.caption_enabled.isChecked()
        c.vlm_backend = self.vlm_backend.currentText()
        c.vlm_base_url = self.vlm_base_url.text().strip()
        c.vlm_model = self.vlm_model.text().strip()
        c.vlm_api_key = self.vlm_api_key.text().strip() or "EMPTY"
        c.summarize_enabled = self.summarize_enabled.isChecked()
        c.chat_backend = self.chat_backend.currentText()
        c.chat_base_url = self.chat_base_url.text().strip()
        c.chat_model = self.chat_model.text().strip()
        c.chat_api_key = self.chat_api_key.text().strip() or "EMPTY"
        c.top_k = int(self.top_k.value())
        c.thumb_size = int(self.thumb_size.value())
        c.graph_backend = self.graph_backend.currentData()
        return c
