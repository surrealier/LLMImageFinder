"""설정 다이얼로그: 데이터셋 루트, 백엔드 선택, 모델 ID/엔드포인트, 검색 관련 옵션을 편집한다.

각 입력 위젯은 AppConfig의 필드와 1:1로 대응한다. 다이얼로그는 원본 config를
깊은 복사해 들고 있다가, 사용자가 '저장'을 누르면 result_config()에서 위젯 값을
읽어 채운 사본을 돌려준다(취소 시 원본은 그대로 유지됨).
"""

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
    """앱 설정을 그룹별로 편집하는 모달 다이얼로그."""

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.setMinimumWidth(560)
        # 원본 config를 깊은 복사해 작업한다. '취소'를 눌러도 호출자의 설정이
        # 더럽혀지지 않도록 보장하기 위한 것이다.
        self._cfg = copy.deepcopy(config)

        root = QVBoxLayout(self)

        # --- 데이터셋 그룹 ---
        ds_box = QGroupBox("데이터셋")
        ds_form = QFormLayout(ds_box)
        # 데이터셋 루트 입력 + '찾아보기' 버튼을 한 줄에 나란히 둔다.
        row = QHBoxLayout()
        self.dataset_root = QLineEdit(self._cfg.dataset_root)
        browse = QPushButton("찾아보기…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.dataset_root, 1)
        row.addWidget(browse)
        ds_form.addRow("데이터셋 루트", row)
        # 색인 단위 선택. 콤보 항목의 '데이터'에 config에 저장될 키("folder"/"image")를 담아둔다.
        self.granularity = QComboBox()
        self.granularity.addItem("폴더당 대표 1장 (유사 장면 폴더)", "folder")
        self.granularity.addItem("이미지별 개별 색인 (다양한 사진)", "image")
        # 저장된 값이 "image"면 두 번째 항목(인덱스 1), 아니면 첫 번째 항목을 선택한다.
        self.granularity.setCurrentIndex(1 if self._cfg.index_granularity == "image" else 0)
        self.granularity.setToolTip(
            "변경하면 기존 인덱스와 호환되지 않아 전체 재빌드가 필요합니다."
        )
        ds_form.addRow("색인 단위", self.granularity)
        root.addWidget(ds_box)

        # --- 임베더 그룹 (검색 품질을 좌우하는 핵심 설정) ---
        emb_box = QGroupBox("임베딩 (검색 핵심)")
        emb_form = QFormLayout(emb_box)
        # 백엔드 선택: mock(결정적 더미)과 실제 jina-clip 중 하나.
        self.embedder_backend = QComboBox()
        self.embedder_backend.addItems(["mock", "jina-clip"])
        self.embedder_backend.setCurrentText(self._cfg.embedder_backend)
        self.embedder_model = QLineEdit(self._cfg.embedder_model)
        # 디바이스: auto(자동 선택)/cpu/cuda. auto면 런타임에 GPU 가용성에 따라 결정된다.
        self.embedder_device = QComboBox()
        self.embedder_device.addItems(["auto", "cpu", "cuda"])
        self.embedder_device.setCurrentText(self._cfg.embedder_device)
        self.embed_dim = QSpinBox()
        # jina-clip-v2의 Matryoshka(가변 차원) 지원 범위는 64~1024다. 이를 넘는 값을 주면
        # 다음 시작 시 모델 로드가 실패하고, 앱이 조용히 mock 모드로 떨어진다. 따라서
        # 입력 단계에서 범위를 강제해 그런 상황을 애초에 막는다.
        self.embed_dim.setRange(64, 1024)
        # 64 단위로만 조절(Matryoshka 차원이 64의 배수 단위로 의미가 있기 때문).
        self.embed_dim.setSingleStep(64)
        # 저장된 값이 1024를 넘더라도 1024로 잘라(min) 범위 안에 들어오게 한다.
        self.embed_dim.setValue(min(int(self._cfg.embed_dim), 1024))
        self.embed_dim.setToolTip(
            "jina-clip-v2는 64~1024 차원을 지원합니다. 변경 시 전체 재빌드가 필요합니다."
        )
        emb_form.addRow("백엔드", self.embedder_backend)
        emb_form.addRow("모델 ID", self.embedder_model)
        emb_form.addRow("디바이스", self.embedder_device)
        emb_form.addRow("임베딩 차원", self.embed_dim)
        root.addWidget(emb_box)

        # --- 캡셔너(VLM) 그룹: 인덱싱 시 이미지 캡션을 생성하는 비전-언어 모델 설정 ---
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
        # API 키는 비밀번호처럼 가려서 표시한다(어깨너머 노출 방지).
        self.vlm_api_key.setEchoMode(QLineEdit.Password)
        vlm_form.addRow(self.caption_enabled)
        vlm_form.addRow("백엔드", self.vlm_backend)
        vlm_form.addRow("Base URL", self.vlm_base_url)
        vlm_form.addRow("모델", self.vlm_model)
        vlm_form.addRow("API Key", self.vlm_api_key)
        root.addWidget(vlm_box)

        # --- 채팅 sLLM 그룹: 검색 결과를 자연어로 요약하는 소형 LLM 설정 ---
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
        # VLM과 마찬가지로 채팅 API 키도 가려서 표시한다.
        self.chat_api_key.setEchoMode(QLineEdit.Password)
        chat_form.addRow(self.summarize_enabled)
        chat_form.addRow("백엔드", self.chat_backend)
        chat_form.addRow("Base URL", self.chat_base_url)
        chat_form.addRow("모델", self.chat_model)
        chat_form.addRow("API Key", self.chat_api_key)
        root.addWidget(chat_box)

        # --- 검색/표시 그룹: 결과 개수와 썸네일 크기 등 화면 관련 옵션 ---
        ret_box = QGroupBox("검색/표시")
        ret_form = QFormLayout(ret_box)
        # 검색 시 가져올 상위 결과 개수(top-k).
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 200)
        self.top_k.setValue(self._cfg.top_k)
        # 갤러리 썸네일 한 변의 픽셀 크기. 32px 단위로 조절한다.
        self.thumb_size = QSpinBox()
        self.thumb_size.setRange(96, 512)
        self.thumb_size.setSingleStep(32)
        self.thumb_size.setValue(self._cfg.thumb_size)
        ret_form.addRow("결과 개수 (top-k)", self.top_k)
        ret_form.addRow("썸네일 크기(px)", self.thumb_size)
        root.addWidget(ret_box)

        # --- 객체 그래프 그룹: 객체 동시 출현을 저장/조회할 GraphDB 백엔드 선택 ---
        graph_box = QGroupBox("객체 그래프 (GraphDB)")
        graph_form = QFormLayout(graph_box)
        # 콤보 항목 '데이터'에 config 키("memory"/"kuzu")를 담아 둔다(표시 문구와 분리).
        self.graph_backend = QComboBox()
        self.graph_backend.addItem("메모리 (내장, 의존성 없음)", "memory")
        self.graph_backend.addItem("kuzu (임베디드 GraphDB, [graph] 설치 필요)", "kuzu")
        self.graph_backend.setCurrentIndex(1 if self._cfg.graph_backend == "kuzu" else 0)
        self.graph_backend.setToolTip(
            "kuzu는 'uv sync --extra graph' 설치가 필요하며, 없으면 메모리 그래프로 자동 대체됩니다."
        )
        graph_form.addRow("그래프 백엔드", self.graph_backend)
        root.addWidget(graph_box)

        # 저장/취소 버튼. 저장은 accept(), 취소는 reject()로 다이얼로그를 닫는다.
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _browse(self) -> None:
        """'찾아보기' 버튼: 폴더 선택 대화상자를 열어 데이터셋 루트를 채운다."""
        # 현재 입력된 경로를 시작 위치로 사용한다(없으면 빈 문자열).
        start = self.dataset_root.text() or ""
        chosen = QFileDialog.getExistingDirectory(self, "데이터셋 폴더 선택", start)
        # 사용자가 취소하면 빈 문자열이 오므로 그때는 기존 값을 그대로 둔다.
        if chosen:
            self.dataset_root.setText(chosen)

    def result_config(self) -> AppConfig:
        """현재 위젯 값들을 읽어 채운 AppConfig 사본을 돌려준다(저장 시 호출).

        문자열 입력은 strip()으로 앞뒤 공백을 제거하고, API 키가 비어 있으면
        OpenAI 호환 클라이언트가 요구하는 더미 값 "EMPTY"로 대체한다.
        """
        c = self._cfg
        c.dataset_root = self.dataset_root.text().strip()
        # 콤보의 표시 문구가 아니라 항목에 심어둔 데이터("folder"/"image")를 저장한다.
        c.index_granularity = self.granularity.currentData()
        c.embedder_backend = self.embedder_backend.currentText()
        c.embedder_model = self.embedder_model.text().strip()
        c.embedder_device = self.embedder_device.currentText()
        c.embed_dim = int(self.embed_dim.value())
        c.caption_enabled = self.caption_enabled.isChecked()
        c.vlm_backend = self.vlm_backend.currentText()
        c.vlm_base_url = self.vlm_base_url.text().strip()
        c.vlm_model = self.vlm_model.text().strip()
        # 키가 비면 "EMPTY"로 대체 — OpenAI 호환 클라이언트는 빈 키를 허용하지 않기 때문.
        c.vlm_api_key = self.vlm_api_key.text().strip() or "EMPTY"
        c.summarize_enabled = self.summarize_enabled.isChecked()
        c.chat_backend = self.chat_backend.currentText()
        c.chat_base_url = self.chat_base_url.text().strip()
        c.chat_model = self.chat_model.text().strip()
        # 채팅 키도 동일하게 빈 값이면 "EMPTY"로 대체한다.
        c.chat_api_key = self.chat_api_key.text().strip() or "EMPTY"
        c.top_k = int(self.top_k.value())
        c.thumb_size = int(self.thumb_size.value())
        # 그래프 백엔드도 항목 데이터("memory"/"kuzu")를 저장한다.
        c.graph_backend = self.graph_backend.currentData()
        return c
