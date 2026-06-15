"""애플리케이션 설정: JSON으로 영속화되는 dataclass.

백엔드는 문자열(``"mock"`` / ``"jina-clip"`` / ``"vllm"``)로 선택하므로, 앱의 나머지
코드는 모델 라이브러리를 직접 import하지 않는다(느슨한 결합). ``degraded`` /
``degraded_reason``은 런타임 전용 필드로, 요청한 실제 백엔드를 쓸 수 없어 mock으로
폴백했을 때만 설정되며 디스크에는 저장되지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# 데이터셋 스캔 시 이미지로 인정할 확장자 목록.
DEFAULT_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"]
# 이미지 옆에 함께 놓인 캡션/메타(사이드카) 텍스트 파일로 인정할 확장자.
SIDECAR_TEXT_EXTS = [".txt", ".caption", ".json"]


@dataclass
class AppConfig:
    """앱 전역 설정 묶음. 각 필드는 config.json에 그대로 직렬화/역직렬화된다."""

    dataset_root: str = ""

    # --- embedder (retrieval core: image + text -> shared space) ---
    embedder_backend: str = "mock"  # "mock" | "jina-clip"
    embedder_model: str = "jinaai/jina-clip-v2"
    embedder_device: str = "auto"  # "auto" | "cpu" | "cuda"
    embed_dim: int = 512

    # --- captioner (VLM) ---
    caption_enabled: bool = True
    vlm_backend: str = "mock"  # "mock" | "vllm"
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_api_key: str = "EMPTY"

    # --- chat sLLM ---
    summarize_enabled: bool = True
    chat_backend: str = "mock"  # "mock" | "vllm"
    chat_base_url: str = "http://localhost:8000/v1"
    chat_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    chat_api_key: str = "EMPTY"

    # --- indexing granularity ---
    # "folder": one representative image per leaf folder (near-duplicate scenes).
    # "image":  every image is its own record (diverse, individually-labeled datasets).
    index_granularity: str = "folder"
    # YOLO class-id -> display name, e.g. {"0": "사람", "3": "불"}. Used for captions/RAG.
    class_names: dict = field(default_factory=dict)

    # --- retrieval / ui knobs ---
    top_k: int = 24
    score_threshold: float = 0.0  # hide hits below this similarity (display-side filter)
    # retrieval mode: "vector" (CLIP only), "keyword" (BM25 only), "hybrid" (RRF of both)
    search_mode: str = "hybrid"
    agentic_enabled: bool = False  # route queries through the planner->retrieve->graph->summarize pipeline
    thumb_size: int = 256

    # --- object graph (GraphDB) ---
    graph_enabled: bool = True
    graph_backend: str = "memory"  # "memory" (pure-python) | "kuzu" (embedded GraphDB)
    max_members_for_repr: int = 64
    image_exts: list[str] = field(default_factory=lambda: list(DEFAULT_IMAGE_EXTS))

    # --- 런타임 전용 (절대 영속화되지 않음) ---
    # compare=False: 두 설정의 동등성 비교 시 이 폴백 상태 필드는 무시한다.
    degraded: bool = field(default=False, compare=False)
    degraded_reason: str = field(default="", compare=False)

    # to_dict/load에서 제외할 런타임 전용 필드 이름 목록(저장 대상이 아님).
    _RUNTIME = ("degraded", "degraded_reason")

    def to_dict(self) -> dict:
        """저장용 dict로 변환한다. 런타임 전용 필드(degraded 등)는 제거한다."""
        d = asdict(self)
        for k in self._RUNTIME:
            # 키가 없을 수도 있으므로 pop의 기본값으로 안전하게 제거.
            d.pop(k, None)
        return d

    def save(self, path: Path) -> None:
        """설정을 JSON 파일로 저장한다. 한글이 깨지지 않도록 ensure_ascii=False 사용."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        """JSON에서 설정을 읽는다. 파일이 없거나 깨졌으면 기본 설정으로 폴백한다."""
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 손상된 설정 때문에 앱이 못 뜨면 안 되므로 조용히 기본값으로 시작한다.
            return cls()
        # 알 수 없는 키(이전 버전 잔재 등)와 런타임 전용 키는 버려 생성자 오류를 피한다.
        known = {f.name for f in fields(cls)} - set(cls._RUNTIME)
        clean = {k: v for k, v in raw.items() if k in known}
        return cls(**clean)

    def mark_degraded(self, reason: str) -> None:
        """실제 백엔드 사용 실패로 mock 폴백했음을 표시한다(UI 경고 표기에 사용)."""
        self.degraded = True
        self.degraded_reason = reason


def save_class_names_yaml(path: Path, names: dict) -> None:
    """사용자의 클래스명 매핑을 ultralytics 스타일 YAML로 저장한다.

    이 파일은 앱 데이터 디렉터리에 두며(데이터셋 내부가 아님) 손으로 편집할 수 있다.
    시작 시 :func:`load_class_names_yaml`가 다시 읽어 들인다.
    """
    import yaml

    def _key(k):
        """가능하면 키를 정수로 변환한다(ultralytics는 클래스 id를 정수로 쓴다)."""
        ks = str(k)
        return int(ks) if ks.isdigit() else ks

    # values는 항상 문자열(표시명)로 강제해 YAML 타입이 흔들리지 않게 한다.
    payload = {"names": {_key(k): str(v) for k, v in names.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_unicode=True 덕분에 한글이 \uXXXX로 이스케이프되지 않고 그대로 기록된다.
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )


def load_class_names_yaml(path: Path) -> dict[str, str]:
    """클래스명 YAML을 읽어 {id문자열: 이름} 딕셔너리로 반환한다.

    허용 형식: ultralytics 스타일 ``names: {0: 사람}`` 또는 ``names: [사람, …]``,
    혹은 최상위가 곧 매핑인 형태. 파일이 없거나 형식이 이상하면 빈 dict({})를 반환한다.
    """
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        # 파싱 실패/파일 부재 등 어떤 오류든 빈 매핑으로 처리해 호출부를 단순화한다.
        return {}
    if not isinstance(data, dict):
        return {}
    # 'names' 키가 있으면 그 값을, 없으면 최상위 dict 자체를 매핑으로 본다.
    names = data.get("names", data)
    if isinstance(names, dict):
        # 키/값을 모두 문자열로 정규화(정수 id도 "0" 형태로 통일).
        return {str(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        # 리스트형이면 인덱스를 곧 클래스 id로 간주한다(ultralytics 관례).
        return {str(i): str(v) for i, v in enumerate(names)}
    return {}
