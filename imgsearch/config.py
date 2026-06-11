"""Application configuration: a JSON-persisted dataclass.

Backends are selected by string (``"mock"`` / ``"jina-clip"`` / ``"vllm"``) so the rest
of the app never imports a model library directly. ``degraded`` / ``degraded_reason`` are
runtime-only (set when a requested real backend is unavailable and we fall back to mock).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"]
SIDECAR_TEXT_EXTS = [".txt", ".caption", ".json"]


@dataclass
class AppConfig:
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
    thumb_size: int = 256
    max_members_for_repr: int = 64
    image_exts: list[str] = field(default_factory=lambda: list(DEFAULT_IMAGE_EXTS))

    # --- runtime only (never persisted) ---
    degraded: bool = field(default=False, compare=False)
    degraded_reason: str = field(default="", compare=False)

    _RUNTIME = ("degraded", "degraded_reason")

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in self._RUNTIME:
            d.pop(k, None)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        known = {f.name for f in fields(cls)} - set(cls._RUNTIME)
        clean = {k: v for k, v in raw.items() if k in known}
        return cls(**clean)

    def mark_degraded(self, reason: str) -> None:
        self.degraded = True
        self.degraded_reason = reason


def save_class_names_yaml(path: Path, names: dict) -> None:
    """Write the user's class-name mapping as an ultralytics-style YAML.

    Lives in the app-data dir (never inside the dataset) and is hand-editable;
    :func:`load_class_names_yaml` reads it back on startup.
    """
    import yaml

    def _key(k):
        ks = str(k)
        return int(ks) if ks.isdigit() else ks

    payload = {"names": {_key(k): str(v) for k, v in names.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )


def load_class_names_yaml(path: Path) -> dict[str, str]:
    """Read a class-name YAML: either ``names: {0: 사람}`` / ``names: [사람, …]``
    (ultralytics style) or a bare top-level mapping. Returns {} when absent/invalid."""
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    names = data.get("names", data)
    if isinstance(names, dict):
        return {str(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {str(i): str(v) for i, v in enumerate(names)}
    return {}
