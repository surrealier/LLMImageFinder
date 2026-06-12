"""Per-user application data locations.

Everything the app writes lives under the OS per-user app-data dirs (never inside the
user's dataset). Set the ``IMGSEARCH_HOME`` environment variable to override the root
(used by tests to get an isolated, disposable home).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs

_DIRS = PlatformDirs(appname="ImgSearch", appauthor="MarkAny")


@dataclass(frozen=True)
class AppPaths:
    config_file: Path
    class_names_file: Path  # user-editable YAML (id -> name), lives next to config
    history_file: Path  # recent search queries (JSON list)
    index_meta_file: Path  # last build info (when/what/how long), next to chroma
    chroma_dir: Path
    thumbs_dir: Path
    logs_dir: Path


def app_paths() -> AppPaths:
    """Resolve the app-data paths (does not create them; call :func:`ensure_dirs`)."""
    home = os.environ.get("IMGSEARCH_HOME")
    if home:
        root = Path(home)
        return AppPaths(
            config_file=root / "config.json",
            class_names_file=root / "class_names.yaml",
            history_file=root / "history.json",
            index_meta_file=root / "index_meta.json",
            chroma_dir=root / "chroma",
            thumbs_dir=root / "thumbs",
            logs_dir=root / "logs",
        )
    cfg_dir = Path(_DIRS.user_config_dir)
    data_dir = Path(_DIRS.user_data_dir)
    return AppPaths(
        config_file=cfg_dir / "config.json",
        class_names_file=cfg_dir / "class_names.yaml",
        history_file=cfg_dir / "history.json",
        index_meta_file=data_dir / "index_meta.json",
        chroma_dir=data_dir / "chroma",
        thumbs_dir=Path(_DIRS.user_cache_dir) / "thumbs",
        logs_dir=Path(_DIRS.user_log_dir),
    )


def ensure_dirs(p: AppPaths) -> AppPaths:
    p.config_file.parent.mkdir(parents=True, exist_ok=True)
    p.chroma_dir.mkdir(parents=True, exist_ok=True)
    p.thumbs_dir.mkdir(parents=True, exist_ok=True)
    p.logs_dir.mkdir(parents=True, exist_ok=True)
    return p
