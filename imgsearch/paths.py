"""사용자별 애플리케이션 데이터 저장 위치.

앱이 쓰는 모든 산출물은 OS의 사용자별 앱 데이터 디렉터리 아래에 둔다(사용자의 데이터셋
폴더 안에는 절대 쓰지 않는다 — 원본 오염 방지). ``IMGSEARCH_HOME`` 환경변수를 설정하면
루트를 덮어쓸 수 있는데, 테스트가 격리되고 버려도 되는 홈을 쓰기 위해 활용한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs

# OS별 표준 경로(예: Windows의 AppData)를 산출하는 헬퍼. 앱/조직 이름으로 경로가 결정된다.
_DIRS = PlatformDirs(appname="ImgSearch", appauthor="MarkAny")


@dataclass(frozen=True)
class AppPaths:
    """앱이 사용하는 모든 파일/디렉터리 경로를 한곳에 모은 불변(frozen) 묶음."""

    config_file: Path
    class_names_file: Path  # 사용자가 편집 가능한 YAML(id -> 이름), config 옆에 위치
    history_file: Path  # 최근 검색어(JSON 리스트)
    index_meta_file: Path  # 마지막 빌드 정보(언제/무엇을/얼마나), chroma 옆에 위치
    chroma_dir: Path
    graph_dir: Path  # 임베디드 GraphDB(kuzu) 영속 데이터, chroma 옆에 위치
    thumbs_dir: Path
    logs_dir: Path


def app_paths() -> AppPaths:
    """앱 데이터 경로들을 계산해 반환한다(생성은 하지 않음; :func:`ensure_dirs` 사용)."""
    # 환경변수로 홈이 지정되면(주로 테스트) 그 한 루트 아래에 모든 경로를 평평하게 둔다.
    home = os.environ.get("IMGSEARCH_HOME")
    if home:
        root = Path(home)
        return AppPaths(
            config_file=root / "config.json",
            class_names_file=root / "class_names.yaml",
            history_file=root / "history.json",
            index_meta_file=root / "index_meta.json",
            chroma_dir=root / "chroma",
            graph_dir=root / "graph",
            thumbs_dir=root / "thumbs",
            logs_dir=root / "logs",
        )
    # 일반 실행: OS 관례에 따라 설정/데이터/캐시/로그를 서로 다른 표준 위치에 분산한다.
    cfg_dir = Path(_DIRS.user_config_dir)
    data_dir = Path(_DIRS.user_data_dir)
    return AppPaths(
        config_file=cfg_dir / "config.json",
        class_names_file=cfg_dir / "class_names.yaml",
        history_file=cfg_dir / "history.json",
        index_meta_file=data_dir / "index_meta.json",
        chroma_dir=data_dir / "chroma",
        graph_dir=data_dir / "graph",
        # 썸네일은 재생성 가능한 캐시이므로 캐시 디렉터리에 둔다(지워져도 무방).
        thumbs_dir=Path(_DIRS.user_cache_dir) / "thumbs",
        logs_dir=Path(_DIRS.user_log_dir),
    )


def ensure_dirs(p: AppPaths) -> AppPaths:
    """앱 구동에 필요한 디렉터리들을 미리 만들어 두고, 받은 경로 묶음을 그대로 반환한다."""
    # config_file의 부모를 만들면 같은 디렉터리에 사는 class_names/history도 함께 커버된다.
    p.config_file.parent.mkdir(parents=True, exist_ok=True)
    # 나머지 디렉터리들은 각자 위치가 다를 수 있으므로 개별로 보장한다.
    p.chroma_dir.mkdir(parents=True, exist_ok=True)
    p.thumbs_dir.mkdir(parents=True, exist_ok=True)
    p.logs_dir.mkdir(parents=True, exist_ok=True)
    return p
