"""설정(config)에 따라 GraphStore 백엔드를 고르고, 실패하면 메모리로 대체한다.

core.registry와 같은 패턴을 따른다: 임베디드 kuzu 백엔드를 try 블록 안에서 지연
임포트하고, 사용 불가하거나(``[graph]`` extra 미설치) 열기에 실패하면 순수 파이썬
MemoryGraphStore로 떨어진다 — 객체 그래프 기능이 어떤 환경에서도 항상 동작하도록 보장한다.
"""

from __future__ import annotations

from pathlib import Path

from imgsearch.config import AppConfig
from imgsearch.graph.base import GraphStore
from imgsearch.graph.memory_graph import MemoryGraphStore
from imgsearch.logging_setup import get_logger

log = get_logger("graph.registry")


def build_graph_store(cfg: AppConfig, graph_dir: str | Path) -> GraphStore:
    """설정값에 맞는 GraphStore를 생성해 반환. kuzu 실패 시 메모리 백엔드로 대체한다.

    반환 타입은 구체 클래스가 아니라 GraphStore 프로토콜이므로, 호출부는 어떤 백엔드가
    선택됐는지 신경 쓸 필요가 없다.
    """
    if cfg.graph_backend == "kuzu":
        try:
            # 지연 임포트: kuzu(및 [graph] extra)가 없는 환경에서도 이 모듈 자체는 임포트되도록.
            from imgsearch.graph.kuzu_graph import KuzuGraphStore

            store = KuzuGraphStore(graph_dir)
            log.info("GraphStore: kuzu (%s)", graph_dir)
            return store
        except Exception as e:  # extra 미설치, 네이티브 로드 오류, DB 잠김 등
            # 어떤 이유로든 kuzu가 안 되면 경고만 남기고 메모리로 떨어진다(기능 자체는 유지).
            log.warning("kuzu unavailable, using in-memory graph: %s", e)
            # 사용자에게 보이는 "성능 저하(degraded)" 상태로 표시 — UI 문구라 한글 그대로 유지.
            cfg.mark_degraded(f"GraphDB(kuzu) 사용 불가 — 메모리 그래프로 대체: {e}")
    # kuzu가 아니거나 위에서 대체로 떨어진 경우의 기본 백엔드.
    return MemoryGraphStore()
