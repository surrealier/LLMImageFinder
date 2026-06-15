"""Pick a GraphStore backend from config, falling back to memory.

Mirrors core.registry: the embedded kuzu backend is lazily imported inside a try;
if it is unavailable (missing ``[graph]`` extra) or fails to open, we use the
pure-python MemoryGraphStore so the object-graph feature always works.
"""

from __future__ import annotations

from pathlib import Path

from imgsearch.config import AppConfig
from imgsearch.graph.base import GraphStore
from imgsearch.graph.memory_graph import MemoryGraphStore
from imgsearch.logging_setup import get_logger

log = get_logger("graph.registry")


def build_graph_store(cfg: AppConfig, graph_dir: str | Path) -> GraphStore:
    if cfg.graph_backend == "kuzu":
        try:
            from imgsearch.graph.kuzu_graph import KuzuGraphStore

            store = KuzuGraphStore(graph_dir)
            log.info("GraphStore: kuzu (%s)", graph_dir)
            return store
        except Exception as e:  # missing extra, native load error, locked db, …
            log.warning("kuzu unavailable, using in-memory graph: %s", e)
            cfg.mark_degraded(f"GraphDB(kuzu) 사용 불가 — 메모리 그래프로 대체: {e}")
    return MemoryGraphStore()
