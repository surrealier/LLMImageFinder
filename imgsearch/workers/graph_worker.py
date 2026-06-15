"""객체 그래프를 UI 스레드 바깥에서 구축한다(데이터셋 라벨을 순회; 1k 이미지에 약 1초)."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, Signal, Slot

from imgsearch.config import AppConfig
from imgsearch.graph.base import GraphStore
from imgsearch.graph.builder import build_records


class GraphBuildWorker(QObject):
    """데이터셋 라벨로부터 그래프 레코드를 만들어 GraphStore에 적재하는 워커."""

    # 그래프에 담긴, 라벨이 있는 이미지 개수
    finished = Signal(int)  # number of labeled images in the graph
    error = Signal(str)

    def __init__(self, graph: GraphStore, root: str, config: AppConfig) -> None:
        """대상 그래프 저장소, 데이터셋 루트, 설정(클래스명 매핑 등)을 보관한다."""
        super().__init__()
        self._graph = graph
        self._root = root
        self._cfg = config

    @Slot()
    def run(self) -> None:
        """라벨을 읽어 레코드를 만들고 그래프를 빌드한 뒤, 적재된 개수를 보고한다."""
        try:
            # 데이터셋 루트를 훑어 (이미지, 객체 라벨) 레코드 목록을 만든다.
            records = build_records(self._root, self._cfg)
            self._graph.build(records)
            self.finished.emit(self._graph.count())
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
