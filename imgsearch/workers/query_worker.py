"""검색을 UI 스레드 바깥에서 실행해, 임베딩/LLM 호출 중에도 창이 멈추지 않게 한다.

두 가지 모드를 지원한다: 일반 SearchService.query, 또는 에이전트형(A2A) 파이프라인.
에이전트 모드에서는 단계별 추적 로그를 ``trace`` 시그널로 내보내며, 이 시그널은
큐 연결을 통해 GUI 스레드로 안전하게 전달된다.
"""

from __future__ import annotations

import traceback
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from imgsearch.core.services import SearchService


class QueryWorker(QObject):
    """질의 텍스트를 받아 검색을 수행하고 결과/오류/추적 시그널을 내보내는 워커."""

    # 검색 결과(QueryResult)
    finished = Signal(object)  # QueryResult
    # 예외 메시지
    error = Signal(str)
    # 에이전트 모드에서만 사용하는 단계별 추적 한 줄
    trace = Signal(str)  # agentic per-step trace line (agentic mode only)

    def __init__(
        self,
        service: SearchService,
        text: str,
        top_k: int,
        agent: Optional[object] = None,  # AgenticSearch when agentic mode is on
    ) -> None:
        """검색에 필요한 입력을 보관한다. agent가 주어지면 에이전트 모드로 동작한다."""
        super().__init__()
        self._service = service
        self._text = text
        self._k = top_k
        # None이면 일반 검색, 객체가 있으면 에이전트형 파이프라인 사용 여부를 결정.
        self._agent = agent

    @Slot()
    def run(self) -> None:
        """에이전트 유무에 따라 검색을 실행하고 결과 또는 오류를 시그널로 전달한다."""
        try:
            if self._agent is not None:
                # 에이전트 모드: step_cb로 단계별 진행을 trace 시그널에 흘려보낸다.
                result = self._agent.run(self._text, step_cb=self.trace.emit, k=self._k)
            else:
                # 일반 모드: 임베딩 기반(또는 하이브리드) 단일 검색.
                result = self._service.query(self._text, self._k)
            self.finished.emit(result)
        except Exception as e:
            # 워커 스레드 예외는 표면화되지 않으므로 스택을 찍고 error로 보고한다.
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
