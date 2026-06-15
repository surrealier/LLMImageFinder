"""오래 걸리는 인덱싱 작업을 백그라운드 스레드에서 수행한다.

진행률(progress)/로그(log)/완료(finished)/오류(error) 시그널을 내보내며, 협력적
취소(cancel)도 지원한다. UI 스레드가 임베딩/캡션 생성으로 멈추지 않도록 분리한다.
"""

from __future__ import annotations

import traceback
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal, Slot

from imgsearch.index.indexer import Indexer


class IndexWorker(QObject):
    """Indexer.build / refresh_captions를 워커 스레드에서 실행하는 QObject 워커."""

    # 인덱싱 진행 상황 객체(IndexProgress)를 UI로 전달
    progress = Signal(object)  # IndexProgress
    # 사람이 읽을 로그 한 줄
    log = Signal(str)
    # 완료 결과: build 모드는 BuildReport, refresh 모드는 갱신 건수(int)
    finished = Signal(object)  # BuildReport (build) | int updated (refresh)
    # 예기치 못한 예외 메시지
    error = Signal(str)

    def __init__(
        self,
        indexer: Indexer,
        root: str,
        full_rebuild: bool = False,
        thumb_cb: Optional[Callable[[str], None]] = None,
        mode: str = "build",  # "build" | "refresh" (caption-only, no re-embedding)
    ) -> None:
        """인덱싱 파라미터를 보관한다(실제 작업은 run()에서 스레드 위에서 수행).

        mode="refresh"는 캡션만 다시 만들고 임베딩은 재생성하지 않는 가벼운 갱신이다.
        thumb_cb가 None이면 아무것도 하지 않는 no-op 콜백으로 대체한다.
        """
        super().__init__()
        self._indexer = indexer
        self._root = root
        self._full = full_rebuild
        # 콜백 미지정 시 None 분기를 매번 검사하지 않도록 no-op 람다로 정규화한다.
        self._thumb_cb = thumb_cb or (lambda _p: None)
        self._mode = mode
        # 취소 플래그: cancel() 슬롯이 True로 바꾸면 인덱서가 should_cancel로 감지한다.
        self._cancel = False

    @Slot()
    def cancel(self) -> None:
        """취소 요청 플래그만 세운다(즉시 중단이 아니라 다음 체크포인트에서 멈춤)."""
        self._cancel = True

    @Slot()
    def run(self) -> None:
        """모드에 따라 캡션 갱신 또는 전체 빌드를 실행하고 결과/오류 시그널을 낸다."""
        try:
            if self._mode == "refresh":
                # 캡션만 다시 생성(임베딩 재계산 없음) — 클래스명 변경 후 빠른 반영용.
                n = self._indexer.refresh_captions(
                    self._root,
                    progress_cb=self.progress.emit,
                    log_cb=self.log.emit,
                    # 람다로 매번 현재 취소 플래그를 읽어 협력적 취소를 가능하게 한다.
                    should_cancel=lambda: self._cancel,
                )
            else:
                # 전체 빌드: 임베딩/캡션/썸네일까지 생성. full_rebuild면 기존 인덱스를 폐기.
                n = self._indexer.build(
                    self._root,
                    progress_cb=self.progress.emit,
                    log_cb=self.log.emit,
                    should_cancel=lambda: self._cancel,
                    thumb_cb=self._thumb_cb,
                    full_rebuild=self._full,
                )
            self.finished.emit(n)
        except Exception as e:
            # 워커 스레드에서 예외가 새어나가면 조용히 죽으므로, 스택을 찍고 error로 보고.
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
