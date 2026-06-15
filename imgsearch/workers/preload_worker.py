"""실제 백엔드(jina-clip 가중치를 GPU에 적재 등)를 UI 스레드 바깥에서 로딩한다.

창은 "모델 로딩 중…" 상태로 즉시 표시되고, 이 워커가 끝나면 MainWindow가 GUI
스레드에서 실제(live) 백엔드로 교체(swap)한다. 무거운 모델 로딩으로 첫 화면이
지연되는 것을 막기 위한 구조다.
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, Signal, Slot

from imgsearch.config import AppConfig
from imgsearch.core.registry import build_backends


class PreloadWorker(QObject):
    """설정에 맞는 백엔드 묶음을 백그라운드에서 생성해 finished로 전달하는 워커."""

    # (embedder, captioner, chat) 튜플을 UI로 넘긴다.
    finished = Signal(object)  # (embedder, captioner, chat)
    error = Signal(str)

    def __init__(self, config: AppConfig) -> None:
        """로딩에 사용할 앱 설정을 보관한다(실제 적재는 run()에서 수행)."""
        super().__init__()
        self._cfg = config

    @Slot()
    def run(self) -> None:
        """백엔드를 빌드해 finished로 보낸다. 진짜 예상 밖 오류만 error로 보고한다."""
        try:
            # build_backends는 실패 시 자체적으로 mock으로 폴백하고 cfg.mark_degraded를
            # 호출한다. 따라서 여기서 error가 발생하는 경우는 정말 예상치 못한 예외뿐이다.
            backends = build_backends(self._cfg)
            self.finished.emit(backends)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
