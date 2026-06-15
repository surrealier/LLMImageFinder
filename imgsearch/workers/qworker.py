"""워커 QObject를 별도의 QThread 위에서 올바른 생명주기로 실행한다.

스레드와 워커 양쪽에 대한 참조를 보관한다(이 패턴에서 크래시가 발생하는 #1 원인은
실행 도중 이 둘이 가비지 컬렉션으로 수거되도록 방치하는 것이다). 워커는 ``run()``
메서드와 더불어 종료 시그널 ``finished``, ``error``를 반드시 노출해야 한다.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread


class ThreadRunner(QObject):
    """워커-스레드 쌍을 묶어 소유하고, 시작/대기 등 생명주기를 관리하는 헬퍼.

    워커가 끝나면 스레드를 자동으로 종료(quit)시키고 양쪽 객체를 안전하게
    deleteLater로 정리하도록 시그널을 미리 배선해 둔다.
    """

    def __init__(self, worker: QObject, parent: QObject | None = None) -> None:
        """워커를 새 QThread로 옮기고 시그널-슬롯 연결을 구성한다(아직 시작하지는 않음)."""
        super().__init__(parent)
        # 워커는 종료 시그널을 반드시 노출해야 한다. 그렇지 않으면 스레드가 영원히
        # quit되지 않아 앱 종료가 막히게 된다.
        assert hasattr(worker, "finished") and hasattr(worker, "error"), (
            "worker must expose 'finished' and 'error' signals"
        )
        # 강한 참조를 보관해 둔다(이것이 GC로 인한 중간 크래시를 막는 핵심).
        self._worker = worker
        self._thread = QThread()
        # 워커의 슬롯/시그널이 워커 스레드의 이벤트 루프에서 실행되도록 소속을 옮긴다.
        worker.moveToThread(self._thread)
        # 스레드가 시작되면 워커 스레드 컨텍스트에서 run()이 호출된다.
        self._thread.started.connect(worker.run)

        # 정상 완료(finished)든 오류(error)든 어느 쪽이든 스레드 이벤트 루프를 종료시킨다.
        for name in ("finished", "error"):
            getattr(worker, name).connect(self._thread.quit)

        # 스레드가 완전히 끝난 뒤에야 워커와 스레드 객체를 삭제한다(실행 중 삭제 금지).
        self._thread.finished.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

    @property
    def worker(self) -> QObject:
        """소유 중인 워커 객체를 반환한다(시그널 연결 등 외부에서 접근할 때 사용)."""
        return self._worker

    def start(self) -> None:
        """백그라운드 스레드를 시작한다(연결된 started 시그널을 통해 run()이 실행됨)."""
        self._thread.start()

    def is_running(self) -> bool:
        """스레드가 현재 실행 중인지 여부를 반환한다."""
        return self._thread.isRunning()

    def wait(self, ms: int = 8000) -> None:
        """스레드가 실행 중이면 최대 ``ms`` 밀리초 동안 종료를 대기한다(타임아웃 방어)."""
        if self._thread.isRunning():
            self._thread.wait(ms)
