"""썸네일을 QThreadPool에서 디코드/생성하고, 완성된 QImage를 UI 스레드로 전달한다.

QImage는 워커 스레드에서 만들어도 안전하다(허용됨). 반면 QPixmap은 반드시 GUI
스레드에서 생성해야 하므로, 갤러리는 done 슬롯 안에서 QImage를 QPixmap으로 변환한다.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, Signal

from imgsearch.thumbs import ensure_thumb


class ThumbSignals(QObject):
    """QRunnable은 시그널을 가질 수 없으므로, 완료 통지를 위한 시그널을 별도 QObject로 둔다."""

    # (이미지 경로, 생성된 QImage, 세대 번호) — 세대 번호로 오래된 결과를 폐기 판단한다.
    done = Signal(str, object, int)  # (image_path, QImage, generation)


class ThumbTask(QRunnable):
    """단일 이미지의 썸네일을 생성하는 스레드풀 작업(QRunnable)."""

    def __init__(self, image_path: str, size: int, signals: ThumbSignals, generation: int = 0) -> None:
        """작업 파라미터 보관. generation은 검색 결과가 바뀔 때 이전 작업을 무효화하는 토큰."""
        super().__init__()
        self.image_path = image_path
        self.size = size
        self.signals = signals
        self.generation = generation
        # 실행이 끝나면 풀이 이 객체를 자동 삭제하도록 한다(수명 관리 단순화).
        self.setAutoDelete(True)

    def run(self) -> None:
        """워커 스레드에서 썸네일 파일을 확보하고 QImage로 로드해 done 시그널로 보낸다."""
        # QtGui import를 함수 안으로 미룬다: 이 모듈을 임포트하는 비-GUI 코드(테스트 등)에서
        # 불필요한 GUI 의존성을 끌어오지 않기 위함.
        from PySide6.QtGui import QImage

        # 디스크 캐시에서 썸네일 경로를 얻거나 없으면 생성(실패 시 None).
        path = ensure_thumb(self.image_path, self.size)
        # 생성 실패 시 빈 QImage를 보내 슬롯 측에서 일관되게 처리하도록 한다.
        img = QImage(str(path)) if path else QImage()
        self.signals.done.emit(self.image_path, img, self.generation)
