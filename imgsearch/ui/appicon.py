"""앱 아이콘을 코드로 생성한다(별도 바이너리 에셋 없이).

mock-first / 무(無)에셋 설계 철학에 맞춰, 아이콘 PNG/ICO 파일을 저장소에 두지 않고
qtawesome 글리프(돋보기)를 런타임에 픽스맵으로 렌더링해 QIcon을 만든다. 글꼴이 없거나
qtawesome 초기화가 실패해도 앱이 죽지 않도록 항상 (빈) QIcon으로 폴백한다.
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtGui import QIcon

# 앱 강조색(검색/하이라이트에 쓰는 파랑) — 아이콘 글리프 색.
_ACCENT = "#3b82f6"


@lru_cache(maxsize=1)
def app_icon() -> QIcon:
    """앱 윈도우/작업표시줄용 QIcon을 만든다(결과는 한 번만 만들어 캐시).

    여러 픽셀 크기를 한 QIcon에 담아 OS가 상황(타이틀바·작업표시줄·Alt-Tab)에 맞는
    해상도를 고르게 한다. 실패 시 빈 QIcon — 아이콘이 없을 뿐 앱 동작에는 지장이 없다.
    """
    try:
        import qtawesome as qta

        icon = QIcon()
        # 흔히 쓰이는 아이콘 크기들. 각 크기로 글리프를 픽스맵으로 굽는다.
        for size in (16, 24, 32, 48, 64, 128, 256):
            pm = qta.icon("fa5s.search", color=_ACCENT).pixmap(size, size)
            if not pm.isNull():
                icon.addPixmap(pm)
        return icon
    except Exception:
        # qtawesome/글꼴 문제로 실패해도 호출부가 그대로 쓸 수 있도록 빈 아이콘 반환.
        return QIcon()
