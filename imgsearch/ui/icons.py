"""qtawesome 아이콘 헬퍼 모음 (FontAwesome 5 Solid 글꼴 사용).

각 함수는 도구 모음/버튼에서 쓸 QIcon을 만들어 돌려준다. 색상은 용도별로 의미를 담아
지정한다(예: 위험 동작은 주황/빨강, 정보는 파랑). 색을 생략하면 기본 무채색(_MUTED)을 쓴다.
"""

from __future__ import annotations

import qtawesome as qta
from PySide6.QtGui import QIcon

# 기본 아이콘 색상(차분한 회색-파랑). 강조가 필요 없는 아이콘에 쓰인다.
_MUTED = "#9aa4b2"


def _i(name: str, color: str = _MUTED, **kw) -> QIcon:
    """qtawesome 아이콘 생성을 감싸는 공통 헬퍼.

    qtawesome 초기화 실패(글꼴 누락 등)나 잘못된 아이콘 이름으로 예외가 나도
    앱이 죽지 않도록, 실패 시 빈 QIcon()을 돌려준다(아이콘 없이도 버튼은 동작).
    """
    try:
        return qta.icon(name, color=color, **kw)
    except Exception:
        return QIcon()


def send() -> QIcon:
    """전송(종이비행기) 아이콘 — 파란색."""
    return _i("fa5s.paper-plane", "#3b82f6")


def folder() -> QIcon:
    """폴더 열기 아이콘 — 노란색."""
    return _i("fa5s.folder-open", "#eab308")


def index() -> QIcon:
    """색인(동기화 화살표) 아이콘 — 초록색."""
    return _i("fa5s.sync-alt", "#22c55e")


def rebuild() -> QIcon:
    """전체 재빌드 아이콘 — 경고색(주황). 이 동작은 기존 인덱스를 지우기 때문이다."""
    return _i("fa5s.redo", "#f59e0b")


def search() -> QIcon:
    """검색(돋보기) 아이콘 — 기본색."""
    return _i("fa5s.search")


def settings() -> QIcon:
    """설정(톱니바퀴) 아이콘 — 기본색."""
    return _i("fa5s.cog")


def images() -> QIcon:
    """이미지 모음 아이콘 — 기본색."""
    return _i("fa5s.images")


def zoom_in() -> QIcon:
    """확대(돋보기+) 아이콘 — 기본색."""
    return _i("fa5s.search-plus")


def zoom_out() -> QIcon:
    """축소(돋보기-) 아이콘 — 기본색."""
    return _i("fa5s.search-minus")


def fit() -> QIcon:
    """창에 맞춤(확장) 아이콘 — 기본색."""
    return _i("fa5s.expand")


def prev() -> QIcon:
    """이전(왼쪽 화살표) 아이콘 — 기본색."""
    return _i("fa5s.chevron-left")


def nxt() -> QIcon:
    """다음(오른쪽 화살표) 아이콘 — 기본색."""
    return _i("fa5s.chevron-right")


def tags() -> QIcon:
    """라벨/태그 아이콘 — 보라색."""
    return _i("fa5s.tags", "#a78bfa")


def warn() -> QIcon:
    """경고(삼각형 느낌표) 아이콘 — 주황색."""
    return _i("fa5s.exclamation-triangle", "#f59e0b")


def cancel() -> QIcon:
    """취소(X) 아이콘 — 빨간색."""
    return _i("fa5s.times", "#ef4444")


def copy() -> QIcon:
    """복사 아이콘 — 기본색."""
    return _i("fa5s.copy")


def export() -> QIcon:
    """내보내기 아이콘 — 초록색."""
    return _i("fa5s.file-export", "#34d399")


def info() -> QIcon:
    """정보(i) 아이콘 — 파란색."""
    return _i("fa5s.info-circle", "#60a5fa")


def similar() -> QIcon:
    """유사 이미지(겹친 사각형) 아이콘 — 하늘색."""
    return _i("fa5s.clone", "#7dd3fc")


def eye() -> QIcon:
    """보기(눈) 아이콘 — 기본색."""
    return _i("fa5s.eye")


def graph() -> QIcon:
    """객체 그래프(다이어그램) 아이콘 — 보라색."""
    return _i("fa5s.project-diagram", "#a78bfa")
