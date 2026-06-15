"""사용자별 로그 디렉터리에 회전(rotating) 파일 로거를 두고 콘솔로도 미러링한다."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 로그 포맷: 시각, 레벨(7칸 정렬), 로거 이름, 메시지.
_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(logs_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """'imgsearch' 루트 로거에 파일+콘솔 핸들러를 설정한다(중복 호출에 안전).

    파일 핸들러는 2MB마다 회전하며 최대 3개의 백업을 유지한다.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("imgsearch")
    if logger.handlers:  # 이미 설정됨 → 핸들러 중복 추가를 막기 위한 멱등 처리
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(_FMT)

    # 파일 핸들러: 용량 기반 회전으로 로그가 무한정 커지는 것을 방지.
    fh = RotatingFileHandler(
        logs_dir / "imgsearch.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 콘솔 핸들러: 개발/디버깅 중 표준 스트림에서도 같은 로그를 보기 위함.
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 루트 로거로의 전파를 끊어 메시지가 중복 출력되지 않게 한다.
    logger.propagate = False
    return logger


def get_logger(name: str = "imgsearch") -> logging.Logger:
    """'imgsearch' 네임스페이스 하위 로거를 반환한다(이름을 일관되게 접두사 처리)."""
    # 이미 'imgsearch' 또는 그 하위 이름이면 그대로 쓰고,
    if name == "imgsearch" or name.startswith("imgsearch."):
        return logging.getLogger(name)
    # 그 외 짧은 이름은 'imgsearch.' 접두사를 붙여 같은 트리에 묶는다.
    return logging.getLogger(f"imgsearch.{name}")
