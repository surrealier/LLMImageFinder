"""Rotating file logger (+ console mirror) under the per-user logs dir."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(logs_dir: Path, level: int = logging.INFO) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("imgsearch")
    if logger.handlers:  # idempotent
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(_FMT)

    fh = RotatingFileHandler(
        logs_dir / "imgsearch.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.propagate = False
    return logger


def get_logger(name: str = "imgsearch") -> logging.Logger:
    if name == "imgsearch" or name.startswith("imgsearch."):
        return logging.getLogger(name)
    return logging.getLogger(f"imgsearch.{name}")
