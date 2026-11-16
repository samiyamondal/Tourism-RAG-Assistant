"""
logger.py
---------
Standardized logger factory for the Tourism RAG Assistant.

Produces a logger that writes to both the console and a rotating log file
under `APP_LOG_DIR`. All modules should obtain their logger via
`get_logger(__name__)` rather than configuring logging directly.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_CONFIGURED_LOGGERS = set()

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: str = "INFO",
    log_filename: str = "tourism_rag.log",
) -> logging.Logger:
    """
    Return a configured logger. Safe to call multiple times with the same
    name — handlers are only attached once per logger name.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (best-effort; don't crash the app if the filesystem is read-only)
    try:
        resolved_log_dir = Path(log_dir) if log_dir else Path("./logs")
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            resolved_log_dir / log_filename,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not attach file handler for logs; continuing with console only.")

    _CONFIGURED_LOGGERS.add(name)
    return logger
