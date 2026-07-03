"""Logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_path: str | None = None, level: str = "INFO") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
