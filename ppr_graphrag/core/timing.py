"""Timing utilities."""

from __future__ import annotations

import logging
import time


class Timer:
    """Context manager that logs elapsed wall-clock time."""

    def __init__(self, name: str, logger: logging.Logger | None = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.elapsed: float | None = None

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start
        self.logger.info("%s finished in %.3fs", self.name, self.elapsed)
