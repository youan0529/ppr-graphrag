"""Embedding base interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class BaseEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts."""

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query."""
