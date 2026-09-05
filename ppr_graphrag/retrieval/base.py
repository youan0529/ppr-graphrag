"""Retriever base interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    query_id: str
    doc_id: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
