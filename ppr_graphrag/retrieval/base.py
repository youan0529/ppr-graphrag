"""Retriever base interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.data.schema import Document, Query


@dataclass
class RetrievalResult:
    query_id: str
    doc_id: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(Protocol):
    def build(self, corpus: list[Document], artifact_manager: ArtifactManager | None = None) -> None:
        """Build an index from corpus documents."""

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        """Retrieve ranked results for a query."""

    def save(self, artifact_manager: ArtifactManager) -> None:
        """Save retriever state."""

    def load(self, artifact_manager: ArtifactManager) -> None:
        """Load retriever state."""
