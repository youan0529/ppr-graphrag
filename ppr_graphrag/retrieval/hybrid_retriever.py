"""Hybrid retriever that linearly combines scores from child retrievers."""

from __future__ import annotations

from collections import defaultdict

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.data.schema import Document, Query
from ppr_graphrag.retrieval.base import BaseRetriever, RetrievalResult


class HybridRetriever:
    def __init__(self, retrievers: list[BaseRetriever], weights: list[float] | None = None):
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one child retriever")
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)

    def build(self, corpus: list[Document], artifact_manager: ArtifactManager | None = None) -> None:
        for retriever in self.retrievers:
            retriever.build(corpus, artifact_manager)

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        scores: dict[str, float] = defaultdict(float)
        metadata: dict[str, dict] = defaultdict(dict)
        for weight, retriever in zip(self.weights, self.retrievers, strict=True):
            results = retriever.retrieve(query, top_k)
            max_score = max((abs(r.score) for r in results), default=1.0) or 1.0
            for result in results:
                scores[result.doc_id] += weight * (result.score / max_score)
                metadata[result.doc_id]["hybrid"] = True
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievalResult(query_id=query.query_id, doc_id=doc_id, score=float(score), rank=rank, metadata=metadata[doc_id])
            for rank, (doc_id, score) in enumerate(ranked, start=1)
        ]

    def save(self, artifact_manager: ArtifactManager) -> None:
        for i, retriever in enumerate(self.retrievers):
            retriever.save(artifact_manager)
        artifact_manager.save_json_atomic({"weights": self.weights}, "artifacts", "hybrid.json")

    def load(self, artifact_manager: ArtifactManager) -> None:
        for retriever in self.retrievers:
            retriever.load(artifact_manager)
