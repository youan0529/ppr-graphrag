"""Dense vector retriever."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.data.schema import Document, Query
from ppr_graphrag.embedding.base import BaseEmbedder
from ppr_graphrag.retrieval.base import RetrievalResult


class DenseRetriever:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.corpus: list[Document] = []
        self.embeddings: np.ndarray | None = None

    def build(self, corpus: list[Document], artifact_manager: ArtifactManager | None = None) -> None:
        self.corpus = corpus
        texts = [f"{doc.title or ''}\n{doc.text}".strip() for doc in corpus]
        self.embeddings = self.embedder.embed_texts(texts)

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        if self.embeddings is None:
            raise RuntimeError("DenseRetriever has not been built or loaded")
        q = self.embedder.embed_query(query.question)
        scores = self.embeddings @ q
        ranked = np.argsort(-scores)[:top_k]
        return [
            RetrievalResult(query_id=query.query_id, doc_id=self.corpus[int(i)].doc_id, score=float(scores[int(i)]), rank=rank)
            for rank, i in enumerate(ranked, start=1)
        ]

    def save(self, artifact_manager: ArtifactManager) -> None:
        if self.embeddings is None:
            raise RuntimeError("DenseRetriever has no embeddings to save")
        artifact_manager.save_json_atomic([asdict(doc) for doc in self.corpus], "artifacts", "dense_corpus.json")
        artifact_manager.save_pickle_atomic(self.embeddings, "artifacts", "dense_embeddings.pkl")

    def load(self, artifact_manager: ArtifactManager) -> None:
        rows = artifact_manager.load_json("artifacts", "dense_corpus.json")
        self.corpus = [Document(**row) for row in rows]
        self.embeddings = artifact_manager.load_pickle("artifacts", "dense_embeddings.pkl")
