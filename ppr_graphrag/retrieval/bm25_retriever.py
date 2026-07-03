"""BM25 retriever with token-overlap fallback."""

from __future__ import annotations

import re
from dataclasses import asdict

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.data.schema import Document, Query
from ppr_graphrag.retrieval.base import RetrievalResult


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


class BM25Retriever:
    def __init__(self) -> None:
        self.corpus: list[Document] = []
        self.doc_tokens: list[list[str]] = []
        self.model = None

    def build(self, corpus: list[Document], artifact_manager: ArtifactManager | None = None) -> None:
        self.corpus = corpus
        self.doc_tokens = [tokenize(f"{doc.title or ''} {doc.text}") for doc in corpus]
        try:
            from rank_bm25 import BM25Okapi

            self.model = BM25Okapi(self.doc_tokens)
        except ImportError:
            self.model = None

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        q_tokens = tokenize(query.question)
        if self.model is not None:
            scores = list(map(float, self.model.get_scores(q_tokens)))
        else:
            q_set = set(q_tokens)
            scores = [float(len(q_set & set(tokens))) for tokens in self.doc_tokens]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievalResult(query_id=query.query_id, doc_id=self.corpus[i].doc_id, score=score, rank=rank)
            for rank, (i, score) in enumerate(ranked, start=1)
        ]

    def save(self, artifact_manager: ArtifactManager) -> None:
        artifact_manager.save_json_atomic([asdict(doc) for doc in self.corpus], "artifacts", "bm25_corpus.json")

    def load(self, artifact_manager: ArtifactManager) -> None:
        rows = artifact_manager.load_json("artifacts", "bm25_corpus.json")
        self.build([Document(**row) for row in rows], artifact_manager)
