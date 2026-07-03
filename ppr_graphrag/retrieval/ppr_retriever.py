"""Personalized PageRank retriever skeleton."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import networkx as nx

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.core.config import GraphConfig, RetrievalConfig
from ppr_graphrag.data.schema import Document, Query
from ppr_graphrag.graph.builders import build_token_overlap_graph
from ppr_graphrag.retrieval.base import RetrievalResult
from ppr_graphrag.retrieval.bm25_retriever import BM25Retriever


class PPRRetriever:
    def __init__(self, retrieval_config: RetrievalConfig, graph_config: GraphConfig, seed_retriever: BM25Retriever | None = None):
        self.retrieval_config = retrieval_config
        self.graph_config = graph_config
        self.seed_retriever = seed_retriever or BM25Retriever()
        self.corpus: list[Document] = []
        self.graph: nx.Graph | None = None

    def build(self, corpus: list[Document], artifact_manager: ArtifactManager | None = None) -> None:
        self.corpus = corpus
        self.seed_retriever.build(corpus, artifact_manager)
        self.graph = build_token_overlap_graph(corpus, self.graph_config.directed, self.graph_config.add_self_loops)

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        if self.graph is None:
            raise RuntimeError("PPRRetriever has not been built or loaded")
        seed_results = self.seed_retriever.retrieve(query, min(max(top_k, 3), len(self.corpus)))
        personalization = {node: 0.0 for node in self.graph.nodes}
        total = 0.0
        for result in seed_results:
            weight = max(result.score, 0.0) + 1e-6
            personalization[result.doc_id] = personalization.get(result.doc_id, 0.0) + weight
            total += weight
        if total == 0.0 and seed_results:
            for result in seed_results:
                personalization[result.doc_id] = 1.0 / len(seed_results)
        scores = nx.pagerank(
            self.graph,
            alpha=self.retrieval_config.ppr_alpha,
            personalization=personalization,
            max_iter=self.retrieval_config.ppr_max_iter,
            tol=self.retrieval_config.ppr_tol,
            weight="weight",
        )
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievalResult(query_id=query.query_id, doc_id=str(doc_id), score=float(score), rank=rank, metadata={"method": "ppr"})
            for rank, (doc_id, score) in enumerate(ranked, start=1)
        ]

    def save(self, artifact_manager: ArtifactManager) -> None:
        if self.graph is None:
            raise RuntimeError("PPRRetriever has no graph to save")
        artifact_manager.save_json_atomic([asdict(doc) for doc in self.corpus], "artifacts", "ppr_corpus.json")
        graph_path = Path(self.graph_config.graph_path)
        parts = graph_path.parts if not graph_path.is_absolute() else (str(graph_path),)
        artifact_manager.save_pickle_atomic(self.graph, *parts)

    def load(self, artifact_manager: ArtifactManager) -> None:
        rows = artifact_manager.load_json("artifacts", "ppr_corpus.json")
        self.corpus = [Document(**row) for row in rows]
        self.seed_retriever.build(self.corpus, artifact_manager)
        graph_path = Path(self.graph_config.graph_path)
        parts = graph_path.parts if not graph_path.is_absolute() else (str(graph_path),)
        self.graph = artifact_manager.load_pickle(*parts)
