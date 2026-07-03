"""Index-building pipeline helpers."""

from __future__ import annotations

from pathlib import Path

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.core.cache import SQLiteCache
from ppr_graphrag.core.config import AppConfig
from ppr_graphrag.data.schema import Document
from ppr_graphrag.embedding.cached_embedder import CachedEmbedder
from ppr_graphrag.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from ppr_graphrag.retrieval.base import BaseRetriever
from ppr_graphrag.retrieval.bm25_retriever import BM25Retriever
from ppr_graphrag.retrieval.dense_retriever import DenseRetriever
from ppr_graphrag.retrieval.hybrid_retriever import HybridRetriever
from ppr_graphrag.retrieval.ppr_retriever import PPRRetriever


def run_dir(config: AppConfig) -> Path:
    return Path(config.experiment.output_dir) / config.experiment.run_name


def _cache_path(artifact_manager: ArtifactManager, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else artifact_manager.path(*path.parts)


def make_embedder(config: AppConfig, artifact_manager: ArtifactManager):
    base = SentenceTransformerEmbedder(config.embedding)
    if not config.embedding.cache_enabled:
        return base
    return CachedEmbedder(base, SQLiteCache(_cache_path(artifact_manager, config.embedding.cache_path)), config.embedding)


def make_retriever(config: AppConfig, artifact_manager: ArtifactManager | None = None) -> BaseRetriever:
    method = config.retrieval.method.lower()
    if method == "bm25":
        return BM25Retriever()
    if method == "dense":
        if artifact_manager is None:
            raise ValueError("DenseRetriever requires an ArtifactManager for embedding cache paths")
        return DenseRetriever(make_embedder(config, artifact_manager))
    if method == "ppr":
        return PPRRetriever(config.retrieval, config.graph)
    if method == "hybrid":
        return HybridRetriever([BM25Retriever(), PPRRetriever(config.retrieval, config.graph)], weights=[0.5, 0.5])
    raise ValueError(f"Unknown retrieval.method: {config.retrieval.method}")


def build_and_save_index(config: AppConfig, corpus: list[Document], artifact_manager: ArtifactManager) -> BaseRetriever:
    retriever = make_retriever(config, artifact_manager)
    retriever.build(corpus, artifact_manager)
    retriever.save(artifact_manager)
    artifact_manager.mark_stage_done("index", {"method": config.retrieval.method, "num_docs": len(corpus)})
    artifact_manager.write_manifest(config)
    return retriever
