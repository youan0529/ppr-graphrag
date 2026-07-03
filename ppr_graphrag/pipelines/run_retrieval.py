"""Retrieval-running pipeline helpers."""

from __future__ import annotations

from dataclasses import asdict

from tqdm import tqdm

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.core.config import AppConfig
from ppr_graphrag.data.schema import Document, Query
from ppr_graphrag.evaluation.retrieval_metrics import hit_at_k, mrr_at_k, recall_at_k
from ppr_graphrag.pipelines.build_index import build_and_save_index, make_retriever
from ppr_graphrag.retrieval.base import BaseRetriever, RetrievalResult


def load_or_build_retriever(config: AppConfig, corpus: list[Document], artifact_manager: ArtifactManager) -> BaseRetriever:
    retriever = make_retriever(config, artifact_manager)
    if (
        config.resume.resume_index
        and artifact_manager.is_stage_done("index")
        and not config.resume.force_rebuild_index
    ):
        retriever.load(artifact_manager)
        return retriever
    return build_and_save_index(config, corpus, artifact_manager)


def run_retrieval(
    config: AppConfig,
    corpus: list[Document],
    queries: list[Query],
    artifact_manager: ArtifactManager,
) -> dict[str, float]:
    retriever = load_or_build_retriever(config, corpus, artifact_manager)
    if config.resume.force_retrieve:
        artifact_manager.delete("artifacts", "retrieval_results.jsonl")
        artifact_manager.delete("artifacts", "retrieval.done.json")
    existing_rows = artifact_manager.load_jsonl("artifacts", "retrieval_results.jsonl")
    done = {row["query_id"] for row in existing_rows} if config.resume.resume_retrieval else set()
    results_by_query: dict[str, list[RetrievalResult]] = {}
    for row in existing_rows:
        if row["query_id"] in done:
            results_by_query.setdefault(row["query_id"], []).append(RetrievalResult(**row))

    for query in tqdm(queries, desc="retrieving"):
        if query.query_id in done:
            continue
        results = retriever.retrieve(query, config.retrieval.top_k)
        results_by_query[query.query_id] = results
        for result in results:
            artifact_manager.append_jsonl(asdict(result), "artifacts", "retrieval_results.jsonl")

    k = config.retrieval.top_k
    metrics = {
        f"recall@{k}": recall_at_k(results_by_query, queries, k),
        f"hit@{k}": hit_at_k(results_by_query, queries, k),
        f"mrr@{k}": mrr_at_k(results_by_query, queries, k),
    }
    artifact_manager.save_json_atomic(metrics, "metrics", "retrieval_metrics.json")
    artifact_manager.mark_stage_done("retrieval", {"num_queries": len(queries), "top_k": k})
    return metrics
