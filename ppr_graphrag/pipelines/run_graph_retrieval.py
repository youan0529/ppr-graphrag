"""Resumable retrieval and evaluation over the Fact-Entity graph."""

from __future__ import annotations

import time
from typing import Any

from tqdm import tqdm

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.core.config import AppConfig
from ppr_graphrag.data.schema import Query
from ppr_graphrag.evaluation.retrieval_metrics import hit_at_k, mrr_at_k, recall_at_k
from ppr_graphrag.retrieval.base import RetrievalResult
from ppr_graphrag.retrieval.fact_entity_retriever import FactEntityGraphRetriever


def _paths(method: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    root = ("artifacts", "graph_retrieval")
    return (
        (*root, f"{method}_results.jsonl"),
        (*root, f"{method}_errors.jsonl"),
        ("metrics", f"{method}_metrics.json"),
    )


def _metrics(
    rows: list[dict[str, Any]],
    queries: list[Query],
    top_k: int,
) -> dict[str, float | int]:
    results_by_query = {
        row["query_id"]: [RetrievalResult(**result) for result in row["results"]]
        for row in rows
    }
    metrics: dict[str, float | int] = {
        "num_queries": len(queries),
        "num_queries_with_results": len(results_by_query),
    }
    for k in sorted({value for value in (2, 5, 10, 20, top_k) if value <= top_k}):
        metrics[f"recall@{k}"] = recall_at_k(results_by_query, queries, k)
        metrics[f"hit@{k}"] = hit_at_k(results_by_query, queries, k)
        metrics[f"mrr@{k}"] = mrr_at_k(results_by_query, queries, k)
    return metrics


def run_graph_retrieval(
    config: AppConfig,
    queries: list[Query],
    artifacts: ArtifactManager,
    method: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    if method not in {"dense", "ppr"}:
        raise ValueError("method must be 'dense' or 'ppr'")
    result_parts, error_parts, metric_parts = _paths(method)
    stage_name = f"graph_retrieval_{method}"
    if overwrite:
        artifacts.delete(*result_parts)
        artifacts.delete(*error_parts)
        artifacts.delete(*metric_parts)
        artifacts.delete("artifacts", f"{stage_name}.done.json")

    rows = artifacts.load_jsonl(*result_parts)
    errors = artifacts.load_jsonl(*error_parts)
    completed = {row["query_id"] for row in rows} | {row["query_id"] for row in errors}
    pending = [query for query in queries if query.query_id not in completed]
    started_at = time.perf_counter()

    if pending:
        retriever = FactEntityGraphRetriever(config, artifacts)
        try:
            retriever.prepare_queries(pending)
            for query in tqdm(pending, desc=f"graph {method} retrieval"):
                try:
                    results, trace = retriever.retrieve(query, config.retrieval.top_k, method)
                    row = {
                        "query_id": query.query_id,
                        "question": query.question,
                        "method": method,
                        "results": retriever.serialize_results(results),
                        "trace": trace,
                    }
                    artifacts.append_jsonl(row, *result_parts)
                    rows.append(row)
                except Exception as exc:
                    error = {
                        "query_id": query.query_id,
                        "question": query.question,
                        "method": method,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    artifacts.append_jsonl(error, *error_parts)
                    errors.append(error)
        finally:
            retriever.close()

    metrics = _metrics(rows, queries, config.retrieval.top_k)
    report = {
        "method": method,
        "num_queries": len(queries),
        "num_successes": len(rows),
        "num_errors": len(errors),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "metrics": metrics,
    }
    artifacts.save_json_atomic(report, *metric_parts)
    artifacts.mark_stage_done(
        stage_name,
        {
            "num_queries": len(queries),
            "num_successes": len(rows),
            "num_errors": len(errors),
        },
    )
    artifacts.write_manifest(config)
    return report
