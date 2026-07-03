"""Retrieval metrics based on supporting_doc_ids."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from ppr_graphrag.data.schema import Query
from ppr_graphrag.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


def _eligible(queries: Iterable[Query]) -> list[Query]:
    valid = [q for q in queries if q.supporting_doc_ids]
    skipped = len(list(queries)) - len(valid) if not isinstance(queries, list) else len(queries) - len(valid)
    if skipped:
        logger.warning("Skipping %d queries without supporting_doc_ids", skipped)
    return valid


def recall_at_k(results_by_query: dict[str, list[RetrievalResult]], queries: list[Query], k: int) -> float:
    values: list[float] = []
    for query in _eligible(queries):
        gold = set(query.supporting_doc_ids)
        retrieved = {r.doc_id for r in results_by_query.get(query.query_id, [])[:k]}
        values.append(len(gold & retrieved) / len(gold))
    return sum(values) / len(values) if values else 0.0


def hit_at_k(results_by_query: dict[str, list[RetrievalResult]], queries: list[Query], k: int) -> float:
    values: list[float] = []
    for query in _eligible(queries):
        gold = set(query.supporting_doc_ids)
        retrieved = {r.doc_id for r in results_by_query.get(query.query_id, [])[:k]}
        values.append(1.0 if gold & retrieved else 0.0)
    return sum(values) / len(values) if values else 0.0


def mrr_at_k(results_by_query: dict[str, list[RetrievalResult]], queries: list[Query], k: int) -> float:
    values: list[float] = []
    for query in _eligible(queries):
        gold = set(query.supporting_doc_ids)
        rr = 0.0
        for result in results_by_query.get(query.query_id, [])[:k]:
            if result.doc_id in gold:
                rr = 1.0 / result.rank
                break
        values.append(rr)
    return sum(values) / len(values) if values else 0.0
