"""JSONL loaders for corpus and query files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ppr_graphrag.data.schema import Document, Query


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {p}:{line_no}: {exc}") from exc
    return rows


def _ensure_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str | int | float | bool):
        return [value]
    raise ValueError(f"{field_name} must be a list, scalar, or null; got {type(value).__name__}: {value!r}")


def load_corpus(path: str | Path) -> list[Document]:
    docs = []
    for row in _read_jsonl(path):
        if not row.get("doc_id") or not row.get("text"):
            raise ValueError(f"Corpus rows require doc_id and text: {row}")
        docs.append(
            Document(
                doc_id=str(row["doc_id"]),
                title=str(row.get("title") or ""),
                text=str(row["text"]),
                metadata=row.get("metadata") or {},
            )
        )
    return docs


def load_queries(path: str | Path) -> list[Query]:
    queries = []
    for row in _read_jsonl(path):
        if not row.get("query_id") or not row.get("question"):
            raise ValueError(f"Query rows require query_id and question: {row}")
        queries.append(
            Query(
                query_id=str(row["query_id"]),
                question=str(row["question"]),
                answers=[str(v) for v in _ensure_list(row.get("answers"), "answers")],
                supporting_doc_ids=[str(v) for v in _ensure_list(row.get("supporting_doc_ids"), "supporting_doc_ids")],
                supporting_facts=_ensure_list(row.get("supporting_facts"), "supporting_facts"),
                metadata=row.get("metadata") or {},
            )
        )
    return queries
