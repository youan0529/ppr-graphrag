#!/usr/bin/env python
"""Create a tiny corpus and query set for smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    corpus = [
        {"doc_id": "doc-1", "title": "PageRank", "text": "PageRank ranks web pages using links and random walks.", "metadata": {}},
        {"doc_id": "doc-2", "title": "Personalized PageRank", "text": "Personalized PageRank biases random walks toward seed nodes.", "metadata": {}},
        {"doc_id": "doc-3", "title": "BM25", "text": "BM25 is a lexical ranking function for search engines.", "metadata": {}},
        {"doc_id": "doc-4", "title": "Dense retrieval", "text": "Dense retrieval compares learned embeddings with vector similarity.", "metadata": {}},
        {"doc_id": "doc-5", "title": "GraphRAG", "text": "GraphRAG combines graph structure with retrieval augmented generation.", "metadata": {}},
    ]
    queries = [
        {
            "query_id": "q-1",
            "question": "What method biases random walks toward seed nodes?",
            "answers": ["Personalized PageRank"],
            "supporting_doc_ids": ["doc-2"],
            "supporting_facts": [],
            "metadata": {},
        },
        {
            "query_id": "q-2",
            "question": "What retrieval method compares embeddings using vector similarity?",
            "answers": ["Dense retrieval"],
            "supporting_doc_ids": ["doc-4"],
            "supporting_facts": [],
            "metadata": {},
        },
    ]
    write_jsonl(Path("data/toy/corpus.jsonl"), corpus)
    write_jsonl(Path("data/toy/queries.jsonl"), queries)
    print("Wrote data/toy/corpus.jsonl and data/toy/queries.jsonl")


if __name__ == "__main__":
    main()
