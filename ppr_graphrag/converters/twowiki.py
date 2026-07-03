"""2WikiMultiHopQA converter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ppr_graphrag.converters.common import (
    make_doc_id,
    normalize_context,
    normalize_supporting_facts,
    read_json_or_jsonl,
    sentences_to_text,
    write_jsonl,
)


def convert_twowiki(
    input_path: str | Path,
    output_dir: str | Path,
    dataset_name: str = "twowiki",
    split: str | None = None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    examples = read_json_or_jsonl(input_path)
    if max_examples is not None:
        examples = examples[:max_examples]
    output = Path(output_dir)
    corpus: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    conflict_count = 0
    conflict_examples: list[dict[str, Any]] = []
    missing_support_titles: dict[str, list[str]] = {}

    for i, example in enumerate(examples):
        query_id = str(example.get("_id") or example.get("id") or f"{dataset_name}-{split or 'unknown'}-{i}")
        context = normalize_context(example.get("context"))
        title_to_doc_id: dict[str, str] = {}
        context_doc_ids: list[str] = []
        for paragraph in context:
            title = paragraph["title"]
            text = sentences_to_text(paragraph["sentences"])
            doc_id = make_doc_id(dataset_name, title)
            title_to_doc_id[title] = doc_id
            context_doc_ids.append(doc_id)
            record = {
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "metadata": {
                    "dataset": dataset_name,
                    "split": split,
                    "source": str(input_path),
                    "original_title": title,
                },
            }
            if doc_id in corpus and corpus[doc_id]["text"] != text:
                conflict_count += 1
                if len(conflict_examples) < 5:
                    conflict_examples.append({"doc_id": doc_id, "title": title, "query_id": query_id})
            else:
                corpus.setdefault(doc_id, record)

        supporting_facts = normalize_supporting_facts(example.get("supporting_facts"))
        supporting_doc_ids: list[str] = []
        missing: list[str] = []
        for fact in supporting_facts:
            doc_id = title_to_doc_id.get(fact["title"])
            if doc_id is None:
                missing.append(fact["title"])
            elif doc_id not in supporting_doc_ids:
                supporting_doc_ids.append(doc_id)
        if missing:
            missing_support_titles[query_id] = missing

        answer = example.get("answer")
        queries.append(
            {
                "query_id": query_id,
                "question": str(example.get("question", "")),
                "answers": [] if answer is None else [str(answer)],
                "supporting_doc_ids": supporting_doc_ids,
                "supporting_facts": supporting_facts,
                "metadata": {
                    "dataset": dataset_name,
                    "split": split,
                    "type": example.get("type"),
                    "level": example.get("level"),
                    "evidences": example.get("evidences"),
                    "evidences_id": example.get("evidences_id"),
                    "answer_id": example.get("answer_id"),
                    "entity_ids": example.get("entity_ids"),
                    "original_id": query_id,
                    "context_doc_ids": context_doc_ids,
                    "missing_support_titles": missing,
                },
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "corpus.jsonl", corpus.values())
    write_jsonl(output / "queries.jsonl", queries)
    report = {
        "dataset": dataset_name,
        "split": split,
        "input_path": str(input_path),
        "num_examples": len(examples),
        "num_documents": len(corpus),
        "num_queries": len(queries),
        "doc_conflicts": conflict_count,
        "doc_conflict_examples": conflict_examples,
        "missing_support_titles": missing_support_titles,
    }
    with (output / "conversion_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return report
