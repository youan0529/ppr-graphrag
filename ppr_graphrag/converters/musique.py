"""MuSiQue converter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ppr_graphrag.converters.common import make_doc_id, normalize_title, read_json_or_jsonl, write_jsonl


def convert_musique(
    input_path: str | Path,
    output_dir: str | Path,
    dataset_name: str = "musique",
    split: str | None = None,
    max_examples: int | None = None,
    answerable_only: bool = False,
) -> dict[str, Any]:
    examples = read_json_or_jsonl(input_path)
    if answerable_only:
        examples = [example for example in examples if example.get("answerable") is not False]
    if max_examples is not None:
        examples = examples[:max_examples]
    output = Path(output_dir)
    corpus: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    conflict_count = 0
    conflict_examples: list[dict[str, Any]] = []

    for i, example in enumerate(examples):
        query_id = str(example.get("id") or example.get("_id") or f"{dataset_name}-{split or 'unknown'}-{i}")
        paragraphs = example.get("paragraphs") or []
        if not isinstance(paragraphs, list):
            raise ValueError(f"MuSiQue paragraphs must be a list for query {query_id}")
        idx_to_doc_id: dict[Any, str] = {}
        idx_to_title: dict[Any, str] = {}
        context_doc_ids: list[str] = []
        supporting_idx: list[Any] = []

        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                raise ValueError(f"MuSiQue paragraph must be object for query {query_id}: {paragraph!r}")
            idx = paragraph.get("idx")
            title = normalize_title(str(paragraph.get("title", "")))
            text = str(paragraph.get("paragraph_text", ""))
            doc_id = make_doc_id(dataset_name, title, text=text)
            idx_to_doc_id[idx] = doc_id
            idx_to_title[idx] = title
            context_doc_ids.append(doc_id)
            if paragraph.get("is_supporting") is True and idx not in supporting_idx:
                supporting_idx.append(idx)
            record = {
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "metadata": {
                    "dataset": dataset_name,
                    "split": split,
                    "source": str(input_path),
                    "original_query_id": query_id,
                    "paragraph_idx": idx,
                    "is_supporting": paragraph.get("is_supporting"),
                },
            }
            if doc_id in corpus and corpus[doc_id]["text"] != text:
                conflict_count += 1
                if len(conflict_examples) < 5:
                    conflict_examples.append({"doc_id": doc_id, "title": title, "query_id": query_id})
            else:
                corpus.setdefault(doc_id, record)

        question_decomposition = example.get("question_decomposition") or []
        if not isinstance(question_decomposition, list):
            raise ValueError(f"question_decomposition must be a list for query {query_id}")
        for step in question_decomposition:
            if not isinstance(step, dict):
                continue
            idx = step.get("paragraph_support_idx")
            if idx is not None and idx not in supporting_idx:
                supporting_idx.append(idx)

        supporting_doc_ids = [idx_to_doc_id[idx] for idx in supporting_idx if idx in idx_to_doc_id]
        supporting_facts = [{"paragraph_idx": idx, "title": idx_to_title.get(idx, "")} for idx in supporting_idx if idx in idx_to_doc_id]
        answers = []
        if example.get("answer") is not None:
            answers.append(str(example["answer"]))

        queries.append(
            {
                "query_id": query_id,
                "question": str(example.get("question", "")),
                "answers": answers,
                "supporting_doc_ids": supporting_doc_ids,
                "supporting_facts": supporting_facts,
                "metadata": {
                    "dataset": dataset_name,
                    "split": split,
                    "original_id": query_id,
                    "answerable": example.get("answerable"),
                    "answer_aliases": example.get("answer_aliases") or [],
                    "question_decomposition": question_decomposition,
                    "context_doc_ids": context_doc_ids,
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
        "answerable_only": answerable_only,
    }
    with (output / "conversion_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return report
