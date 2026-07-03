"""Lightweight converters from raw HuggingFace JSONL to project JSONL."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {p}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title).strip())


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def make_title_doc_id(dataset: str, title: str) -> str:
    return f"{dataset}:{normalize_title(title)}"


def make_musique_doc_id(title: str, paragraph_text: str) -> str:
    key = normalize_title(title) + "\n" + normalize_text(paragraph_text)
    short_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return f"musique:{normalize_title(title)}:{short_hash}"


def unique_preserve_order(items: list[Any]) -> list[Any]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _check_outputs(output_dir: str | Path, overwrite: bool) -> tuple[Path, Path, Path, Path]:
    output = Path(output_dir)
    corpus_path = output / "corpus.jsonl"
    queries_path = output / "queries.jsonl"
    report_path = output / "conversion_report.json"
    preview_path = output / "conversion_preview.json"
    if not overwrite:
        existing = [str(path) for path in (corpus_path, queries_path, report_path, preview_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Output files already exist: {existing}. Re-run with --overwrite to replace them.")
    output.mkdir(parents=True, exist_ok=True)
    return corpus_path, queries_path, report_path, preview_path


def _id_rules(dataset: str) -> dict[str, str]:
    if dataset == "hotpotqa":
        return {
            "query_id": "raw['id']",
            "doc_id": "hotpotqa:<normalized_title>",
            "supporting_doc_ids": "supporting_facts.title -> hotpotqa:<normalized_title>",
        }
    if dataset == "twowiki":
        return {
            "query_id": "raw['id']",
            "doc_id": "twowiki:<normalized_title>",
            "supporting_doc_ids": "supporting_facts.title -> twowiki:<normalized_title>",
            "note": "evidences are preserved in query.metadata and are not used to construct supporting_doc_ids",
        }
    return {
        "query_id": "raw['id']",
        "doc_id": "musique:<normalized_title>:<md5(title + '\\n' + paragraph_text)[:12]>",
        "supporting_doc_ids": "question_decomposition paragraph_support_idx first, then is_supporting paragraphs",
        "note": "MuSiQue uses paragraph-level docs, not title-level docs",
    }


def _write_outputs(
    dataset: str,
    input_path: str | Path,
    output_dir: str | Path,
    corpus: dict[str, dict[str, Any]],
    queries: list[dict[str, Any]],
    num_raw_examples: int,
    doc_conflicts: list[dict[str, Any]],
    missing_support_docs: list[dict[str, Any]],
    preview: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    corpus_path, queries_path, report_path, preview_path = _check_outputs(output_dir, overwrite)
    write_jsonl(corpus_path, list(corpus.values()))
    write_jsonl(queries_path, queries)
    with preview_path.open("w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)
        f.write("\n")
    report = {
        "dataset": dataset,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "corpus_path": str(corpus_path),
        "queries_path": str(queries_path),
        "preview_path": str(preview_path),
        "num_raw_examples": num_raw_examples,
        "num_documents": len(corpus),
        "num_queries": len(queries),
        "num_doc_conflicts": len(doc_conflicts),
        "num_missing_support_docs": len(missing_support_docs),
        "doc_conflict_examples": doc_conflicts[:5],
        "missing_support_examples": missing_support_docs[:5],
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return report


def _add_doc(
    corpus: dict[str, dict[str, Any]],
    doc: dict[str, Any],
    query_id: str,
    doc_conflicts: list[dict[str, Any]],
) -> None:
    existing = corpus.get(doc["doc_id"])
    if existing is not None:
        if existing["text"] != doc["text"]:
            doc_conflicts.append({"query_id": query_id, "doc_id": doc["doc_id"], "title": doc["title"]})
        return
    corpus[doc["doc_id"]] = doc


def convert_hotpotqa(input_path: str | Path, output_dir: str | Path, overwrite: bool = False) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    corpus: dict[str, dict[str, Any]] = {}
    queries = []
    doc_conflicts: list[dict[str, Any]] = []
    missing_support_docs: list[dict[str, Any]] = []
    preview: dict[str, Any] | None = None

    for raw in rows:
        query_id = str(raw["id"])
        titles = raw["context"]["title"]
        sentence_groups = raw["context"]["sentences"]
        context_doc_ids = []
        title_to_doc_id = {}
        converted_docs = []

        for title, sentences in zip(titles, sentence_groups, strict=True):
            norm_title = normalize_title(title)
            doc_id = make_title_doc_id("hotpotqa", norm_title)
            title_to_doc_id[norm_title] = doc_id
            context_doc_ids.append(doc_id)
            doc = {
                "doc_id": doc_id,
                "title": norm_title,
                "text": normalize_text(" ".join(sentences)),
                "metadata": {
                    "dataset": "hotpotqa",
                    "source": "context",
                    "original_title": title,
                    "sentences": sentences,
                },
            }
            converted_docs.append(doc)
            _add_doc(corpus, doc, query_id, doc_conflicts)

        support_titles = raw.get("supporting_facts", {}).get("title", [])
        support_sent_ids = raw.get("supporting_facts", {}).get("sent_id", [])
        supporting_facts = [
            {"title": normalize_title(title), "sent_id": sent_id}
            for title, sent_id in zip(support_titles, support_sent_ids, strict=True)
        ]
        support_ids = []
        missing_titles = []
        for fact in supporting_facts:
            doc_id = title_to_doc_id.get(fact["title"])
            if doc_id is None:
                missing_titles.append(fact["title"])
            else:
                support_ids.append(doc_id)
        support_ids = unique_preserve_order(support_ids)
        if missing_titles:
            missing_support_docs.append({"query_id": query_id, "titles": missing_titles})

        query = {
            "query_id": query_id,
            "question": str(raw["question"]),
            "answers": [] if raw.get("answer") is None else [str(raw["answer"])],
            "supporting_doc_ids": support_ids,
            "supporting_facts": supporting_facts,
            "metadata": {
                "dataset": "hotpotqa",
                "type": raw.get("type"),
                "level": raw.get("level"),
                "context_doc_ids": context_doc_ids,
                "missing_support_titles": missing_titles,
            },
        }
        queries.append(query)

        if preview is None:
            preview = {
                "dataset": "hotpotqa",
                "input_path": str(input_path),
                "id_rules": _id_rules("hotpotqa"),
                "raw_example_brief": {
                    "id": raw["id"],
                    "question": raw["question"],
                    "answer": raw.get("answer"),
                    "type": raw.get("type"),
                    "level": raw.get("level"),
                    "supporting_titles": list(support_titles),
                },
                "converted_query": query,
                "converted_corpus_docs": converted_docs,
            }

    preview = preview or {
        "dataset": "hotpotqa",
        "input_path": str(input_path),
        "id_rules": _id_rules("hotpotqa"),
        "raw_example_brief": {},
        "converted_query": {},
        "converted_corpus_docs": [],
    }
    return _write_outputs(
        "hotpotqa",
        input_path,
        output_dir,
        corpus,
        queries,
        len(rows),
        doc_conflicts,
        missing_support_docs,
        preview,
        overwrite,
    )


def convert_twowiki(input_path: str | Path, output_dir: str | Path, overwrite: bool = False) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    corpus: dict[str, dict[str, Any]] = {}
    queries = []
    doc_conflicts: list[dict[str, Any]] = []
    missing_support_docs: list[dict[str, Any]] = []
    preview: dict[str, Any] | None = None

    for raw in rows:
        query_id = str(raw["id"])
        titles = raw["context"]["title"]
        sentence_groups = raw["context"]["sentences"]
        context_doc_ids = []
        title_to_doc_id = {}
        converted_docs = []

        for title, sentences in zip(titles, sentence_groups, strict=True):
            norm_title = normalize_title(title)
            doc_id = make_title_doc_id("twowiki", norm_title)
            title_to_doc_id[norm_title] = doc_id
            context_doc_ids.append(doc_id)
            doc = {
                "doc_id": doc_id,
                "title": norm_title,
                "text": normalize_text(" ".join(sentences)),
                "metadata": {
                    "dataset": "twowiki",
                    "source": "context",
                    "original_title": title,
                    "sentences": sentences,
                },
            }
            converted_docs.append(doc)
            _add_doc(corpus, doc, query_id, doc_conflicts)

        support_titles = raw.get("supporting_facts", {}).get("title", [])
        support_sent_ids = raw.get("supporting_facts", {}).get("sent_id", [])
        supporting_facts = [
            {"title": normalize_title(title), "sent_id": sent_id}
            for title, sent_id in zip(support_titles, support_sent_ids, strict=True)
        ]
        support_ids = []
        missing_titles = []
        for fact in supporting_facts:
            doc_id = title_to_doc_id.get(fact["title"])
            if doc_id is None:
                missing_titles.append(fact["title"])
            else:
                support_ids.append(doc_id)
        support_ids = unique_preserve_order(support_ids)
        if missing_titles:
            missing_support_docs.append({"query_id": query_id, "titles": missing_titles})

        query = {
            "query_id": query_id,
            "question": str(raw["question"]),
            "answers": [] if raw.get("answer") is None else [str(raw["answer"])],
            "supporting_doc_ids": support_ids,
            "supporting_facts": supporting_facts,
            "metadata": {
                "dataset": "twowiki",
                "type": raw.get("type"),
                "evidences": raw.get("evidences", []),
                "context_doc_ids": context_doc_ids,
                "missing_support_titles": missing_titles,
            },
        }
        queries.append(query)

        if preview is None:
            preview = {
                "dataset": "twowiki",
                "input_path": str(input_path),
                "id_rules": _id_rules("twowiki"),
                "raw_example_brief": {
                    "id": raw["id"],
                    "question": raw["question"],
                    "answer": raw.get("answer"),
                    "type": raw.get("type"),
                    "supporting_titles": list(support_titles),
                    "evidences": raw.get("evidences", []),
                },
                "converted_query": query,
                "converted_corpus_docs": converted_docs,
            }

    preview = preview or {
        "dataset": "twowiki",
        "input_path": str(input_path),
        "id_rules": _id_rules("twowiki"),
        "raw_example_brief": {},
        "converted_query": {},
        "converted_corpus_docs": [],
    }
    return _write_outputs(
        "twowiki",
        input_path,
        output_dir,
        corpus,
        queries,
        len(rows),
        doc_conflicts,
        missing_support_docs,
        preview,
        overwrite,
    )


def convert_musique(input_path: str | Path, output_dir: str | Path, overwrite: bool = False) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    corpus: dict[str, dict[str, Any]] = {}
    queries = []
    doc_conflicts: list[dict[str, Any]] = []
    missing_support_docs: list[dict[str, Any]] = []
    preview: dict[str, Any] | None = None

    for raw in rows:
        query_id = str(raw["id"])
        idx_to_doc_id = {}
        idx_to_paragraph = {}
        context_doc_ids = []
        converted_docs = []

        for paragraph in raw["paragraphs"]:
            idx = paragraph["idx"]
            title = normalize_title(paragraph["title"])
            text = normalize_text(paragraph["paragraph_text"])
            doc_id = make_musique_doc_id(title, text)
            idx_to_doc_id[idx] = doc_id
            idx_to_paragraph[idx] = paragraph
            context_doc_ids.append(doc_id)
            doc = {
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "metadata": {
                    "dataset": "musique",
                    "source": "paragraphs",
                    "original_query_id": query_id,
                    "paragraph_idx": idx,
                    "is_supporting": paragraph.get("is_supporting", False),
                },
            }
            converted_docs.append(doc)
            _add_doc(corpus, doc, query_id, doc_conflicts)

        support_idxs = []
        supporting_facts = []
        for step in raw.get("question_decomposition", []):
            idx = step.get("paragraph_support_idx")
            if idx is None:
                continue
            support_idxs.append(idx)
            paragraph = idx_to_paragraph[idx]
            supporting_facts.append(
                {
                    "paragraph_idx": idx,
                    "title": normalize_title(paragraph["title"]),
                    "decomposition_id": step.get("id"),
                    "decomposition_question": step.get("question"),
                    "decomposition_answer": step.get("answer"),
                }
            )

        for paragraph in raw["paragraphs"]:
            idx = paragraph["idx"]
            if paragraph.get("is_supporting", False) and idx not in support_idxs:
                support_idxs.append(idx)
                supporting_facts.append({"paragraph_idx": idx, "title": normalize_title(paragraph["title"])})

        support_idxs = unique_preserve_order(support_idxs)
        support_ids = [idx_to_doc_id[idx] for idx in support_idxs if idx in idx_to_doc_id]
        missing_idxs = [idx for idx in support_idxs if idx not in idx_to_doc_id]
        if missing_idxs:
            missing_support_docs.append({"query_id": query_id, "paragraph_idxs": missing_idxs})

        query = {
            "query_id": query_id,
            "question": str(raw["question"]),
            "answers": [] if raw.get("answer") is None else [str(raw["answer"])],
            "supporting_doc_ids": support_ids,
            "supporting_facts": supporting_facts,
            "metadata": {
                "dataset": "musique",
                "answerable": raw.get("answerable"),
                "answer_aliases": raw.get("answer_aliases", []),
                "question_decomposition": raw.get("question_decomposition", []),
                "context_doc_ids": context_doc_ids,
            },
        }
        queries.append(query)

        if preview is None:
            preview = {
                "dataset": "musique",
                "input_path": str(input_path),
                "id_rules": _id_rules("musique"),
                "raw_example_brief": {
                    "id": raw["id"],
                    "question": raw["question"],
                    "answer": raw.get("answer"),
                    "answerable": raw.get("answerable"),
                    "supporting_paragraph_idxs": support_idxs,
                    "question_decomposition": raw.get("question_decomposition", []),
                },
                "converted_query": query,
                "converted_corpus_docs": converted_docs,
            }

    preview = preview or {
        "dataset": "musique",
        "input_path": str(input_path),
        "id_rules": _id_rules("musique"),
        "raw_example_brief": {},
        "converted_query": {},
        "converted_corpus_docs": [],
    }
    return _write_outputs(
        "musique",
        input_path,
        output_dir,
        corpus,
        queries,
        len(rows),
        doc_conflicts,
        missing_support_docs,
        preview,
        overwrite,
    )


def convert_dataset(dataset: str, input_path: str | Path, output_dir: str | Path, overwrite: bool = False) -> dict[str, Any]:
    if dataset == "hotpotqa":
        return convert_hotpotqa(input_path, output_dir, overwrite)
    if dataset == "twowiki":
        return convert_twowiki(input_path, output_dir, overwrite)
    if dataset == "musique":
        return convert_musique(input_path, output_dir, overwrite)
    raise ValueError(f"Unknown dataset: {dataset}. Valid choices: hotpotqa, twowiki, musique")
