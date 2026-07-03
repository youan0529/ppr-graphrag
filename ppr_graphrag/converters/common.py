"""Shared helpers for dataset converters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ppr_graphrag.core.hashing import sha256_text, stable_json_dumps


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {p}:{line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object in {p}:{line_no}, got {type(obj).__name__}")
                records.append(obj)
        return records
    if p.suffix.lower() != ".json":
        raise ValueError(f"Unsupported input suffix for {p}; expected .json or .jsonl")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return _ensure_dict_records(data, str(p))
    if isinstance(data, dict):
        for key in ("data", "examples", "instances", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return _ensure_dict_records(value, f"{p}:{key}")
        list_values = [value for value in data.values() if isinstance(value, list)]
        if len(list_values) == 1:
            return _ensure_dict_records(list_values[0], str(p))
    raise ValueError(f"Cannot find a list of examples in {p}")


def _ensure_dict_records(records: list[Any], source: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Expected object in {source} at index {i}, got {type(record).__name__}")
        output.append(record)
    return output


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title or "").strip())


def make_doc_id(dataset: str, title: str, text: str | None = None, local_id: str | None = None) -> str:
    norm_title = normalize_title(title)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", norm_title.lower()).strip("_")[:64] or "doc"
    if local_id is not None:
        payload: Any = {"dataset": dataset, "local_id": str(local_id)}
    elif dataset.lower() == "musique":
        payload = {"dataset": dataset, "title": norm_title, "text": text or ""}
    else:
        payload = {"dataset": dataset, "title": norm_title}
    digest = sha256_text(stable_json_dumps(payload))[:12]
    return f"{dataset.lower()}-{slug}-{digest}"


def sentences_to_text(sentences: Any) -> str:
    if isinstance(sentences, str):
        return sentences
    if isinstance(sentences, list) and all(isinstance(sentence, str) for sentence in sentences):
        return " ".join(sentence.strip() for sentence in sentences if sentence.strip())
    raise ValueError(f"sentences must be str or list[str], got {type(sentences).__name__}: {sentences!r}")


def normalize_supporting_facts(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        facts: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                title = item.get("title")
                sent_id = item.get("sent_id", item.get("sent_idx"))
            elif isinstance(item, list | tuple) and len(item) >= 2:
                title, sent_id = item[0], item[1]
            else:
                raise ValueError(f"Unsupported supporting fact item: {item!r}")
            facts.append({"title": normalize_title(str(title)), "sent_id": sent_id})
        return facts
    if isinstance(raw, dict):
        titles = raw.get("title", raw.get("titles"))
        sent_ids = raw.get("sent_id", raw.get("sent_ids", raw.get("sent_idx", raw.get("sent_idxs"))))
        if not isinstance(titles, list) or not isinstance(sent_ids, list):
            raise ValueError(f"Dict supporting_facts requires list title and sent_id fields: {raw!r}")
        if len(titles) != len(sent_ids):
            raise ValueError(f"supporting_facts title and sent_id length mismatch: {raw!r}")
        return [{"title": normalize_title(str(title)), "sent_id": sent_id} for title, sent_id in zip(titles, sent_ids, strict=True)]
    raise ValueError(f"Unsupported supporting_facts type: {type(raw).__name__}")


def normalize_context(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        context: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                title = item.get("title", "")
                sentences = item.get("sentences", item.get("paragraph_text", ""))
            elif isinstance(item, list | tuple) and len(item) >= 2:
                title, sentences = item[0], item[1]
            else:
                raise ValueError(f"Unsupported context item: {item!r}")
            if isinstance(sentences, str):
                sentence_list = [sentences]
            elif isinstance(sentences, list) and all(isinstance(sentence, str) for sentence in sentences):
                sentence_list = sentences
            else:
                raise ValueError(f"context sentences must be str or list[str]: {item!r}")
            context.append({"title": normalize_title(str(title)), "sentences": sentence_list})
        return context
    if isinstance(raw, dict):
        titles = raw.get("title", raw.get("titles"))
        sentences = raw.get("sentences", raw.get("context"))
        if not isinstance(titles, list) or not isinstance(sentences, list):
            raise ValueError(f"Dict context requires list title and sentences fields: {raw!r}")
        if len(titles) != len(sentences):
            raise ValueError(f"context title and sentences length mismatch: {raw!r}")
        return normalize_context([[title, sent] for title, sent in zip(titles, sentences, strict=True)])
    raise ValueError(f"Unsupported context type: {type(raw).__name__}")
