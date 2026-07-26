"""Passage-local fact extraction with cached LLM calls and JSONL resume."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

from tqdm import tqdm

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.core.cache import SQLiteCache
from ppr_graphrag.core.config import AppConfig
from ppr_graphrag.data.schema import Document
from ppr_graphrag.graph.schema import TYPE_TO_KIND, extraction_response_format
from ppr_graphrag.llm.cached_llm import CachedLLM
from ppr_graphrag.llm.openai_compatible import OpenAICompatibleLLM


EXTRACTION_PARTS = ("artifacts", "extraction")


def parse_extraction(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    facts = payload["facts"]
    if not isinstance(facts, list):
        raise ValueError("LLM output field 'facts' must be a list")
    parsed = []
    for fact in facts:
        fact_text = str(fact["text"]).strip()
        relation = str(fact["relation"]).strip()
        mentions = fact["mentions"]
        if not fact_text or not relation or not isinstance(mentions, list):
            raise ValueError("Each fact requires non-empty text, relation, and a mentions list")
        clean_mentions = []
        for mention in mentions:
            name = str(mention["name"]).strip()
            entity_type = str(mention["type"]).strip()
            if not name:
                raise ValueError("Mention name cannot be empty")
            kind = TYPE_TO_KIND.get(entity_type)
            if kind is None:
                raise ValueError(f"Unknown mention type: {entity_type}")
            clean_mentions.append({"name": name, "kind": kind, "type": entity_type})
        parsed.append({"text": fact_text, "relation": relation, "mentions": clean_mentions})
    return parsed


def extraction_messages(prompt: str, document: Document) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Title: {document.title}\n\nPassage:\n{document.text}"},
    ]


def _extract_document(
    document: Document,
    llm: CachedLLM,
    prompt: str,
    reasoning_effort: str,
) -> tuple[bool, dict[str, Any]]:
    response_text: str | None = None
    try:
        response = llm.generate(
            extraction_messages(prompt, document),
            reasoning_effort=reasoning_effort,
            response_format=extraction_response_format(),
        )
        response_text = response.text
        return True, {
            "passage_id": document.doc_id,
            "title": document.title,
            "facts": parse_extraction(response.text),
            "response_text": response.text,
            "usage": response.usage,
            "cache_hit": response.cache_hit,
        }
    except Exception as exc:
        return False, {
            "passage_id": document.doc_id,
            "title": document.title,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "response_text": response_text,
        }


def _write_report(
    artifacts: ArtifactManager,
    corpus_size: int,
    successes: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    started_at: float,
    cache_stats: dict[str, Any],
) -> dict[str, Any]:
    completed_ids = {row["passage_id"] for row in successes} | {row["passage_id"] for row in errors}
    report = {
        "num_passages": corpus_size,
        "num_successes": len(successes),
        "num_errors": len(errors),
        "num_completed": len(completed_ids),
        "coverage": len(completed_ids) / corpus_size if corpus_size else 1.0,
        "error_types": dict(Counter(row["error_type"] for row in errors)),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "cache": cache_stats,
    }
    artifacts.save_json_atomic(report, *EXTRACTION_PARTS, "extraction_report.json")
    return report


def extract_facts(
    config: AppConfig,
    corpus: list[Document],
    artifacts: ArtifactManager,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not config.llm.cache_enabled:
        raise ValueError("Fact extraction requires llm.cache_enabled=true")
    success_parts = (*EXTRACTION_PARTS, "raw_extractions.jsonl")
    error_parts = (*EXTRACTION_PARTS, "extraction_errors.jsonl")
    report_parts = (*EXTRACTION_PARTS, "extraction_report.json")
    if overwrite:
        artifacts.delete(*success_parts)
        artifacts.delete(*error_parts)
        artifacts.delete(*report_parts)

    successes = artifacts.load_jsonl(*success_parts)
    errors = artifacts.load_jsonl(*error_parts)
    completed = {row["passage_id"] for row in successes} | {row["passage_id"] for row in errors}
    prompt = Path(config.graph.extraction_prompt_path).read_text(encoding="utf-8").strip()
    cache = SQLiteCache(config.llm.cache_path)
    base_urls = config.llm.base_urls or [config.llm.base_url]
    llms = []
    for base_url in base_urls:
        llm_config = replace(config.llm, base_url=base_url)
        llms.append(CachedLLM(OpenAICompatibleLLM(llm_config), cache, llm_config))
    started_at = time.perf_counter()

    try:
        pending = [document for document in corpus if document.doc_id not in completed]
        workers = max(1, config.graph.extraction_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: list[Future[tuple[bool, dict[str, Any]]]] = []
            for index, document in enumerate(pending):
                llm = llms[index % len(llms)]
                futures.append(
                    executor.submit(
                        _extract_document,
                        document,
                        llm,
                        prompt,
                        config.llm.reasoning_effort,
                    )
                )
            for index, future in enumerate(
                tqdm(as_completed(futures), total=len(futures), desc="extracting facts"),
                start=1,
            ):
                succeeded, row = future.result()
                if succeeded:
                    artifacts.append_jsonl(row, *success_parts)
                    successes.append(row)
                else:
                    artifacts.append_jsonl(row, *error_parts)
                    errors.append(row)
                if index % 100 == 0:
                    _write_report(artifacts, len(corpus), successes, errors, started_at, cache.stats())
    finally:
        report = _write_report(artifacts, len(corpus), successes, errors, started_at, cache.stats())
        cache.close()

    artifacts.write_manifest(config)
    return report
