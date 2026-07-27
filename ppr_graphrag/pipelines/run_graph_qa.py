"""Resumable reader QA over saved graph-retrieval results."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

from tqdm import tqdm

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.core.cache import SQLiteCache
from ppr_graphrag.core.config import AppConfig
from ppr_graphrag.data.schema import Document, Query
from ppr_graphrag.evaluation.qa_metrics import exact_match, token_f1
from ppr_graphrag.llm.cached_llm import CachedLLM
from ppr_graphrag.llm.openai_compatible import OpenAICompatibleLLM


def answer_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "short_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            },
        },
    }


def _parse_answer(text: str) -> str:
    answer = str(json.loads(text)["answer"]).strip()
    if not answer:
        raise ValueError("LLM answer cannot be empty")
    return answer


def _messages(prompt: str, query: Query, contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
    passage_text = "\n\n".join(
        f"[Passage {context['rank']}]\nTitle: {context['title']}\n{context['text']}" for context in contexts
    )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Question: {query.question}\n\n{passage_text}"},
    ]


def _answer_query(
    query: Query,
    retrieval: dict[str, Any],
    corpus: dict[str, Document],
    llm: CachedLLM,
    prompt: str,
    config: AppConfig,
) -> tuple[bool, dict[str, Any]]:
    response_text: str | None = None
    try:
        contexts = []
        for result in retrieval["results"][: config.qa.retrieval_top_k]:
            document = corpus[result["doc_id"]]
            contexts.append(
                {
                    "rank": result["rank"],
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "text": document.text,
                    "retrieval_score": result["score"],
                }
            )
        response = llm.generate(
            _messages(prompt, query, contexts),
            max_tokens=config.qa.max_tokens,
            reasoning_effort=config.llm.reasoning_effort,
            response_format=answer_response_format(),
        )
        response_text = response.text
        prediction = _parse_answer(response.text)
        return True, {
            "query_id": query.query_id,
            "question": query.question,
            "retrieval_method": retrieval["method"],
            "retrieval_top_k": config.qa.retrieval_top_k,
            "contexts": contexts,
            "gold_answers": query.answers,
            "predicted_answer": prediction,
            "exact_match": exact_match(prediction, query.answers),
            "f1": token_f1(prediction, query.answers),
            "response_text": response.text,
            "usage": response.usage,
            "cache_hit": response.cache_hit,
        }
    except Exception as exc:
        return False, {
            "query_id": query.query_id,
            "question": query.question,
            "retrieval_method": retrieval.get("method"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "response_text": response_text,
        }


def _report(
    method: str,
    queries: list[Query],
    answers: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    started_at: float,
    cache_stats: dict[str, Any],
    previous_elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    scored = [row for row in answers if row["gold_answers"]]
    return {
        "method": method,
        "num_queries": len(queries),
        "num_answers": len(answers),
        "num_missing_answers": len(queries) - len({row["query_id"] for row in answers}),
        "num_error_events": len(errors),
        "error_types": dict(Counter(row["error_type"] for row in errors)),
        "retrieval_top_k": answers[0]["retrieval_top_k"] if answers else None,
        "exact_match": sum(row["exact_match"] for row in scored) / len(scored) if scored else 0.0,
        "f1": sum(row["f1"] for row in scored) / len(scored) if scored else 0.0,
        "unknown_predictions": sum(row["predicted_answer"].lower() == "unknown" for row in answers),
        "elapsed_seconds": round(previous_elapsed_seconds + time.perf_counter() - started_at, 3),
        "cache": cache_stats,
    }


def run_graph_qa(
    config: AppConfig,
    corpus_documents: list[Document],
    queries: list[Query],
    artifacts: ArtifactManager,
    method: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    if method not in {"dense", "ppr"}:
        raise ValueError("method must be 'dense' or 'ppr'")
    if not config.llm.cache_enabled:
        raise ValueError("Graph QA requires llm.cache_enabled=true")

    retrieval_parts = ("artifacts", "graph_retrieval", f"{method}_results.jsonl")
    answer_parts = ("artifacts", "qa", f"{method}_answers.jsonl")
    error_parts = ("artifacts", "qa", f"{method}_errors.jsonl")
    metric_parts = ("metrics", f"{method}_qa_metrics.json")
    stage_name = f"graph_qa_{method}"
    if overwrite:
        for parts in (answer_parts, error_parts, metric_parts):
            artifacts.delete(*parts)
        artifacts.delete("artifacts", f"{stage_name}.done.json")

    previous_report = artifacts.load_json(*metric_parts) if artifacts.exists(*metric_parts) else {}
    previous_elapsed_seconds = float(previous_report.get("elapsed_seconds", 0.0))
    retrieval_by_query = {row["query_id"]: row for row in artifacts.load_jsonl(*retrieval_parts)}
    corpus = {document.doc_id: document for document in corpus_documents}
    answers = artifacts.load_jsonl(*answer_parts)
    errors = artifacts.load_jsonl(*error_parts)
    completed = {row["query_id"] for row in answers}
    pending = [query for query in queries if query.query_id not in completed]
    missing_retrieval = [query.query_id for query in pending if query.query_id not in retrieval_by_query]
    if missing_retrieval:
        raise ValueError(
            f"Missing saved {method} retrieval results for {len(missing_retrieval)} queries; "
            f"first IDs: {missing_retrieval[:5]}"
        )
    prompt = Path(config.qa.prompt_path).read_text(encoding="utf-8").strip()
    cache = SQLiteCache(config.llm.cache_path)
    base_urls = config.qa.base_urls or config.llm.base_urls or [config.llm.base_url]
    llms = []
    for base_url in base_urls:
        llm_config = replace(config.llm, base_url=base_url)
        llms.append(CachedLLM(OpenAICompatibleLLM(llm_config), cache, llm_config))
    started_at = time.perf_counter()

    try:
        workers = max(1, config.qa.workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            query_iter = iter(enumerate(pending))
            futures: set[Future[tuple[bool, dict[str, Any]]]] = set()

            def submit_next() -> bool:
                try:
                    index, query = next(query_iter)
                except StopIteration:
                    return False
                llm = llms[index % len(llms)]
                futures.add(
                    executor.submit(
                        _answer_query,
                        query,
                        retrieval_by_query[query.query_id],
                        corpus,
                        llm,
                        prompt,
                        config,
                    )
                )
                return True

            for _ in range(min(workers, len(pending))):
                submit_next()

            with tqdm(total=len(pending), desc=f"graph {method} QA") as progress:
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        succeeded, row = future.result()
                        if succeeded:
                            artifacts.append_jsonl(row, *answer_parts)
                            answers.append(row)
                        else:
                            artifacts.append_jsonl(row, *error_parts)
                            errors.append(row)
                        progress.update()
                        submit_next()
    finally:
        report = _report(
            method,
            queries,
            answers,
            errors,
            started_at,
            cache.stats(),
            previous_elapsed_seconds,
        )
        artifacts.save_json_atomic(report, *metric_parts)
        cache.close()

    if report["num_answers"] == len(queries):
        artifacts.mark_stage_done(stage_name, {"num_answers": len(answers), "num_error_events": len(errors)})
    artifacts.write_manifest(config)
    return report
