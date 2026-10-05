"""Source-grounded QA, separate chain/coverage metrics, and readable case artifacts."""
import json
from pathlib import Path
from ppr_graphrag.evaluation.qa_metrics import exact_match, token_f1
from ppr_graphrag.llm.chain_controller import prompt, reader_chain
from ppr_graphrag.llm.chain_schemas import READER
from ppr_graphrag.retrieval.chain_display import reader_text
from ppr_graphrag.retrieval.source_context import add_context
from ppr_graphrag.llm.structured_runtime import dump


def read_answer(result, graph, models, config):
    if result.get("reader_executed"):
        return result
    add_context(result,graph,models,config,prompt("reader"))
    docs = [graph["sources"][d] for d in result["supporting_doc_ids"]]
    reader = config["reader"]
    result["candidate_complete"] = result["evidence_complete"]
    result["evidence_complete"] = False
    result["chain_verified"] = False
    result["reader_executed"] = False
    result["retrieval_stop_reason"] = result["stop_reason"]
    data = reader_text(result['original_question'], result['selected_paths'], docs)
    messages = [
        {"role": "system", "content": prompt("reader")},
        {"role": "user", "content": data},
    ]
    feasible = len(docs) <= reader["top_k"] and (
        models.token_count(messages) + reader["output_tokens"] <= config["llm"]["context"]
    )
    result["context_budget_feasible"] = feasible
    if not feasible or not docs:
        result["stop_reason"] = "context_budget_exceeded" if not feasible else "no_candidate_evidence"
        result["reader_input"] = None
        return result

    def validate(obj):
        if obj.get("chain_verdict") not in ("valid", "corrected", "invalid", "insufficient"):
            raise ValueError("Invalid chain verification verdict")
        expected = obj["chain_verdict"] in ("valid", "corrected")
        if obj.get("evidence_sufficient") is not expected:
            raise ValueError("Verdict and evidence_sufficient disagree")
        if not isinstance(obj.get("answer"), str) or not obj["answer"].strip() or not obj.get("reason"):
            raise ValueError("Reader requires answer and brief verification reason")
        if not expected and obj["answer"] != "unknown":
            raise ValueError("Unsupported evidence must yield unknown")
        refs = obj.get("reopen_chain_ids")
        valid_ids = {p["id"] for p in result["selected_paths"]}
        if not isinstance(refs, list) or any(not isinstance(x, str) or x not in valid_ids for x in refs):
            raise ValueError("Unknown chain in reopen_chain_ids")
        if obj["chain_verdict"] == "valid" and refs:
            raise ValueError("Valid evidence cannot require reopening a chain")
        return obj

    try:
        answer = models.call("reader", prompt("reader"), data, validate, reader["output_tokens"], schema=READER)
    except ValueError as exc:
        # A failed answer must not erase successful retrieval or masquerade as evidence.
        result.update(reader_input=data,reader_error=str(exc),reader_executed=True,
            reader_output=dict(chain_verdict="insufficient",evidence_sufficient=False,
                answer="unknown",reason="Reader output invalid after bounded retries",reopen_chain_ids=[]),
            stop_reason="reader_output_error")
        return result
    result.update(
        reader_input=data,
        reader_output=answer,
        reader_executed=True,
        evidence_complete=answer["evidence_sufficient"],
        chain_verified=answer["chain_verdict"] == "valid",
        stop_reason="verified_qa" if answer["evidence_sufficient"] else "final_verification_failed",
    )
    return result


def evaluate(selected_queries, results, out):
    by_id = {r["query_id"]: r for r in results}
    scores = []
    for q in selected_queries:
        r = by_id.get(q["query_id"], {})
        gold = set(q.get("supporting_doc_ids", []))
        retrieved = set(r.get("supporting_doc_ids", []))
        complete = bool(gold) and gold <= retrieved
        feasible = r.get("context_budget_feasible", False)
        prediction = r.get("reader_output", {}).get("answer", "")
        scores.append(
            {
                "query_id": q["query_id"],
                "finished": bool(r),
                "gold_doc_ids": sorted(gold),
                "retrieved_doc_ids": sorted(retrieved),
                "recall": len(gold & retrieved) / len(gold) if gold else None,
                "complete_evidence": complete,
                "budget_feasible_complete_evidence": complete and feasible,
                "checker_sufficient": r.get("candidate_complete", False),
                "chain_verified": r.get("chain_verified", False),
                "chain_corrected": r.get("reader_output", {}).get("chain_verdict") == "corrected",
                "final_evidence_sufficient": r.get("evidence_complete", False),
                "context_budget_feasible": feasible,
                "reader_executed": r.get("reader_executed", False),
                "em": exact_match(prediction, q.get("answers", [])),
                "f1": token_f1(prediction, q.get("answers", [])),
                "stop_reason": r.get("stop_reason", "not_finished"),
            }
        )

    def mean(key):
        values = [x[key] for x in scores if x[key] is not None]
        return sum(values) / len(values) if values else None

    metrics = {
        "num_selected_queries": len(scores),
        "num_finished": sum(x["finished"] for x in scores),
        **{
            k: mean(k)
            for k in (
                "recall",
                "complete_evidence",
                "budget_feasible_complete_evidence",
                "checker_sufficient",
                "chain_verified",
                "chain_corrected",
                "final_evidence_sufficient",
                "em",
                "f1",
            )
        },
        "reader_executed": sum(x["reader_executed"] for x in scores),
        "per_query": scores,
        "note": "All selected queries in denominator. Small corpus runs are pipeline diagnostics only.",
    }
    dump(Path(out) / "metrics.json", metrics)
    return metrics


def write_case(result, out):
    lines = [
        f"# {result['query_id']}",
        "",
        result["original_question"],
        "",
        f"Stop: {result['stop_reason']}; candidate complete: {result.get('candidate_complete')}; source verified: {result.get('chain_verified')}; "
        f"context feasible: {result.get('context_budget_feasible')}",
        "",
    ]
    for chain in result["selected_paths"]:
        lines += [f"## Chain {chain['id']}", ""]
        for f in chain["evidence"]["facts"]:
            lines += [f"- {' → '.join(f['directed_triple'])}", f"  Original: {f['source_triple']}; qualifiers: {f['qualifiers']}", f"  Identity before: {f['identity_before']}", f"  Source: {f['doc_id']}: {f['quote']}"]
        lines += [
            "",
            "Identity dependencies: " + json.dumps(chain["evidence"]["identity_support"], ensure_ascii=False),
            "",
        ]
    lines += [
        "Documents: " + ", ".join(result["supporting_doc_ids"]),
        "",
        "Reader: " + json.dumps(result.get("reader_output"), ensure_ascii=False),
        "",
        "Each round JSON preserves paths, dispositions, epsilon reasons and exact sampling probabilities.",
    ]
    path = Path(out) / "queries" / result["query_id"] / "case.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
