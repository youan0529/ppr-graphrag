import json

from ppr_graphrag.data.jsonl_loader import load_corpus, load_queries


def test_load_corpus_and_queries(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    corpus_path.write_text(
        json.dumps({"doc_id": "d1", "title": "T", "text": "hello", "metadata": {"x": 1}}) + "\n",
        encoding="utf-8",
    )
    queries_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "question": "hello?",
                "answers": ["hello"],
                "supporting_doc_ids": ["d1"],
                "supporting_facts": [],
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    docs = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    assert docs[0].doc_id == "d1"
    assert docs[0].metadata["x"] == 1
    assert queries[0].supporting_doc_ids == ["d1"]
