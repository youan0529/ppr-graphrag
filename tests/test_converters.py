import json

from ppr_graphrag.converters.hotpotqa import convert_hotpotqa
from ppr_graphrag.converters.musique import convert_musique
from ppr_graphrag.converters.twowiki import convert_twowiki


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_convert_hotpotqa_official_like(tmp_path) -> None:
    input_path = tmp_path / "hotpot.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "_id": "h1",
                    "question": "Who won?",
                    "answer": "Alice",
                    "supporting_facts": [["Alpha", 0]],
                    "context": [["Alpha", ["Alice won."]], ["Beta", ["Bob lost."]]],
                    "type": "bridge",
                    "level": "easy",
                }
            ]
        ),
        encoding="utf-8",
    )
    report = convert_hotpotqa(input_path, tmp_path / "out", split="dev_distractor")
    corpus = _read_jsonl(tmp_path / "out" / "corpus.jsonl")
    queries = _read_jsonl(tmp_path / "out" / "queries.jsonl")
    assert report["num_documents"] == 2
    assert queries[0]["supporting_doc_ids"] == [corpus[0]["doc_id"]]
    assert queries[0]["metadata"]["type"] == "bridge"


def test_convert_hotpotqa_hf_like_and_max_examples(tmp_path) -> None:
    input_path = tmp_path / "hotpot_hf.json"
    input_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "h1",
                        "question": "Where?",
                        "answer": "Paris",
                        "supporting_facts": {"title": ["France"], "sent_id": [1]},
                        "context": {"title": ["France"], "sentences": [["Paris is in France.", "It is a city."]]},
                    },
                    {
                        "id": "h2",
                        "question": "Skipped?",
                        "answer": "Yes",
                        "supporting_facts": [],
                        "context": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    report = convert_hotpotqa(input_path, tmp_path / "out", max_examples=1)
    queries = _read_jsonl(tmp_path / "out" / "queries.jsonl")
    assert report["num_examples"] == 1
    assert queries[0]["answers"] == ["Paris"]
    assert len(queries[0]["supporting_doc_ids"]) == 1


def test_convert_twowiki_preserves_metadata(tmp_path) -> None:
    input_path = tmp_path / "twowiki.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "_id": "t1",
                    "question": "Q?",
                    "answer": "A",
                    "supporting_facts": [["Title", 0]],
                    "context": [["Title", ["Evidence sentence."]]],
                    "evidences": [["Title", "Evidence sentence."]],
                    "evidences_id": [1],
                    "answer_id": "a1",
                    "entity_ids": ["e1"],
                    "type": "comparison",
                    "level": "hard",
                }
            ]
        ),
        encoding="utf-8",
    )
    convert_twowiki(input_path, tmp_path / "out", split="dev")
    query = _read_jsonl(tmp_path / "out" / "queries.jsonl")[0]
    assert query["metadata"]["evidences_id"] == [1]
    assert query["metadata"]["answer_id"] == "a1"
    assert query["metadata"]["entity_ids"] == ["e1"]


def test_convert_musique_support_union_and_metadata(tmp_path) -> None:
    input_path = tmp_path / "musique.jsonl"
    sample = {
        "id": "m1",
        "question": "Q?",
        "answer": "A",
        "answer_aliases": ["Alias"],
        "answerable": True,
        "paragraphs": [
            {"idx": 0, "title": "T0", "paragraph_text": "Not support.", "is_supporting": False},
            {"idx": 1, "title": "T1", "paragraph_text": "Support one.", "is_supporting": True},
            {"idx": 2, "title": "T2", "paragraph_text": "Support two.", "is_supporting": False},
        ],
        "question_decomposition": [{"id": 1, "question": "sub?", "answer": "x", "paragraph_support_idx": 2}],
    }
    input_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    convert_musique(input_path, tmp_path / "out", split="dev")
    query = _read_jsonl(tmp_path / "out" / "queries.jsonl")[0]
    assert len(query["supporting_doc_ids"]) == 2
    assert [fact["paragraph_idx"] for fact in query["supporting_facts"]] == [1, 2]
    assert query["metadata"]["answer_aliases"] == ["Alias"]
    assert query["metadata"]["answerable"] is True
