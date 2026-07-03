from ppr_graphrag.core.artifacts import ArtifactManager


def test_artifact_json_jsonl_pickle_and_stage(tmp_path) -> None:
    manager = ArtifactManager(tmp_path / "run")
    manager.save_json_atomic({"x": 1}, "artifacts", "x.json")
    assert manager.load_json("artifacts", "x.json") == {"x": 1}
    manager.delete("artifacts", "x.json")
    assert not manager.exists("artifacts", "x.json")
    manager.delete("artifacts", "x.json")
    manager.append_jsonl({"a": 1}, "artifacts", "rows.jsonl")
    manager.append_jsonl({"a": 2}, "artifacts", "rows.jsonl")
    assert manager.load_jsonl("artifacts", "rows.jsonl") == [{"a": 1}, {"a": 2}]
    manager.save_pickle_atomic({"p": [1, 2]}, "artifacts", "p.pkl")
    assert manager.load_pickle("artifacts", "p.pkl") == {"p": [1, 2]}
    assert not manager.is_stage_done("index")
    manager.mark_stage_done("index", {"n": 1})
    assert manager.is_stage_done("index")
