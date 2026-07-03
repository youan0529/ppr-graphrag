from ppr_graphrag.core.hashing import make_id, sha256_text, stable_json_dumps


def test_stable_json_dumps_sorts_keys() -> None:
    assert stable_json_dumps({"b": 1, "a": [2, 1]}) == '{"a":[2,1],"b":1}'


def test_sha256_text_is_stable() -> None:
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abcd")


def test_make_id_handles_equivalent_dict_order() -> None:
    assert make_id("doc", {"a": 1, "b": 2}) == make_id("doc", {"b": 2, "a": 1})
    assert make_id("doc", {"a": 1}).startswith("doc-")
