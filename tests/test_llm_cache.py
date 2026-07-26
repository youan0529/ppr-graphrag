from ppr_graphrag.core.cache import SQLiteCache
from ppr_graphrag.core.config import LLMConfig
from ppr_graphrag.llm.base import LLMResponse
from ppr_graphrag.llm.cached_llm import CachedLLM


class DummyLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        return LLMResponse(text=f"ok-{self.calls}", raw={"messages": messages}, usage={"tokens": 1})


def test_cached_llm_hits_cache(tmp_path) -> None:
    llm = DummyLLM()
    cached = CachedLLM(llm, SQLiteCache(tmp_path / "llm.sqlite"), LLMConfig())
    messages = [{"role": "user", "content": "hello"}]
    first = cached.generate(messages)
    second = cached.generate(messages)
    assert first.text == "ok-1"
    assert first.cache_hit is False
    assert second.text == "ok-1"
    assert second.cache_hit is True
    assert llm.calls == 1


def test_cached_llm_reuses_model_and_messages_across_generation_params(tmp_path) -> None:
    llm = DummyLLM()
    cached = CachedLLM(llm, SQLiteCache(tmp_path / "llm.sqlite"), LLMConfig())
    messages = [{"role": "user", "content": "hello"}]
    cached.generate(messages, temperature=0.0)
    cached.generate(messages, temperature=0.7)
    assert llm.calls == 1
