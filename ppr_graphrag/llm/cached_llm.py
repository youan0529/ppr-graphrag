"""Cache wrapper for LLM clients."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ppr_graphrag.core.cache import SQLiteCache
from ppr_graphrag.core.config import LLMConfig
from ppr_graphrag.core.hashing import make_id
from ppr_graphrag.llm.base import BaseLLM, LLMResponse


class CachedLLM:
    def __init__(self, llm: BaseLLM, cache: SQLiteCache, config: LLMConfig):
        self.llm = llm
        self.cache = cache
        self.config = config

    def _key(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
        request = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
        }
        return make_id("llm", request)

    def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        key = self._key(messages, kwargs)
        cached = self.cache.get(key)
        if cached is not None:
            payload = dict(cached)
            payload.pop("cache_hit", None)
            return LLMResponse(**payload, cache_hit=True)
        response = self.llm.generate(messages, **kwargs)
        payload = asdict(response)
        payload.pop("cache_hit", None)
        if response.text.strip():
            self.cache.set(key, payload, metadata={"model": self.config.model})
        response.cache_hit = False
        return response
