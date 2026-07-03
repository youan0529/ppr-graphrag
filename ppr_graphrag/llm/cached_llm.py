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
            "provider": self.config.provider,
            "base_url": self.config.base_url,
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "seed": kwargs.get("seed"),
            "response_format": kwargs.get("response_format"),
            "kwargs": kwargs,
        }
        return make_id("llm", request)

    def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        key = self._key(messages, kwargs)
        cached = self.cache.get(key)
        if cached is not None:
            cached["cache_hit"] = True
            return LLMResponse(**cached)
        response = self.llm.generate(messages, **kwargs)
        payload = asdict(response)
        payload["cache_hit"] = False
        self.cache.set(key, payload, metadata={"model": self.config.model})
        return response
