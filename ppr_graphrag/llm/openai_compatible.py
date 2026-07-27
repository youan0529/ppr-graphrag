"""OpenAI-compatible chat completion client."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ppr_graphrag.core.config import LLMConfig
from ppr_graphrag.llm.base import LLMResponse


class OpenAICompatibleLLM:
    def __init__(self, config: LLMConfig):
        try:
            from openai import DefaultHttpxClient, OpenAI
        except ImportError as exc:
            raise ImportError("Install openai to use OpenAICompatibleLLM: pip install openai") from exc
        self.config = config
        hostname = urlparse(config.base_url).hostname
        http_client = (
            DefaultHttpxClient(timeout=config.timeout, trust_env=False)
            if hostname in {"localhost", "127.0.0.1", "::1"}
            else None
        )
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            http_client=http_client,
        )

    def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        params = {
            "model": kwargs.pop("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
            **kwargs,
        }
        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]
        usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
        raw = response.model_dump() if hasattr(response, "model_dump") else response
        return LLMResponse(text=choice.message.content or "", raw=raw, usage=usage)
