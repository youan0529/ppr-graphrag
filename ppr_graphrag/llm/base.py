"""LLM base interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False


class BaseLLM(Protocol):
    def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        """Generate a response from chat-completion messages."""
