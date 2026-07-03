"""Dataclasses for corpus documents and retrieval queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    title: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    query_id: str
    question: str
    answers: list[str] = field(default_factory=list)
    supporting_doc_ids: list[str] = field(default_factory=list)
    supporting_facts: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
