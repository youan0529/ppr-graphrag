"""Fact extraction and graph schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REFERENT_TYPES = (
    "person",
    "organization",
    "place",
    "event",
    "creative_work",
    "product",
    "object",
    "law",
    "award",
    "other_referent",
)
CATEGORY_TYPES = (
    "occupation",
    "nationality",
    "religion",
    "language",
    "field",
    "genre",
    "medical_condition",
    "species",
    "concept",
)
LITERAL_TYPES = (
    "date",
    "time",
    "number",
    "ordinal",
    "percentage",
    "money",
    "quantity",
    "other_literal",
)
TYPE_TO_KIND = {
    **{name: "referent" for name in REFERENT_TYPES},
    **{name: "category" for name in CATEGORY_TYPES},
    **{name: "literal" for name in LITERAL_TYPES},
    "unknown": "uncertain",
}
GRAPH_KINDS = {"referent", "category"}


def extraction_response_format() -> dict[str, Any]:
    """Return the strict JSON schema sent to OpenAI-compatible models."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "passage_fact_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["facts"],
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "relation", "mentions"],
                            "properties": {
                                "text": {"type": "string"},
                                "relation": {"type": "string"},
                                "mentions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["name", "kind", "type"],
                                        "properties": {
                                            "name": {"type": "string"},
                                            "kind": {
                                                "type": "string",
                                                "enum": ["referent", "category", "literal", "uncertain"],
                                            },
                                            "type": {"type": "string", "enum": list(TYPE_TO_KIND)},
                                        },
                                    },
                                },
                            },
                        },
                    }
                },
            },
        },
    }


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
