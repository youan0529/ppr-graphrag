"""Graph node and edge dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
