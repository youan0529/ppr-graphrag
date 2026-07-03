"""NetworkX graph persistence."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx


def save_graph(graph: nx.Graph, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(graph, f)
    tmp.replace(p)


def load_graph(path: str | Path) -> nx.Graph:
    with Path(path).open("rb") as f:
        return pickle.load(f)
