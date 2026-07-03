"""Small graph builders used by early PPR retrieval experiments."""

from __future__ import annotations

import re

import networkx as nx

from ppr_graphrag.data.schema import Document


def simple_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 2}


def build_token_overlap_graph(corpus: list[Document], directed: bool = False, add_self_loops: bool = False) -> nx.Graph:
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    token_sets: dict[str, set[str]] = {}
    for doc in corpus:
        graph.add_node(doc.doc_id, node_type="document", title=doc.title, text=doc.text)
        token_sets[doc.doc_id] = simple_tokens(f"{doc.title or ''} {doc.text}")
        if add_self_loops:
            graph.add_edge(doc.doc_id, doc.doc_id, weight=1.0)
    doc_ids = [doc.doc_id for doc in corpus]
    for i, left in enumerate(doc_ids):
        for right in doc_ids[i + 1 :]:
            overlap = token_sets[left] & token_sets[right]
            if overlap:
                graph.add_edge(left, right, weight=float(len(overlap)), overlap=sorted(overlap)[:20])
    return graph
