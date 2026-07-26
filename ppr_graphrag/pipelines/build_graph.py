"""Materialize a deterministic undirected Fact-Entity graph."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import time
from typing import Any
import unicodedata

import faiss
import networkx as nx
import numpy as np

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.core.config import AppConfig
from ppr_graphrag.core.hashing import sha256_text, stable_json_dumps
from ppr_graphrag.embedding.nv_embed_v2 import NVEmbedV2Embedder
from ppr_graphrag.embedding.vector_cache import CachedVectorEmbedder, SQLiteVectorCache
from ppr_graphrag.graph.schema import GRAPH_KINDS


GRAPH_PARTS = ("artifacts", "graph")


def normalize_display_name(name: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", name).strip())


def normalize_name_key(name: str) -> str:
    clean = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", name).strip())
    return clean.casefold()


def normalize_fact_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip())


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix}_{sha256_text(stable_json_dumps(payload))[:20]}"


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as file:
        np.save(file, array)
    tmp.replace(path)


def _representative_name(forms: Counter[str]) -> str:
    return min(forms, key=lambda name: (-forms[name], name))


def _entity_similarity_edges(
    entities: list[dict[str, Any]],
    embeddings: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for index, entity in enumerate(entities):
        by_type[entity["type"]].append(index)

    edges = []
    for entity_type, global_indices in sorted(by_type.items()):
        vectors = np.ascontiguousarray(embeddings[global_indices], dtype=np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        limits, distances, neighbors = index.range_search(vectors, threshold)
        for left_local in range(len(global_indices)):
            for offset in range(limits[left_local], limits[left_local + 1]):
                right_local = int(neighbors[offset])
                if right_local <= left_local:
                    continue
                score = float(distances[offset])
                if score < threshold:
                    continue
                left = entities[global_indices[left_local]]["entity_id"]
                right = entities[global_indices[right_local]]["entity_id"]
                edges.append(
                    {
                        "source_id": left,
                        "target_id": right,
                        "edge_type": "semantic_similarity",
                        "weight": score,
                    }
                )
    return edges


def _graph_report(
    graph: nx.Graph,
    facts: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    extraction_rows: list[dict[str, Any]],
    extraction_errors: list[dict[str, Any]],
    mention_kind_counts: Counter[str],
    mention_type_counts: Counter[str],
    embedding_config: dict[str, Any],
    elapsed: dict[str, float],
) -> dict[str, Any]:
    fact_edges = [edge for edge in edges if edge["edge_type"] == "fact_entity"]
    semantic_edges = [edge for edge in edges if edge["edge_type"] == "semantic_similarity"]
    semantic_weights = [edge["weight"] for edge in semantic_edges]
    entity_types = {entity["entity_id"]: entity["type"] for entity in entities}
    components = list(nx.connected_components(graph))
    degrees = [degree for _, degree in graph.degree()]
    return {
        "extraction": {
            "num_successful_passages": len(extraction_rows),
            "num_failed_passages": len(extraction_errors),
        },
        "facts": {
            "count": len(facts),
            "without_graph_mentions": sum(not any(m["in_graph"] for m in fact["mentions"]) for fact in facts),
        },
        "mentions": {
            "count": sum(mention_kind_counts.values()),
            "by_kind": dict(sorted(mention_kind_counts.items())),
            "by_type": dict(sorted(mention_type_counts.items())),
        },
        "entities": {
            "count": len(entities),
            "by_type": dict(sorted(Counter(entity["type"] for entity in entities).items())),
            "exact_mention_merges": sum(mention_kind_counts[kind] for kind in GRAPH_KINDS) - len(entities),
        },
        "edges": {
            "count": len(edges),
            "fact_entity": len(fact_edges),
            "semantic_similarity": len(semantic_edges),
            "semantic_similarity_by_type": dict(
                sorted(Counter(entity_types[edge["source_id"]] for edge in semantic_edges).items())
            ),
            "semantic_similarity_min": min(semantic_weights) if semantic_weights else None,
            "semantic_similarity_mean": float(np.mean(semantic_weights)) if semantic_weights else None,
            "semantic_similarity_max": max(semantic_weights) if semantic_weights else None,
        },
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "components": len(components),
            "largest_component": max((len(component) for component in components), default=0),
            "isolated_nodes": nx.number_of_isolates(graph),
            "degree_min": min(degrees, default=0),
            "degree_mean": float(np.mean(degrees)) if degrees else 0.0,
            "degree_max": max(degrees, default=0),
        },
        "embedding": embedding_config,
        "elapsed_seconds": elapsed,
    }


def build_fact_entity_graph(
    config: AppConfig,
    artifacts: ArtifactManager,
    overwrite: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    output_names = (
        "entities.jsonl",
        "facts.jsonl",
        "edges.jsonl",
        "entity_embeddings.npy",
        "entity_embeddings_meta.json",
        "graph.pkl",
        "graph_report.json",
    )
    existing = [name for name in output_names if artifacts.exists(*GRAPH_PARTS, name)]
    if existing and not overwrite:
        raise FileExistsError(f"Graph outputs already exist: {existing}. Re-run with --overwrite.")

    extraction_rows = artifacts.load_jsonl("artifacts", "extraction", "raw_extractions.jsonl")
    extraction_errors = artifacts.load_jsonl("artifacts", "extraction", "extraction_errors.jsonl")
    extraction_report = artifacts.load_json("artifacts", "extraction", "extraction_report.json")
    if extraction_report["num_completed"] != extraction_report["num_passages"]:
        raise RuntimeError("Fact extraction is incomplete; resume scripts/extract_facts.py before building the graph")

    entity_groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    mention_kind_counts: Counter[str] = Counter()
    mention_type_counts: Counter[str] = Counter()
    for passage in extraction_rows:
        for fact in passage["facts"]:
            for mention in fact["mentions"]:
                mention_kind_counts[mention["kind"]] += 1
                mention_type_counts[mention["type"]] += 1
                if mention["kind"] in GRAPH_KINDS:
                    key = (mention["type"], normalize_name_key(mention["name"]))
                    entity_groups[key][normalize_display_name(mention["name"])] += 1

    entities = []
    entity_id_by_key = {}
    for (entity_type, name_key), forms in entity_groups.items():
        entity_id = _stable_id("E", {"type": entity_type, "name": name_key})
        entity_id_by_key[(entity_type, name_key)] = entity_id
        entities.append(
            {
                "entity_id": entity_id,
                "entity_name": _representative_name(forms),
                "type": entity_type,
            }
        )
    entities.sort(key=lambda entity: entity["entity_id"])

    facts = []
    fact_edges = []
    for passage in extraction_rows:
        for raw_fact in passage["facts"]:
            fact_text = normalize_fact_text(raw_fact["text"])
            fact_id = _stable_id("F", {"passage_id": passage["passage_id"], "text": fact_text})
            mentions = []
            linked_entities = set()
            for raw_mention in raw_fact["mentions"]:
                mention = {
                    "name": normalize_display_name(raw_mention["name"]),
                    "kind": raw_mention["kind"],
                    "type": raw_mention["type"],
                    "in_graph": raw_mention["kind"] in GRAPH_KINDS,
                }
                if mention["in_graph"]:
                    key = (mention["type"], normalize_name_key(mention["name"]))
                    mention["entity_id"] = entity_id_by_key[key]
                    linked_entities.add(mention["entity_id"])
                mentions.append(mention)
            facts.append(
                {
                    "fact_id": fact_id,
                    "text": fact_text,
                    "relation": normalize_fact_text(raw_fact["relation"]),
                    "passage_id": passage["passage_id"],
                    "mentions": mentions,
                }
            )
            for entity_id in sorted(linked_entities):
                fact_edges.append(
                    {
                        "source_id": fact_id,
                        "target_id": entity_id,
                        "edge_type": "fact_entity",
                        "weight": 1.0,
                    }
                )
    facts.sort(key=lambda fact: fact["fact_id"])
    fact_edges.sort(key=lambda edge: (edge["source_id"], edge["target_id"]))

    cache = SQLiteVectorCache(config.embedding.cache_path)
    embedding_started = time.perf_counter()
    try:
        embedder = CachedVectorEmbedder(NVEmbedV2Embedder(config.embedding), cache, config.embedding)
        embeddings = embedder.embed_texts([entity["entity_name"] for entity in entities])
        embedding_cache_stats = cache.stats()
    finally:
        cache.close()
    embedding_elapsed = time.perf_counter() - embedding_started

    semantic_edges = _entity_similarity_edges(
        entities,
        embeddings,
        config.graph.entity_similarity_threshold,
    )
    edges = sorted(
        fact_edges + semantic_edges,
        key=lambda edge: (edge["edge_type"], edge["source_id"], edge["target_id"]),
    )

    graph = nx.Graph()
    for entity in entities:
        graph.add_node(
            entity["entity_id"],
            node_type="entity",
            entity_name=entity["entity_name"],
            entity_type=entity["type"],
        )
    for fact in facts:
        graph.add_node(
            fact["fact_id"],
            node_type="fact",
            text=fact["text"],
            relation=fact["relation"],
            passage_id=fact["passage_id"],
        )
    for edge in edges:
        graph.add_edge(
            edge["source_id"],
            edge["target_id"],
            edge_type=edge["edge_type"],
            weight=edge["weight"],
        )

    graph_dir = artifacts.path(*GRAPH_PARTS)
    _write_jsonl_atomic(graph_dir / "entities.jsonl", entities)
    _write_jsonl_atomic(graph_dir / "facts.jsonl", facts)
    _write_jsonl_atomic(graph_dir / "edges.jsonl", edges)
    _save_npy_atomic(graph_dir / "entity_embeddings.npy", embeddings)
    embedding_meta = {
        "model": config.embedding.model_name_or_path,
        "entity_ids": [entity["entity_id"] for entity in entities],
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "normalized": config.embedding.normalize,
        "cache": embedding_cache_stats,
    }
    artifacts.save_json_atomic(embedding_meta, *GRAPH_PARTS, "entity_embeddings_meta.json")
    artifacts.save_pickle_atomic(graph, *GRAPH_PARTS, "graph.pkl")
    elapsed = {
        "embedding": round(embedding_elapsed, 3),
        "total": round(time.perf_counter() - started_at, 3),
    }
    report = _graph_report(
        graph,
        facts,
        entities,
        edges,
        extraction_rows,
        extraction_errors,
        mention_kind_counts,
        mention_type_counts,
        {
            "model": config.embedding.model_name_or_path,
            "dimension": embeddings.shape[1] if embeddings.size else 0,
            "similarity_threshold": config.graph.entity_similarity_threshold,
            "cache": embedding_cache_stats,
        },
        elapsed,
    )
    artifacts.save_json_atomic(report, *GRAPH_PARTS, "graph_report.json")
    artifacts.write_manifest(config)
    return report
