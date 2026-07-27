"""Dense and PPR retrieval over the materialized Fact-Entity graph."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import faiss
import networkx as nx
import numpy as np
from scipy import sparse

from ppr_graphrag.core.artifacts import ArtifactManager
from ppr_graphrag.core.config import AppConfig
from ppr_graphrag.data.schema import Query
from ppr_graphrag.embedding.nv_embed_v2 import NVEmbedV2Embedder
from ppr_graphrag.embedding.vector_cache import CachedVectorEmbedder, SQLiteVectorCache
from ppr_graphrag.retrieval.base import RetrievalResult


GRAPH_PARTS = ("artifacts", "graph")


class FactEntityGraphRetriever:
    def __init__(self, config: AppConfig, artifacts: ArtifactManager):
        self.config = config
        self.artifacts = artifacts
        self.facts = artifacts.load_jsonl(*GRAPH_PARTS, "facts.jsonl")
        self.entities = artifacts.load_jsonl(*GRAPH_PARTS, "entities.jsonl")
        self.graph: nx.Graph = artifacts.load_pickle(*GRAPH_PARTS, "graph.pkl")
        self.fact_embeddings = np.load(
            artifacts.path(*GRAPH_PARTS, "fact_embeddings.npy"),
            mmap_mode="r",
        )
        self.entity_embeddings = np.load(
            artifacts.path(*GRAPH_PARTS, "entity_embeddings.npy"),
            mmap_mode="r",
        )
        self._validate_artifacts()

        self.fact_index = faiss.IndexFlatIP(self.fact_embeddings.shape[1])
        self.fact_index.add(np.ascontiguousarray(self.fact_embeddings, dtype=np.float32))
        self.entity_index = faiss.IndexFlatIP(self.entity_embeddings.shape[1])
        self.entity_index.add(np.ascontiguousarray(self.entity_embeddings, dtype=np.float32))

        self.fact_ids = [fact["fact_id"] for fact in self.facts]
        self.entity_ids = [entity["entity_id"] for entity in self.entities]
        self.passages = sorted({fact["passage_id"] for fact in self.facts})
        passage_code = {passage_id: index for index, passage_id in enumerate(self.passages)}
        self.fact_passage_codes = np.asarray(
            [passage_code[fact["passage_id"]] for fact in self.facts],
            dtype=np.int32,
        )

        self.node_ids = sorted(self.graph.nodes)
        self.node_index = {node_id: index for index, node_id in enumerate(self.node_ids)}
        self.fact_node_indices = np.asarray([self.node_index[fact_id] for fact_id in self.fact_ids])
        self.entity_node_indices = np.asarray([self.node_index[entity_id] for entity_id in self.entity_ids])
        self.transition, self.dangling = self._build_transition()

        self.vector_cache = SQLiteVectorCache(config.embedding.cache_path)
        self.query_embedder = CachedVectorEmbedder(
            NVEmbedV2Embedder(config.embedding),
            self.vector_cache,
            config.embedding,
        )
        self.query_vectors: dict[str, np.ndarray] = {}

    def _validate_artifacts(self) -> None:
        fact_meta = self.artifacts.load_json(*GRAPH_PARTS, "fact_embeddings_meta.json")
        entity_meta = self.artifacts.load_json(*GRAPH_PARTS, "entity_embeddings_meta.json")
        if fact_meta["fact_ids"] != [fact["fact_id"] for fact in self.facts]:
            raise ValueError("Fact embedding order does not match facts.jsonl")
        if entity_meta["entity_ids"] != [entity["entity_id"] for entity in self.entities]:
            raise ValueError("Entity embedding order does not match entities.jsonl")
        if self.fact_embeddings.shape[0] != len(self.facts):
            raise ValueError("Fact embedding row count does not match facts.jsonl")
        if self.entity_embeddings.shape[0] != len(self.entities):
            raise ValueError("Entity embedding row count does not match entities.jsonl")

    def _build_transition(self) -> tuple[sparse.csr_matrix, np.ndarray]:
        adjacency = nx.to_scipy_sparse_array(
            self.graph,
            nodelist=self.node_ids,
            weight="weight",
            dtype=np.float64,
            format="csr",
        )
        degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        dangling = degree == 0
        inverse_degree = np.zeros_like(degree)
        inverse_degree[~dangling] = 1.0 / degree[~dangling]
        return sparse.diags(inverse_degree).dot(adjacency).tocsr(), dangling

    def prepare_queries(self, queries: list[Query]) -> None:
        missing = [query for query in queries if query.query_id not in self.query_vectors]
        if not missing:
            return
        vectors = self.query_embedder.embed_texts([query.question for query in missing])
        for query, vector in zip(missing, vectors, strict=True):
            self.query_vectors[query.query_id] = vector

    def _search_facts(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self.facts))
        scores, indices = self.fact_index.search(query_vector[None].astype(np.float32), k)
        return [(int(index), float(score)) for index, score in zip(indices[0], scores[0], strict=True)]

    def _search_entities(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self.entities))
        scores, indices = self.entity_index.search(query_vector[None].astype(np.float32), k)
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
            if score >= self.config.retrieval.entity_seed_threshold
        ]

    def _fact_trace(self, candidates: list[tuple[int, float]]) -> list[dict[str, Any]]:
        return [
            {
                "fact_id": self.facts[index]["fact_id"],
                "text": self.facts[index]["text"],
                "passage_id": self.facts[index]["passage_id"],
                "score": score,
            }
            for index, score in candidates
        ]

    def _entity_trace(self, candidates: list[tuple[int, float]]) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": self.entities[index]["entity_id"],
                "entity_name": self.entities[index]["entity_name"],
                "type": self.entities[index]["type"],
                "score": score,
            }
            for index, score in candidates
        ]

    def _rank_candidate_passages(
        self,
        query_id: str,
        candidates: list[tuple[int, float]],
        top_k: int,
        method: str,
    ) -> list[RetrievalResult]:
        best: dict[str, tuple[float, int]] = {}
        for fact_index, score in candidates:
            passage_id = self.facts[fact_index]["passage_id"]
            if passage_id not in best or score > best[passage_id][0]:
                best[passage_id] = (score, fact_index)
        ranked = sorted(best.items(), key=lambda item: (-item[1][0], item[0]))[:top_k]
        return [
            RetrievalResult(
                query_id=query_id,
                doc_id=passage_id,
                score=score,
                rank=rank,
                metadata={
                    "method": method,
                    "best_fact_id": self.facts[fact_index]["fact_id"],
                    "best_fact_text": self.facts[fact_index]["text"],
                },
            )
            for rank, (passage_id, (score, fact_index)) in enumerate(ranked, start=1)
        ]

    def retrieve_dense(
        self,
        query: Query,
        top_k: int,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        query_vector = self.query_vectors[query.query_id]
        candidates = self._search_facts(query_vector, self.config.retrieval.dense_candidate_k)
        results = self._rank_candidate_passages(query.query_id, candidates, top_k, "graph_dense")
        return results, {"fact_candidates": self._fact_trace(candidates)}

    @staticmethod
    def _normalized_weights(candidates: list[tuple[int, float]]) -> np.ndarray:
        weights = np.asarray([max(score, 0.0) for _, score in candidates], dtype=np.float64)
        if not weights.size:
            return weights
        total = weights.sum()
        return weights / total if total > 0 else np.full(weights.shape, 1.0 / len(weights))

    def _personalization(
        self,
        fact_seeds: list[tuple[int, float]],
        entity_seeds: list[tuple[int, float]],
    ) -> np.ndarray:
        personalization = np.zeros(len(self.node_ids), dtype=np.float64)
        fact_mix = min(max(self.config.retrieval.fact_seed_weight, 0.0), 1.0)
        if not entity_seeds:
            fact_mix = 1.0
        if not fact_seeds:
            fact_mix = 0.0
        for (index, _), weight in zip(fact_seeds, self._normalized_weights(fact_seeds), strict=True):
            personalization[self.fact_node_indices[index]] += fact_mix * weight
        for (index, _), weight in zip(entity_seeds, self._normalized_weights(entity_seeds), strict=True):
            personalization[self.entity_node_indices[index]] += (1.0 - fact_mix) * weight
        if personalization.sum() == 0:
            raise ValueError("No Fact or Entity seeds available for PPR")
        return personalization / personalization.sum()

    def _pagerank(self, personalization: np.ndarray) -> tuple[np.ndarray, int, bool]:
        alpha = self.config.retrieval.ppr_alpha
        scores = personalization.copy()
        for iteration in range(1, self.config.retrieval.ppr_max_iter + 1):
            dangling_mass = float(scores[self.dangling].sum())
            updated = alpha * self.transition.T.dot(scores)
            updated += (1.0 - alpha + alpha * dangling_mass) * personalization
            if np.abs(updated - scores).sum() <= self.config.retrieval.ppr_tol:
                return np.asarray(updated), iteration, True
            scores = np.asarray(updated)
        return scores, self.config.retrieval.ppr_max_iter, False

    def _rank_ppr_passages(
        self,
        query_id: str,
        node_scores: np.ndarray,
        top_k: int,
    ) -> tuple[list[RetrievalResult], list[dict[str, Any]]]:
        fact_scores = node_scores[self.fact_node_indices]
        passage_scores = np.full(len(self.passages), -np.inf, dtype=np.float64)
        np.maximum.at(passage_scores, self.fact_passage_codes, fact_scores)
        candidate_count = min(top_k, len(self.passages))
        passage_indices = np.argpartition(-passage_scores, candidate_count - 1)[:candidate_count]
        passage_indices = sorted(passage_indices, key=lambda index: (-passage_scores[index], self.passages[index]))

        results = []
        for rank, passage_index in enumerate(passage_indices, start=1):
            positions = np.flatnonzero(self.fact_passage_codes == passage_index)
            best_fact_index = int(positions[np.argmax(fact_scores[positions])])
            results.append(
                RetrievalResult(
                    query_id=query_id,
                    doc_id=self.passages[passage_index],
                    score=float(passage_scores[passage_index]),
                    rank=rank,
                    metadata={
                        "method": "graph_ppr",
                        "best_fact_id": self.facts[best_fact_index]["fact_id"],
                        "best_fact_text": self.facts[best_fact_index]["text"],
                    },
                )
            )

        fact_count = min(20, len(self.facts))
        top_fact_indices = np.argpartition(-fact_scores, fact_count - 1)[:fact_count]
        top_fact_indices = sorted(top_fact_indices, key=lambda index: (-fact_scores[index], self.fact_ids[index]))
        top_facts = [
            {
                "fact_id": self.fact_ids[index],
                "text": self.facts[index]["text"],
                "passage_id": self.facts[index]["passage_id"],
                "score": float(fact_scores[index]),
            }
            for index in top_fact_indices
        ]
        return results, top_facts

    def retrieve_ppr(
        self,
        query: Query,
        top_k: int,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        query_vector = self.query_vectors[query.query_id]
        fact_seeds = self._search_facts(query_vector, self.config.retrieval.ppr_fact_seed_k)
        entity_seeds = self._search_entities(query_vector, self.config.retrieval.ppr_entity_seed_k)
        personalization = self._personalization(fact_seeds, entity_seeds)
        node_scores, iterations, converged = self._pagerank(personalization)
        results, top_facts = self._rank_ppr_passages(query.query_id, node_scores, top_k)
        return results, {
            "fact_seeds": self._fact_trace(fact_seeds),
            "entity_seeds": self._entity_trace(entity_seeds),
            "top_facts": top_facts,
            "iterations": iterations,
            "converged": converged,
        }

    def retrieve(
        self,
        query: Query,
        top_k: int,
        method: str,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        if query.query_id not in self.query_vectors:
            self.prepare_queries([query])
        if method == "dense":
            return self.retrieve_dense(query, top_k)
        if method == "ppr":
            return self.retrieve_ppr(query, top_k)
        raise ValueError(f"Unknown graph retrieval method: {method}")

    def close(self) -> None:
        self.vector_cache.close()

    @staticmethod
    def serialize_results(results: list[RetrievalResult]) -> list[dict[str, Any]]:
        return [asdict(result) for result in results]
