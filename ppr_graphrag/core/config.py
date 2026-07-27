"""Dataclass configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


@dataclass
class ExperimentConfig:
    project_name: str = "ppr_graphrag"
    run_name: str = "default"
    seed: int = 13
    output_dir: str = "experiments"


@dataclass
class LLMConfig:
    provider: str = "openai_compatible"
    model: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434/v1"
    base_urls: list[str] = field(default_factory=list)
    api_key: str = "ollama"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 120.0
    reasoning_effort: str = "low"
    cache_enabled: bool = True
    cache_path: str = "cache/llm.sqlite"


@dataclass
class EmbeddingConfig:
    provider: str = "sentence_transformers"
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True
    cache_path: str = "cache/embedding.sqlite"


@dataclass
class DataConfig:
    corpus_path: str = "data/toy/corpus.jsonl"
    queries_path: str = "data/toy/queries.jsonl"
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass
class GraphConfig:
    graph_path: str = "artifacts/graph.pkl"
    directed: bool = False
    add_self_loops: bool = False
    extraction_prompt_path: str = "prompts/extract_passage.md"
    extraction_workers: int = 1
    entity_similarity_threshold: float = 0.8


@dataclass
class RetrievalConfig:
    method: str = "bm25"
    top_k: int = 5
    dense_candidate_k: int = 100
    ppr_fact_seed_k: int = 20
    ppr_entity_seed_k: int = 10
    entity_seed_threshold: float = 0.5
    fact_seed_weight: float = 0.8
    ppr_alpha: float = 0.85
    ppr_max_iter: int = 100
    ppr_tol: float = 1e-6


@dataclass
class ResumeConfig:
    resume_index: bool = True
    resume_retrieval: bool = True
    force_rebuild_index: bool = False
    force_retrieve: bool = False


@dataclass
class AppConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)


T = TypeVar("T")


def _from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    data = data or {}
    kwargs: dict[str, Any] = {}
    hints = get_type_hints(cls)
    for f in fields(cls):
        value = data.get(f.name)
        field_type = hints.get(f.name, f.type)
        if value is not None and isinstance(field_type, type) and is_dataclass(field_type):
            kwargs[f.name] = _from_dict(field_type, value)
        elif value is not None:
            kwargs[f.name] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def load_config(path: str | Path) -> AppConfig:
    """Load an AppConfig from YAML, applying dataclass defaults for missing keys."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(AppConfig, raw)
