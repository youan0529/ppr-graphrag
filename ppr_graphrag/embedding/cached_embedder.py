"""SQLite cache wrapper for embeddings."""

from __future__ import annotations

import numpy as np

from ppr_graphrag.core.cache import SQLiteCache
from ppr_graphrag.core.config import EmbeddingConfig
from ppr_graphrag.core.hashing import make_id
from ppr_graphrag.embedding.base import BaseEmbedder


class CachedEmbedder:
    def __init__(self, embedder: BaseEmbedder, cache: SQLiteCache, config: EmbeddingConfig):
        self.embedder = embedder
        self.cache = cache
        self.config = config

    def _key(self, text: str) -> str:
        return make_id(
            "emb",
            {
                "provider": self.config.provider,
                "model_name_or_path": self.config.model_name_or_path,
                "text": text,
                "normalize": self.config.normalize,
            },
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[np.ndarray | None] = []
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        for i, text in enumerate(texts):
            cached = self.cache.get(self._key(text))
            if cached is None:
                vectors.append(None)
                missing_texts.append(text)
                missing_indices.append(i)
            else:
                vectors.append(np.asarray(cached["embedding"], dtype=np.float32))
        if missing_texts:
            computed = self.embedder.embed_texts(missing_texts)
            for idx, text, vec in zip(missing_indices, missing_texts, computed, strict=True):
                arr = np.asarray(vec, dtype=np.float32)
                vectors[idx] = arr
                self.cache.set(self._key(text), {"embedding": arr.tolist()}, metadata={"model": self.config.model_name_or_path})
        return np.vstack([v for v in vectors if v is not None])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
