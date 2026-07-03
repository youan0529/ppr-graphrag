"""SentenceTransformers embedder."""

from __future__ import annotations

import numpy as np

from ppr_graphrag.core.config import EmbeddingConfig


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers to use SentenceTransformerEmbedder: "
                "pip install sentence-transformers"
            ) from exc
        self.config = config
        self.model = SentenceTransformer(config.model_name_or_path, device=config.device)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
