"""Local NV-Embed-v2 wrapper for entity-name embeddings."""

from __future__ import annotations

import numpy as np

from ppr_graphrag.core.config import EmbeddingConfig


class NVEmbedV2Embedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.torch = None
        self.model = None
        self.embedding_dim = 0

    def _load_model(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("NV-Embed-v2 requires torch, transformers, and accelerate") from exc
        self.torch = torch
        self.model = AutoModel.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.embedding_dim = int(self.model.config.hidden_size)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        self._load_model()
        assert self.torch is not None and self.model is not None
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.config.batch_size):
                result = self.model.encode(
                    prompts=texts[start : start + self.config.batch_size],
                    instruction="",
                    max_length=64,
                )
                if isinstance(result, self.torch.Tensor):
                    result = result.detach().float().cpu().numpy()
                batches.append(np.asarray(result, dtype=np.float32))
        vectors = np.vstack(batches)
        if self.config.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
