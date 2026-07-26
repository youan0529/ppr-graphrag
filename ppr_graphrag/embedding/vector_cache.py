"""SQLite BLOB cache for high-dimensional embedding vectors."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from ppr_graphrag.core.config import EmbeddingConfig
from ppr_graphrag.core.hashing import make_id
from ppr_graphrag.embedding.base import BaseEmbedder


class SQLiteVectorCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_vectors (
                key TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                dim INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT vector, dim FROM embedding_vectors WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            return np.frombuffer(row[0], dtype=np.float32, count=row[1]).copy()

    def set(self, key: str, vector: np.ndarray) -> None:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO embedding_vectors(key, vector, dim, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, array.tobytes(), array.size, time.time()),
            )
            self.conn.commit()

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            count = self.conn.execute("SELECT COUNT(*) FROM embedding_vectors").fetchone()[0]
            return {"path": str(self.path), "entries": count}

    def close(self) -> None:
        with self._lock:
            self.conn.close()


class CachedVectorEmbedder:
    def __init__(self, embedder: BaseEmbedder, cache: SQLiteVectorCache, config: EmbeddingConfig):
        self.embedder = embedder
        self.cache = cache
        self.config = config

    def _key(self, text: str) -> str:
        return make_id("emb", {"model": self.config.model_name_or_path, "input": text})

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        for index, text in enumerate(texts):
            cached = self.cache.get(self._key(text))
            if cached is not None:
                vectors[index] = cached
            else:
                missing.setdefault(text, []).append(index)

        if missing:
            missing_texts = list(missing)
            computed = self.embedder.embed_texts(missing_texts)
            for text, vector in zip(missing_texts, computed, strict=True):
                array = np.asarray(vector, dtype=np.float32)
                self.cache.set(self._key(text), array)
                for index in missing[text]:
                    vectors[index] = array
        return np.vstack([vector for vector in vectors if vector is not None])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
