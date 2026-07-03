"""SQLite-backed JSON cache."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class SQLiteCache:
    """A small SQLite cache storing JSON values and metadata."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                metadata_json TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, key: str) -> Any | None:
        row = self.conn.execute("SELECT value_json FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO cache(key, value_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, json.dumps(value, ensure_ascii=False), json.dumps(metadata or {}, ensure_ascii=False), time.time()),
        )
        self.conn.commit()

    def contains(self, key: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM cache WHERE key = ? LIMIT 1", (key,)).fetchone()
        return row is not None

    def stats(self) -> dict[str, Any]:
        count = self.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        return {"path": str(self.path), "entries": count}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SQLiteCache":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
