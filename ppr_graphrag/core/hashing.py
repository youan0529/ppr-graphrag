"""Stable hashing helpers used for IDs, cache keys, and artifact keys."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json_dumps(obj: Any) -> str:
    """Serialize JSON-compatible objects with stable ordering."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """Return the SHA256 hex digest for text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_id(prefix: str, content_or_obj: Any) -> str:
    """Create a stable prefixed ID from text or a JSON-compatible object."""
    content = content_or_obj if isinstance(content_or_obj, str) else stable_json_dumps(content_or_obj)
    return f"{prefix}-{sha256_text(content)[:16]}"
