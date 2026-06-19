"""Stable hashing helpers for provenance and manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_dumps(value: Any) -> str:
    """Serialize JSON in a deterministic way suitable for hashing."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    """Return a prefixed sha256 digest for bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return a prefixed sha256 digest for text."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Return a prefixed sha256 digest for a JSON-compatible object."""
    return sha256_text(stable_json_dumps(value))


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a prefixed sha256 digest for a file."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()
