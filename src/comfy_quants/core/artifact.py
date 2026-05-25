"""Artifact container."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QuantArtifact:
    """In-memory descriptor for a quantized artifact directory."""

    artifact_id: str
    manifest_path: str
    tensor_index_path: str | None = None
    files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
