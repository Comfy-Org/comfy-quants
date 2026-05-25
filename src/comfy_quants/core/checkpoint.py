"""Job checkpoint records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CheckpointRecord:
    """Layer/module-level checkpoint used by resumable jobs."""

    step_id: str
    status: str
    module_name: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    input_hash: str | None = None
    output_hash: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
