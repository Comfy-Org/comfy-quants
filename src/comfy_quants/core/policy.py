"""Quantization policy domain objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QuantPolicy:
    """Declarative policy that selects modules and fallback behavior."""

    name: str
    algorithm: str = "fp8_static"
    target_dtype: str = "fp8_e4m3"
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    fallback_on_error: str = "keep_bf16"
    keep_components: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
