"""Quantization format metadata.

A format is intentionally independent from model families.  One format such as
``fp8_e4m3`` can be selected by many model adapters; adapters decide *which*
modules/tensors to quantize, while the format describes *how* values are stored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class QuantFormatSpec:
    """Static description of a reusable quantized tensor format."""

    name: str
    storage_dtype: str
    bits: int
    category: str
    scale_required: bool
    default_scale_granularity: str
    compatible_families: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["compatible_families"] = list(self.compatible_families)
        data["notes"] = list(self.notes)
        return data
