"""QuantTensor metadata used by Comfy Quants artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from comfy_quants.core.dtypes import get_dtype_spec


@dataclass
class ScaleMetadata:
    dtype: str
    shape: list[int]
    granularity: str
    axis: int | str | None = None
    block_size: int | None = None
    file: str | None = None
    tensor_name: str | None = None


@dataclass
class PackingMetadata:
    subbyte: bool = False
    endianness: str | None = None
    nibble_order: str | None = None
    padding: int | None = None


@dataclass
class PayloadMetadata:
    file: str | None = None
    tensor_name: str | None = None
    storage_dtype: str | None = None


@dataclass
class QuantTensorMetadata:
    name: str
    source_name: str
    shape: list[int]
    source_dtype: str
    quant_dtype: str
    storage_dtype: str
    algorithm: str
    scale: ScaleMetadata
    payload: PayloadMetadata | None = None
    zero_point: dict[str, Any] | None = None
    packing: PackingMetadata | None = None
    rounding: str = "nearest_even"
    fallback: bool = False
    compatibility_level: str = "L1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spec = get_dtype_spec(self.quant_dtype)
        if self.packing is None:
            self.packing = PackingMetadata(subbyte=spec.subbyte)
        if not self.storage_dtype:
            self.storage_dtype = spec.storage_dtype

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantTensorMetadata":
        data = dict(data)
        data["scale"] = ScaleMetadata(**data["scale"])
        if data.get("payload") is not None:
            data["payload"] = PayloadMetadata(**data["payload"])
        if data.get("packing") is not None:
            data["packing"] = PackingMetadata(**data["packing"])
        return cls(**data)
