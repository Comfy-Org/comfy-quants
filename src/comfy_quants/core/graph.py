"""Model graph and inspection domain objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TensorSpec:
    """Description of a tensor discovered during inspection."""

    name: str
    shape: list[int]
    dtype: str
    parameter_count: int = 0
    role: str = "weight"
    scale_axis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModuleSpec:
    """Description of a model module and its quantization eligibility."""

    name: str
    module_type: str
    component: str
    tensors: list[TensorSpec] = field(default_factory=list)
    quantizable: bool = True
    default_action: str = "quantize"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelGraph:
    """Framework-neutral model graph emitted by model adapters."""

    family: str
    model_id: str
    revision: str | None
    modules: list[ModuleSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def total_parameters(self) -> int:
        return sum(t.parameter_count for m in self.modules for t in m.tensors)


@dataclass
class ModelInspection:
    """Summary emitted by inspect commands."""

    family: str
    model_id: str
    revision: str | None
    adapter: str
    total_parameters: int
    quantizable_modules: int
    kept_high_precision_modules: int
    components: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
