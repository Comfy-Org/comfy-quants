"""Static Qwen model contract data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ShapeValue = int | str


@dataclass(frozen=True)
class TensorContract:
    """Tensor declaration emitted into a model graph."""

    name_template: str
    shape_template: tuple[ShapeValue, ...]
    dtype: str = "bf16"
    role: str = "weight"
    scale_axis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModuleContract:
    """Module declaration with quantization default and tensors."""

    name_template: str
    module_type: str
    component: str
    quantizable: bool
    default_action: str
    tensors: tuple[TensorContract, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransformerContract:
    """Qwen transformer dimensions and repeated block declarations."""

    component: str
    block_prefix: str
    block_count: int
    hidden_size: int
    intermediate_size: int
    attention_head_dim: int
    num_attention_heads: int
    joint_attention_dim: int
    in_channels: int
    out_channels: int
    patch_size: int
    timestep_projection_size: int
    pre_modules: tuple[ModuleContract, ...]
    block_modules: tuple[ModuleContract, ...]
    post_modules: tuple[ModuleContract, ...]

    @property
    def output_projection_size(self) -> int:
        return self.patch_size * self.patch_size * self.out_channels

    def dimensions(self) -> dict[str, int]:
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "attention_head_dim": self.attention_head_dim,
            "num_attention_heads": self.num_attention_heads,
            "joint_attention_dim": self.joint_attention_dim,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "patch_size": self.patch_size,
            "output_projection_size": self.output_projection_size,
            "timestep_projection_size": self.timestep_projection_size,
            "modulation_size": self.hidden_size * 6,
            "final_modulation_size": self.hidden_size * 2,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QwenModelContract:
    """Complete adapter-owned contract for one Qwen model family."""

    family: str
    schema_version: str
    artifact_target: str
    contract_mode: str
    preferred_format: str
    transformer: TransformerContract
    extra_components: tuple[ModuleContract, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
