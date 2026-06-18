"""Static Anima model contract data structures.

Reuses the family-agnostic ``TensorContract`` / ``ModuleContract`` primitives from
``qwen_contracts.types``; the transformer contract carries a free ``dimensions`` dict
(anima's dims differ from Qwen's) rather than the Qwen-specific fixed fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from comfy_quants.model_adapters.qwen_contracts.types import ModuleContract, TensorContract  # noqa: F401

__all__ = ["TensorContract", "ModuleContract", "AnimaTransformerContract", "AnimaModelContract"]


@dataclass(frozen=True)
class AnimaTransformerContract:
    """Anima (cosmos_predict2 + llm_adapter) transformer block declarations."""

    component: str
    block_prefix: str
    block_count: int
    num_heads: int
    dims: dict[str, int]
    pre_modules: tuple[ModuleContract, ...]
    block_modules: tuple[ModuleContract, ...]
    post_modules: tuple[ModuleContract, ...]

    def dimensions(self) -> dict[str, int]:
        return dict(self.dims)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnimaModelContract:
    """Complete adapter-owned contract for one Anima size variant."""

    family: str
    schema_version: str
    artifact_target: str
    contract_mode: str
    preferred_format: str
    transformer: AnimaTransformerContract
    extra_components: tuple[ModuleContract, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
