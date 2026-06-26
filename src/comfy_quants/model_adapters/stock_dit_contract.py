"""Generic static contract + graph builder for stock-ComfyUI-native DiT families.

Shared by the ``flux`` / ``ltxv`` / ``ideogram4`` adapters. Like the per-family
qwen / anima builders it consumes declarative :class:`ModuleContract` /
:class:`TensorContract` primitives (reused from ``qwen_contracts.types``) and a free
``dims`` dict, but it additionally supports **multiple repeated block groups** (FLUX
has both ``double_blocks`` and ``single_blocks``) instead of a single block prefix.

Targets the stock-ComfyUI-native formats only (fp8_e4m3 / fp8_e5m2 / mxfp8 / nvfp4);
the artifact uses the bare ComfyUI ``diffusion_models`` module names so it loads
natively. Keys/shapes are authored from ComfyUI's ``comfy/ldm/<family>`` sources and
are validated against a real checkpoint by the writer's strict missing-tensor check at
first export.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from math import prod
from typing import Any

from comfy_quants.core.graph import ModelGraph, ModelInspection, ModuleSpec, TensorSpec
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.qwen_contracts.types import ModuleContract, TensorContract  # noqa: F401

__all__ = [
    "TensorContract",
    "ModuleContract",
    "BlockGroup",
    "StockDitContract",
    "linear",
    "kept_component",
    "build_stock_dit_graph",
    "summarize_stock_dit_graph",
]

ShapeValue = int | str


def linear(
    name: str,
    out_dim: ShapeValue,
    in_dim: ShapeValue,
    *,
    quantizable: bool = True,
    module_type: str = "Linear",
    component: str = "transformer",
    notes: str = "",
) -> ModuleContract:
    """A weight-only Linear module contract (declares only ``.weight``)."""
    return ModuleContract(
        name_template=name,
        module_type=module_type,
        component=component,
        quantizable=quantizable,
        default_action="quantize" if quantizable else "keep_bf16",
        tensors=(
            TensorContract(
                name_template=f"{name}.weight",
                shape_template=(out_dim, in_dim),
                role="weight",
                scale_axis="out_features" if quantizable else None,
            ),
        ),
        notes=notes,
    )


def kept_component(name: str, module_type: str, component: str, notes: str) -> ModuleContract:
    """A coarse high-precision module (tensors copied verbatim by the exporter)."""
    return ModuleContract(
        name_template=name,
        module_type=module_type,
        component=component,
        quantizable=False,
        default_action="keep_bf16",
        tensors=(),
        notes=notes,
    )


@dataclass(frozen=True)
class BlockGroup:
    """A repeated transformer block group (``{prefix}.{block}.*``)."""

    prefix: str
    count: int
    modules: tuple[ModuleContract, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StockDitContract:
    """Complete adapter-owned contract for one stock-ComfyUI DiT variant."""

    family: str
    schema_version: str
    preferred_format: str
    dims: dict[str, int]
    block_groups: tuple[BlockGroup, ...]
    extra_components: tuple[ModuleContract, ...]
    artifact_target: str = "comfyui"
    contract_mode: str = "static_adapter_contract"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def block_count(self) -> int:
        return sum(group.count for group in self.block_groups)

    def dimensions(self) -> dict[str, int]:
        return dict(self.dims)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _render(template: str, *, block: int | None = None) -> str:
    return template if block is None else template.format(block=block)


def _resolve_value(value: ShapeValue, dims: dict[str, int]) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(dims[value])
    except KeyError as exc:
        raise KeyError(f"unknown dimension key {value!r}") from exc


def _tensor_spec(contract: TensorContract, dims: dict[str, int], *, block: int | None = None) -> TensorSpec:
    shape = [_resolve_value(value, dims) for value in contract.shape_template]
    return TensorSpec(
        name=_render(contract.name_template, block=block),
        shape=shape,
        dtype=contract.dtype,
        parameter_count=prod(shape) if shape else 0,
        role=contract.role,
        scale_axis=contract.scale_axis,
    )


def _module_spec(contract: ModuleContract, dims: dict[str, int], *, block: int | None = None) -> ModuleSpec:
    return ModuleSpec(
        name=_render(contract.name_template, block=block),
        module_type=contract.module_type,
        component=contract.component,
        tensors=[_tensor_spec(tensor, dims, block=block) for tensor in contract.tensors],
        quantizable=contract.quantizable,
        default_action=contract.default_action,
        notes=contract.notes,
    )


def _contract_summary(contract: StockDitContract) -> dict[str, Any]:
    return {
        "schema_version": contract.schema_version,
        "family": contract.family,
        "artifact_target": contract.artifact_target,
        "contract_mode": contract.contract_mode,
        "preferred_format": contract.preferred_format,
        "architecture": contract.metadata.get("architecture"),
        "block_groups": {group.prefix: group.count for group in contract.block_groups},
        "block_count": contract.block_count,
        "dims": contract.dimensions(),
    }


def build_stock_dit_graph(
    contract: StockDitContract,
    source: ModelSource,
    *,
    artifact_metadata: dict[str, Any] | None = None,
) -> ModelGraph:
    dims = contract.dimensions()
    modules: list[ModuleSpec] = []
    for group in contract.block_groups:
        for block in range(group.count):
            modules.extend(_module_spec(module, dims, block=block) for module in group.modules)
    modules.extend(_module_spec(module, dims) for module in contract.extra_components)

    metadata: dict[str, Any] = {
        "graph_kind": "static_model_contract",
        "tensor_coverage": "declared_tensors",
        "contract_schema": contract.schema_version,
        "preferred_format": contract.preferred_format,
        "contract_source": "comfy_quants",
        "artifact_target": contract.artifact_target,
        "contract_mode": contract.contract_mode,
        "model_contract": _contract_summary(contract),
    }
    metadata.update(contract.metadata)
    if artifact_metadata:
        metadata.update(artifact_metadata)
        metadata["contract_source"] = "comfy_quants"
        metadata["graph_kind"] = "static_model_contract"
        metadata["tensor_coverage"] = "declared_tensors"
    return ModelGraph(
        family=contract.family,
        model_id=source.model_id,
        revision=source.revision,
        modules=modules,
        metadata=metadata,
    )


def summarize_stock_dit_graph(graph: ModelGraph, adapter: str) -> ModelInspection:
    counter = Counter(module.component for module in graph.modules)
    quantizable = sum(1 for module in graph.modules if module.quantizable)
    kept = len(graph.modules) - quantizable
    return ModelInspection(
        family=graph.family,
        model_id=graph.model_id,
        revision=graph.revision,
        adapter=adapter,
        total_parameters=graph.total_parameters,
        quantizable_modules=quantizable,
        kept_high_precision_modules=kept,
        components=dict(counter),
        warnings=["inspection uses static adapter contract metadata"],
        metadata=graph.metadata,
    )
