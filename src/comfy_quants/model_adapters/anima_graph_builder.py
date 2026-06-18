"""Build ModelGraph objects from static Anima contracts.

Mirrors ``qwen_graph_builder`` (per-family builder pattern) but consumes the
Anima contract's free ``dimensions()`` dict.
"""

from __future__ import annotations

from collections import Counter
from math import prod
from typing import Any

from comfy_quants.core.graph import ModelGraph, ModelInspection, ModuleSpec, TensorSpec
from comfy_quants.model_adapters.anima_contracts.types import AnimaModelContract, ModuleContract, TensorContract
from comfy_quants.model_adapters.base import ModelSource

ShapeValue = int | str


def _render_template(template: str, *, block: int | None = None) -> str:
    return template if block is None else template.format(block=block)


def _resolve_value(value: ShapeValue, dimensions: dict[str, int]) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(dimensions[value])
    except KeyError as exc:
        raise KeyError(f"unknown dimension key {value!r}") from exc


def _resolve_shape(shape_template: tuple[ShapeValue, ...], dimensions: dict[str, int]) -> list[int]:
    return [_resolve_value(value, dimensions) for value in shape_template]


def _tensor_from_contract(contract: TensorContract, dimensions: dict[str, int], *, block: int | None = None) -> TensorSpec:
    shape = _resolve_shape(contract.shape_template, dimensions)
    return TensorSpec(
        name=_render_template(contract.name_template, block=block),
        shape=shape,
        dtype=contract.dtype,
        parameter_count=prod(shape) if shape else 0,
        role=contract.role,
        scale_axis=contract.scale_axis,
    )


def _module_from_contract(contract: ModuleContract, dimensions: dict[str, int], *, block: int | None = None) -> ModuleSpec:
    return ModuleSpec(
        name=_render_template(contract.name_template, block=block),
        module_type=contract.module_type,
        component=contract.component,
        tensors=[_tensor_from_contract(tensor, dimensions, block=block) for tensor in contract.tensors],
        quantizable=contract.quantizable,
        default_action=contract.default_action,
        notes=contract.notes,
    )


def _contract_summary(contract: AnimaModelContract) -> dict[str, Any]:
    transformer = contract.transformer
    dims = transformer.dimensions()
    return {
        "schema_version": contract.schema_version,
        "family": contract.family,
        "artifact_target": contract.artifact_target,
        "contract_mode": contract.contract_mode,
        "preferred_format": contract.preferred_format,
        "architecture": contract.metadata.get("architecture"),
        "transformer_prefix": transformer.block_prefix,
        "block_count": transformer.block_count,
        "model_channels": dims.get("X"),
        "num_heads": transformer.num_heads,
        "context_dim": dims.get("C"),
    }


def build_anima_graph_from_contract(
    contract: AnimaModelContract,
    source: ModelSource,
    *,
    artifact_metadata: dict[str, Any] | None = None,
) -> ModelGraph:
    dimensions = contract.transformer.dimensions()
    modules: list[ModuleSpec] = []
    modules.extend(_module_from_contract(module, dimensions) for module in contract.transformer.pre_modules)
    for block in range(contract.transformer.block_count):
        modules.extend(_module_from_contract(module, dimensions, block=block) for module in contract.transformer.block_modules)
    modules.extend(_module_from_contract(module, dimensions) for module in contract.transformer.post_modules)
    modules.extend(_module_from_contract(module, dimensions) for module in contract.extra_components)

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


def summarize_anima_graph(graph: ModelGraph, adapter: str) -> ModelInspection:
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
