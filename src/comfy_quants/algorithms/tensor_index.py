"""Build quantized tensor index metadata from model graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from typing import Any

from comfy_quants.core.artifact_layout import DEFAULT_ARTIFACT_PAYLOAD_LAYOUT
from comfy_quants.core.graph import ModelGraph, ModuleSpec, TensorSpec
from comfy_quants.core.policy import QuantPolicy
from comfy_quants.formats.quant_tensor import PayloadMetadata, QuantTensorMetadata, ScaleMetadata
from comfy_quants.formats.registry import get_format


@dataclass(frozen=True)
class TensorIndexOptions:
    """Options that turn selected graph tensors into artifact index rows."""

    algorithm: str
    algorithm_version: str
    target_dtype: str
    scale_granularity: str
    scale_axis: str | int | None
    scale_method: str
    rounding: str
    compatibility_level: str
    scale_block_size: int | None = None
    scale_dtype: str = "fp32"
    artifact_state: str = "metadata_only"
    tensor_payload_state: str = "pending_export"
    weight_payload_path: str = DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.weight_payload_path
    scale_payload_path: str = DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.scale_payload_path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def module_selected_by_policy(module: ModuleSpec, policy: QuantPolicy) -> bool:
    """Return whether a module is selected for quantized tensor emission."""
    if not module.quantizable:
        return False
    if module.component in policy.keep_components:
        return False
    if policy.include and not any(fnmatchcase(module.name, pattern) for pattern in policy.include):
        return False
    if any(fnmatchcase(module.name, pattern) for pattern in policy.exclude):
        return False
    return True


def _axis_index(axis: str | int | None, shape: list[int]) -> int | None:
    if axis is None:
        return None
    if isinstance(axis, int):
        index = axis + len(shape) if axis < 0 else axis
        if index < 0 or index >= len(shape):
            raise ValueError(f"scale axis {axis} is outside tensor rank {len(shape)}")
        return index
    if axis == "out_features":
        return 0
    if axis == "in_features":
        if len(shape) < 2:
            raise ValueError("in_features scale axis requires a rank-2 tensor")
        return 1
    raise ValueError(f"unsupported scale axis: {axis}")


def _scale_shape(tensor: TensorSpec, granularity: str, axis: str | int | None, block_size: int | None = None) -> list[int]:
    if granularity == "per_tensor":
        return [1]
    if granularity == "block":
        index = _axis_index(axis, tensor.shape)
        if index is None:
            raise ValueError("block scale requires an axis (the blocked dimension)")
        if not block_size or block_size <= 0:
            raise ValueError("block scale requires a positive block_size")
        # Logical pre-swizzle E8M0 grid: one scale per (row, block) along `axis`.
        # The on-disk weight_scale is the to_blocked swizzle of this grid (a writer detail).
        n_blocks = (tensor.shape[index] + block_size - 1) // block_size
        shape = [dim for i, dim in enumerate(tensor.shape) if i != index]
        return shape + [n_blocks]
    if granularity != "per_channel":
        raise ValueError(f"unsupported scale granularity: {granularity}")
    index = _axis_index(axis, tensor.shape)
    if index is None:
        raise ValueError("per_channel scale requires an axis")
    return [tensor.shape[index]]


def _tensor_metadata(
    *,
    module: ModuleSpec,
    tensor: TensorSpec,
    options: TensorIndexOptions,
    storage_dtype: str,
) -> QuantTensorMetadata:
    if options.scale_granularity == "per_tensor":
        axis = None
    else:
        axis = tensor.scale_axis if tensor.scale_axis is not None else options.scale_axis
    return QuantTensorMetadata(
        name=tensor.name,
        source_name=tensor.name,
        shape=list(tensor.shape),
        source_dtype=tensor.dtype,
        quant_dtype=options.target_dtype,
        storage_dtype=storage_dtype,
        algorithm=options.algorithm,
        scale=ScaleMetadata(
            dtype=options.scale_dtype,
            shape=_scale_shape(tensor, options.scale_granularity, axis, options.scale_block_size),
            granularity=options.scale_granularity,
            axis=axis,
            block_size=options.scale_block_size,
            file=options.scale_payload_path,
            tensor_name=f"{tensor.name}.scale",
        ),
        payload=PayloadMetadata(
            file=options.weight_payload_path,
            tensor_name=tensor.name,
            storage_dtype=storage_dtype,
        ),
        zero_point=None,
        rounding=options.rounding,
        fallback=False,
        compatibility_level=options.compatibility_level,
        metadata={
            "module_name": module.name,
            "component": module.component,
            "module_type": module.module_type,
            "source_role": tensor.role,
            "scale_method": options.scale_method,
        },
    )


def iter_quantized_tensors(graph: ModelGraph, policy: QuantPolicy) -> list[tuple[ModuleSpec, TensorSpec]]:
    """Return selected weight tensors in graph order."""
    selected: list[tuple[ModuleSpec, TensorSpec]] = []
    for module in graph.modules:
        if not module_selected_by_policy(module, policy):
            continue
        for tensor in module.tensors:
            if tensor.role == "weight":
                selected.append((module, tensor))
    return selected


def build_quant_tensor_index(graph: ModelGraph, policy: QuantPolicy, options: TensorIndexOptions) -> dict[str, Any]:
    """Build the artifact quant tensor index for a graph and policy."""
    fmt = get_format(options.target_dtype)
    selected = iter_quantized_tensors(graph, policy)
    tensors = [
        _tensor_metadata(
            module=module,
            tensor=tensor,
            options=options,
            storage_dtype=fmt.storage_dtype,
        ).to_dict()
        for module, tensor in selected
    ]
    modules = {module.name for module, _ in selected}
    index = {
        "schema_version": "quant_tensor_index.v1",
        "artifact_state": options.artifact_state,
        "tensor_payload_state": options.tensor_payload_state,
        "artifact_target": graph.metadata.get("artifact_target"),
        "contract_source": graph.metadata.get("contract_source"),
        "contract_mode": graph.metadata.get("contract_mode"),
        "contract_schema": graph.metadata.get("contract_schema"),
        "artifact_contract": graph.metadata.get("artifact_contract"),
        "graph_kind": graph.metadata.get("graph_kind"),
        "tensor_coverage": graph.metadata.get("tensor_coverage"),
        "payload_layout": DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.to_dict(),
        "format": {
            "name": fmt.name,
            "storage_dtype": fmt.storage_dtype,
            "bits": fmt.bits,
            "category": fmt.category,
            "scale_required": fmt.scale_required,
            "scale_granularity": options.scale_granularity,
            "scale_axis": options.scale_axis,
            "scale_method": options.scale_method,
            "rounding": options.rounding,
        },
        "selection": {
            "algorithm": options.algorithm,
            "algorithm_version": options.algorithm_version,
            "target_dtype": options.target_dtype,
            "quantized_module_count": len(modules),
            "quantized_tensor_count": len(tensors),
            "source_tensor_roles": ["weight"],
            "kept_source_tensor_roles": ["bias"],
            "include": list(policy.include),
            "exclude": list(policy.exclude),
            "keep_components": list(policy.keep_components),
        },
        "tensors": tensors,
    }
    reference_image_mode = graph.metadata.get("reference_image_mode")
    if reference_image_mode is not None:
        index["reference_image_mode"] = reference_image_mode
    return index
