"""SVDQuant W4A4 kitchen tile-pack checkpoint exporter."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from comfy_quants.backends.safetensors_source import SafetensorsTensorSource
from comfy_quants.core.artifact import QuantArtifact
from comfy_quants.core.errors import PayloadWriteError
from comfy_quants.formats.kitchen_tilepack import (
    KITCHEN_GROUP_SIZE,
    KITCHEN_TILEPACK_LAYOUT_NAME,
    SVDQUANT_OPTIONAL_PARAM_KEYS,
    SVDQUANT_REQUIRED_PARAM_KEYS,
    SVDQUANT_W4A4_FORMAT_NAME,
    svdquant_prefixes,
    to_kitchen_tile_packed_params,
)
from comfy_quants.utils.hashing import hash_file


def _require_safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - exercised only without package deps
        raise PayloadWriteError("safetensors is required for INT4 checkpoint export") from exc
    return safe_open, save_file


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without package deps
        raise PayloadWriteError("torch is required for INT4 checkpoint export") from exc
    return torch


@dataclass
class Int4KitchenCheckpointExportReport:
    """Summary of a kitchen tile-packed SVDQuant W4A4 checkpoint export."""

    source_checkpoint: str
    output_checkpoint: str
    repacked_layer_count: int
    repacked_tensor_count: int
    copied_tensor_count: int
    output_tensor_count: int
    schema_version: str = "int4_kitchen_checkpoint_export_report.v1"
    status: str = "model_written"
    source_format: str = "safetensors"
    target_format: str = "safetensors"
    requested_device: str = "auto"
    execution_device: str = "cpu"
    output_tensor_device: str = "cpu"
    artifact_target: str = "comfyui_diffusion_model"
    target_dtype: str = SVDQUANT_W4A4_FORMAT_NAME
    storage_layout: str = KITCHEN_TILEPACK_LAYOUT_NAME
    weight_storage_dtype: str = "int8"
    group_size: int = KITCHEN_GROUP_SIZE
    source_layout: str = "single_file"
    source_tensor_count: int = 0
    source_file_count: int = 0
    selected_source_files: dict[str, int] = field(default_factory=dict)
    repacked_prefixes: list[str] = field(default_factory=list)
    output_bytes: int = 0
    output_hash: str = ""
    output_hash_state: str = "not_requested"
    cuda_max_memory_allocated_bytes: int | None = None
    cuda_max_memory_reserved_bytes: int | None = None
    dtype_counts: dict[str, int] = field(default_factory=dict)
    source_import: dict[str, Any] = field(default_factory=dict)
    written_files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_torch_device(device: str):
    torch = _require_torch()
    requested = str(device or "auto")
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(requested)
    if device_obj.type == "cuda":
        if not torch.cuda.is_available():
            raise PayloadWriteError(f"CUDA device requested but torch.cuda is not available: {device}")
        index = torch.cuda.current_device() if device_obj.index is None else int(device_obj.index)
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return device_obj


def _iter_source_files(source: SafetensorsTensorSource) -> list[tuple[Path, list[str]]]:
    by_ref: dict[str, list[str]] = {}
    for tensor_name, file_ref in source.file_map.items():
        by_ref.setdefault(file_ref, []).append(tensor_name)
    return [(source.base_dir.joinpath(*PurePosixPath(file_ref).parts), sorted(names)) for file_ref, names in sorted(by_ref.items())]


def _metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _dtype_name(tensor: Any) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _count_dtypes(tensors: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in tensors.values():
        dtype = _dtype_name(tensor)
        counts[dtype] = counts.get(dtype, 0) + 1
    return dict(sorted(counts.items()))


def _emit_progress(progress: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if progress is not None:
        progress(event)


def _layer_param_name(prefix: str, param_key: str) -> str:
    return f"{prefix}.{param_key}"


def _layer_params(tensors: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    missing = [key for key in SVDQUANT_REQUIRED_PARAM_KEYS if _layer_param_name(prefix, key) not in tensors]
    if missing:
        raise PayloadWriteError(f"SVDQuant layer {prefix} is missing required tensors: {', '.join(missing)}")
    params = {key: tensors[_layer_param_name(prefix, key)] for key in SVDQUANT_REQUIRED_PARAM_KEYS}
    for key in SVDQUANT_OPTIONAL_PARAM_KEYS:
        name = _layer_param_name(prefix, key)
        if name in tensors:
            params[key] = tensors[name]
    return params


def _move_params(params: Mapping[str, Any], *, device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, tensor in params.items():
        if key == "comfy_quant":
            moved[key] = tensor
        elif device.type == "cuda":
            moved[key] = tensor.to(device=device, non_blocking=True)
        else:
            moved[key] = tensor
    return moved


def repack_svdquant_w4a4_kitchen_state_dict(
    tensors: Mapping[str, Any],
    *,
    device: str = "auto",
    require_svdquant: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[str], int, str, int | None, int | None]:
    """Return a copy of a state dict with SVDQuant W4A4 layers tile-packed."""
    torch = _require_torch()
    requested_device = str(device or "auto")
    execution_device_obj = _resolve_torch_device(requested_device)
    execution_device = str(execution_device_obj)
    cuda_peak_allocated: int | None = None
    cuda_peak_reserved: int | None = None
    if execution_device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(execution_device_obj)

    output_tensors = {name: tensor.detach().contiguous() for name, tensor in tensors.items()}
    prefixes = svdquant_prefixes(set(output_tensors), output_tensors)
    if require_svdquant and not prefixes:
        raise PayloadWriteError("no SVDQuant W4A4 layers were found in the checkpoint")

    repacked_tensor_count = 0
    for index, prefix in enumerate(prefixes, start=1):
        _emit_progress(
            progress,
            stage="repack_layer",
            prefix=prefix,
            layer_index=index,
            layer_count=len(prefixes),
            execution_device=execution_device,
        )
        params = _move_params(_layer_params(output_tensors, prefix), device=execution_device_obj)
        try:
            packed = to_kitchen_tile_packed_params(params)
        except (KeyError, TypeError, ValueError) as exc:
            raise PayloadWriteError(f"failed to tile-pack SVDQuant layer {prefix}: {exc}") from exc

        for key, tensor in packed.items():
            output_tensors[_layer_param_name(prefix, key)] = tensor.detach().to(device="cpu").contiguous()
            repacked_tensor_count += 1
        if execution_device_obj.type == "cuda":
            del params, packed
            torch.cuda.empty_cache()

    if execution_device_obj.type == "cuda":
        torch.cuda.synchronize(execution_device_obj)
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated(execution_device_obj))
        cuda_peak_reserved = int(torch.cuda.max_memory_reserved(execution_device_obj))
        torch.cuda.empty_cache()

    return output_tensors, prefixes, repacked_tensor_count, execution_device, cuda_peak_allocated, cuda_peak_reserved


def write_svdquant_w4a4_kitchen_checkpoint(
    *,
    tensors: Mapping[str, Any],
    output_checkpoint: str | Path,
    source_checkpoint: str = "in_memory",
    source_layout: str = "in_memory",
    device: str = "auto",
    require_svdquant: bool = True,
    hash_output: bool = False,
    metadata: dict[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Int4KitchenCheckpointExportReport:
    """Write a single checkpoint after tile-packing SVDQuant W4A4 tensors."""
    _safe_open, save_file = _require_safetensors()
    output_path = Path(output_checkpoint).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    requested_device = str(device or "auto")

    _emit_progress(
        progress,
        stage="prepare",
        target_dtype=SVDQUANT_W4A4_FORMAT_NAME,
        storage_layout=KITCHEN_TILEPACK_LAYOUT_NAME,
        requested_device=requested_device,
        source_tensor_count=len(tensors),
    )
    output_tensors, prefixes, repacked_tensor_count, execution_device, cuda_peak_allocated, cuda_peak_reserved = (
        repack_svdquant_w4a4_kitchen_state_dict(
            tensors,
            device=requested_device,
            require_svdquant=require_svdquant,
            progress=progress,
        )
    )
    output_metadata = dict(metadata or {})
    output_metadata.update(
        {
            "artifact_target": "comfyui_diffusion_model",
            "artifact_contract": "svdquant_w4a4_kitchen_tilepack.v1",
            "target_dtype": SVDQUANT_W4A4_FORMAT_NAME,
            "storage_layout": KITCHEN_TILEPACK_LAYOUT_NAME,
            "weight_storage_dtype": "int8",
            "group_size": KITCHEN_GROUP_SIZE,
            "repacked_layer_count": len(prefixes),
        }
    )

    _emit_progress(
        progress,
        stage="save_checkpoint",
        output_checkpoint=str(output_path),
        output_tensor_count=len(output_tensors),
        output_tensor_device="cpu",
    )
    save_file(output_tensors, str(output_path), metadata={str(k): _metadata_value(v) for k, v in output_metadata.items()})

    output_hash = ""
    output_hash_state = "not_requested"
    if hash_output:
        _emit_progress(progress, stage="hash_checkpoint", output_checkpoint=str(output_path))
        output_hash = hash_file(output_path)
        output_hash_state = "written"
    output_bytes = output_path.stat().st_size
    copied_tensor_count = len(output_tensors) - repacked_tensor_count
    written_files = [
        {
            "path": str(output_path),
            "kind": "svdquant_w4a4_kitchen_tilepack_checkpoint",
            "state": "written",
            "tensor_count": len(output_tensors),
            "bytes": output_bytes,
            "hash": output_hash,
            "hash_state": output_hash_state,
        }
    ]

    return Int4KitchenCheckpointExportReport(
        source_checkpoint=source_checkpoint,
        output_checkpoint=str(output_path),
        repacked_layer_count=len(prefixes),
        repacked_tensor_count=repacked_tensor_count,
        copied_tensor_count=copied_tensor_count,
        output_tensor_count=len(output_tensors),
        requested_device=requested_device,
        execution_device=execution_device,
        source_layout=source_layout,
        source_tensor_count=len(tensors),
        source_file_count=0,
        repacked_prefixes=prefixes,
        output_bytes=output_bytes,
        output_hash=output_hash,
        output_hash_state=output_hash_state,
        cuda_max_memory_allocated_bytes=cuda_peak_allocated,
        cuda_max_memory_reserved_bytes=cuda_peak_reserved,
        dtype_counts=_count_dtypes(output_tensors),
        written_files=written_files,
    )


def write_svdquant_w4a4_kitchen_checkpoint_from_safetensors(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    device: str = "auto",
    require_svdquant: bool = True,
    hash_output: bool = False,
    metadata: dict[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Int4KitchenCheckpointExportReport:
    """Read local safetensors and write a kitchen tile-packed SVDQuant W4A4 checkpoint."""
    safe_open, _save_file = _require_safetensors()
    source = SafetensorsTensorSource.from_path(source_checkpoint)
    output_path = Path(output_checkpoint).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_files = _iter_source_files(source)
    output_resolved = output_path.resolve(strict=False)
    source_file_paths = {path.resolve(strict=False) for path, _names in source_files}
    if output_resolved in source_file_paths:
        raise PayloadWriteError(f"output checkpoint must not overwrite a source tensor file: {output_path}")

    _emit_progress(
        progress,
        stage="read_prepare",
        source_checkpoint=str(source.source_path),
        source_file_count=len(source_files),
        source_tensor_count=len(source.file_map),
    )
    tensors: dict[str, Any] = {}
    for source_file_index, (source_file, tensor_names) in enumerate(source_files, start=1):
        if not source_file.is_file():
            raise PayloadWriteError(f"source tensor file is missing: {source_file}")
        _emit_progress(
            progress,
            stage="read_source_file",
            source_file=str(source_file),
            source_file_index=source_file_index,
            source_file_count=len(source_files),
            tensor_count=len(tensor_names),
        )
        with safe_open(str(source_file), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for tensor_name in tensor_names:
                if tensor_name not in available:
                    raise PayloadWriteError(f"safetensors index maps {tensor_name} to {source_file}, but the tensor is absent from that file")
                tensor = handle.get_tensor(tensor_name)
                tensors[tensor_name] = tensor.detach().contiguous()

    report = write_svdquant_w4a4_kitchen_checkpoint(
        tensors=tensors,
        output_checkpoint=output_path,
        source_checkpoint=str(source.source_path),
        source_layout=source.layout,
        device=device,
        require_svdquant=require_svdquant,
        hash_output=hash_output,
        metadata=metadata,
        progress=progress,
    )
    report.source_tensor_count = len(source.file_map)
    report.source_file_count = len(set(source.file_map.values()))
    report.selected_source_files = source.selected_file_counts(
        _layer_param_name(prefix, key)
        for prefix in report.repacked_prefixes
        for key in (*SVDQUANT_REQUIRED_PARAM_KEYS, *SVDQUANT_OPTIONAL_PARAM_KEYS)
    )
    report.dtype_counts = _count_dtypes(tensors) if report.repacked_layer_count == 0 else report.dtype_counts
    if report.output_tensor_count != len(tensors):
        raise PayloadWriteError(
            f"internal tensor count mismatch after export: source {len(tensors)}, output {report.output_tensor_count}"
        )
    return report


class Int4KitchenExportBackend:
    backend_name = "int4_kitchen_export"
    version = "0.1.0"

    def check_compatibility(self, artifact: QuantArtifact) -> dict:
        return {"backend": self.backend_name, "level": "layout_writer", "artifact_id": artifact.artifact_id}

    def export(self, artifact: QuantArtifact, output_dir: str) -> dict:
        return {"backend": self.backend_name, "output_dir": output_dir, "artifact_id": artifact.artifact_id}


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_backend(Int4KitchenExportBackend())
