"""Safetensors inference checkpoint exporter."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from comfy_quants.backends.safetensors_source import SafetensorsTensorSource
from comfy_quants.core.errors import PayloadWriteError
from comfy_quants.formats.fp8_common import (
    fp8_checkpoint_quant_config,
    fp8_inference_artifact_contract,
    fp8_inference_checkpoint_kind,
    get_fp8_runtime_spec,
)
from comfy_quants.utils.hashing import hash_file


def _require_safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - exercised only without package deps
        raise PayloadWriteError("safetensors is required for inference checkpoint export") from exc
    return safe_open, save_file


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without package deps
        raise PayloadWriteError("torch is required for inference checkpoint export") from exc
    return torch


def _torch_fp8_dtype(torch, target_dtype: str):
    spec = get_fp8_runtime_spec(target_dtype)
    if not hasattr(torch, spec.torch_dtype_name):
        raise PayloadWriteError(f"torch.{spec.torch_dtype_name} is required for {spec.name} checkpoint export")
    return getattr(torch, spec.torch_dtype_name)


@dataclass
class InferenceCheckpointExportReport:
    """Summary of a full safetensors checkpoint export."""

    source_checkpoint: str
    output_checkpoint: str
    quantized_tensor_count: int
    copied_tensor_count: int
    output_tensor_count: int
    schema_version: str = "inference_checkpoint_export_report.v1"
    status: str = "model_written"
    source_format: str = "safetensors"
    target_format: str = "safetensors"
    requested_device: str = "auto"
    execution_device: str = "cpu"
    output_tensor_device: str = "cpu"
    artifact_target: str = "comfyui_diffusion_model"
    target_dtype: str = "fp8_e4m3"
    quant_storage_dtype: str = "float8_e4m3fn"
    scale_dtype: str = "fp32"
    scale_granularity: str = "per_tensor"
    scale_axis: str | int | None = None
    source_layout: str = "single_file"
    source_tensor_count: int = 0
    source_file_count: int = 0
    selected_source_files: dict[str, int] = field(default_factory=dict)
    missing_tensor_count: int = 0
    missing_tensors: list[str] = field(default_factory=list)
    quant_metadata_tensor_count: int = 0
    scale_tensor_count: int = 0
    input_scale_tensor_count: int = 0
    output_bytes: int = 0
    output_hash: str = ""
    output_hash_state: str = "not_requested"
    config_path: str | None = None
    cuda_max_memory_allocated_bytes: int | None = None
    cuda_max_memory_reserved_bytes: int | None = None
    dtype_counts: dict[str, int] = field(default_factory=dict)
    written_files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_name(row: dict[str, Any]) -> str:
    name = row.get("source_name") or row.get("name")
    if not isinstance(name, str) or not name:
        raise PayloadWriteError(f"tensor row has no source name: {row.get('name')}")
    return name


def _shape_list(tensor: Any) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def _layer_name_from_weight(weight_name: str) -> str:
    suffix = ".weight"
    if not weight_name.endswith(suffix):
        raise PayloadWriteError(f"selected tensor is not a module weight: {weight_name}")
    return weight_name[: -len(suffix)]


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


def _json_bytes_tensor(value: dict[str, Any], *, device: str):
    torch = _require_torch()
    data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8, device=device)


def _unit_input_scale_tensor(*, device: str):
    torch = _require_torch()
    return torch.ones((), dtype=torch.float32, device=device)


def _resolve_target_dtype(tensor_index: dict[str, Any], target_dtype: str | None = None) -> str:
    index_dtype = tensor_index.get("format", {}).get("name")
    resolved = target_dtype or index_dtype
    if not isinstance(resolved, str) or not resolved:
        raise PayloadWriteError("tensor index is missing a target FP8 format")
    try:
        spec = get_fp8_runtime_spec(resolved)
    except KeyError as exc:
        raise PayloadWriteError(str(exc)) from exc
    if index_dtype != spec.name:
        raise PayloadWriteError(f"tensor index format {index_dtype} does not match requested target dtype {spec.name}")
    return spec.name


def _check_selected_rows(tensor_index: dict[str, Any], *, target_dtype: str) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in list(tensor_index.get("tensors") or []):
        if row.get("quant_dtype") != target_dtype:
            raise PayloadWriteError(f"unsupported quant dtype for model export: {row.get('quant_dtype')}")
        if row.get("storage_dtype") != "uint8":
            raise PayloadWriteError(f"unsupported storage dtype for model export: {row.get('storage_dtype')}")
        name = _source_name(row)
        if name in selected:
            raise PayloadWriteError(f"duplicate selected tensor in export plan: {name}")
        _layer_name_from_weight(name)
        selected[name] = row
    return selected


def _quantize_float8(tensor, *, target_dtype: str, scale_granularity: str, scale_axis: str | int | None, rounding: str):
    if rounding != "nearest_even":
        raise PayloadWriteError(f"unsupported rounding mode: {rounding}")
    torch = _require_torch()
    spec = get_fp8_runtime_spec(target_dtype)
    fp8_dtype = _torch_fp8_dtype(torch, spec.name)
    if scale_granularity != "per_tensor":
        raise PayloadWriteError("inference checkpoint export currently requires per_tensor FP8 weight scales")
    if scale_axis is not None:
        raise PayloadWriteError("inference checkpoint export requires scale_axis=None for per_tensor FP8 weight scales")
    values = tensor.detach().to(torch.float32)
    amax = values.abs().max()
    scale = torch.where(
        amax > 0,
        torch.clamp(amax / spec.max_finite, min=1.0e-12),
        torch.ones_like(amax, dtype=torch.float32),
    )
    normalized = (values / scale.to(torch.float32)).clamp(-spec.max_finite, spec.max_finite)
    return normalized.to(fp8_dtype).contiguous(), scale.to(torch.float32).contiguous()


def _iter_source_files(source: SafetensorsTensorSource) -> list[tuple[Path, list[str]]]:
    by_ref: dict[str, list[str]] = {}
    for tensor_name, file_ref in source.file_map.items():
        by_ref.setdefault(file_ref, []).append(tensor_name)
    return [(source.base_dir.joinpath(*PurePosixPath(file_ref).parts), names) for file_ref, names in sorted(by_ref.items())]


def _resolve_model_config_path(source: SafetensorsTensorSource, config_source: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if config_source is not None:
        path = Path(os.path.expandvars(str(config_source))).expanduser()
        if path.is_dir():
            candidates.append(path / "config.json")
        elif path.is_file():
            if path.name == "config.json":
                candidates.append(path)
            candidates.append(path.parent / "config.json")

    if source.source_path.is_dir():
        candidates.append(source.source_path / "config.json")
    if source.index_path is not None:
        candidates.append(source.index_path.parent / "config.json")
    candidates.append(source.base_dir / "config.json")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def _metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _requires_index_timestep_zero_marker(tensor_index: dict[str, Any], metadata: dict[str, Any] | None) -> bool:
    if tensor_index.get("reference_image_mode") == "index_timestep_zero":
        return True
    if metadata and metadata.get("reference_image_mode") == "index_timestep_zero":
        return True
    if metadata and metadata.get("model_family") == "qwen_image_edit" and "2511" in str(metadata.get("model_id", "")):
        return True
    return False


def _emit_progress(progress: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if progress is not None:
        progress(event)


def write_fp8_inference_checkpoint_from_safetensors(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    tensor_index: dict[str, Any],
    target_dtype: str | None = None,
    scale_granularity: str = "per_tensor",
    scale_axis: str | int | None = None,
    device: str = "auto",
    strict: bool = True,
    config_source: str | Path | None = None,
    copy_config: bool = True,
    hash_output: bool = False,
    metadata: dict[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> InferenceCheckpointExportReport:
    """Write a full checkpoint with selected weights stored as supported FP8 tensors."""
    safe_open, save_file = _require_safetensors()
    torch = _require_torch()
    resolved_target_dtype = _resolve_target_dtype(tensor_index, target_dtype)
    spec = get_fp8_runtime_spec(resolved_target_dtype)
    _torch_fp8_dtype(torch, resolved_target_dtype)

    requested_device = str(device or "auto")
    execution_device_obj = _resolve_torch_device(requested_device)
    execution_device = str(execution_device_obj)
    cuda_peak_allocated: int | None = None
    cuda_peak_reserved: int | None = None
    if execution_device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(execution_device_obj)
    source = SafetensorsTensorSource.from_path(source_checkpoint)
    output_path = Path(output_checkpoint).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = _check_selected_rows(tensor_index, target_dtype=resolved_target_dtype)
    selected_names = list(selected)
    missing = source.missing_tensors(selected_names)
    if missing and strict:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise PayloadWriteError(f"source checkpoint is missing selected tensors: {preview}{suffix}")

    output_tensors: dict[str, Any] = {}
    dtype_counts: dict[str, int] = {}
    copied = 0
    quantized = 0
    source_files = _iter_source_files(source)
    output_resolved = output_path.resolve(strict=False)
    source_file_paths = {path.resolve(strict=False) for path, _names in source_files}
    if output_resolved in source_file_paths:
        raise PayloadWriteError(f"output checkpoint must not overwrite a source tensor file: {output_path}")

    _emit_progress(
        progress,
        stage="prepare",
        target_dtype=resolved_target_dtype,
        requested_device=requested_device,
        execution_device=execution_device,
        source_file_count=len(source_files),
        selected_tensor_count=len(selected_names),
    )
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
                row = selected.get(tensor_name)
                if row is None:
                    stored = tensor.detach().contiguous()
                    output_tensors[tensor_name] = stored
                    dtype = str(stored.dtype).replace("torch.", "")
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                    copied += 1
                    continue

                expected_shape = [int(dim) for dim in row.get("shape") or []]
                if _shape_list(tensor) != expected_shape:
                    raise PayloadWriteError(f"source tensor shape mismatch for {tensor_name}: expected {expected_shape}, got {_shape_list(tensor)}")

                if execution_device_obj.type == "cuda":
                    tensor_for_quant = tensor.to(device=execution_device_obj, non_blocking=True)
                else:
                    tensor_for_quant = tensor
                qweight, scale = _quantize_float8(
                    tensor_for_quant,
                    target_dtype=resolved_target_dtype,
                    scale_granularity=scale_granularity,
                    scale_axis=scale_axis,
                    rounding=row.get("rounding", "nearest_even"),
                )
                layer = _layer_name_from_weight(tensor_name)
                output_tensors[tensor_name] = qweight.detach().to(device="cpu").contiguous()
                output_tensors[f"{layer}.weight_scale"] = scale.detach().to(device="cpu").contiguous()
                output_tensors[f"{layer}.input_scale"] = _unit_input_scale_tensor(device="cpu")
                output_tensors[f"{layer}.comfy_quant"] = _json_bytes_tensor(fp8_checkpoint_quant_config(resolved_target_dtype), device="cpu")
                dtype_counts[spec.torch_dtype_name] = dtype_counts.get(spec.torch_dtype_name, 0) + 1
                dtype_counts["float32"] = dtype_counts.get("float32", 0) + 2
                dtype_counts["uint8"] = dtype_counts.get("uint8", 0) + 1
                quantized += 1
                _emit_progress(
                    progress,
                    stage="quantize_tensor",
                    target_dtype=resolved_target_dtype,
                    tensor_name=tensor_name,
                    quantized_tensor_count=quantized,
                    selected_tensor_count=len(selected_names),
                    execution_device=execution_device,
                )
                del qweight, scale
                if execution_device_obj.type == "cuda":
                    del tensor_for_quant

    if _requires_index_timestep_zero_marker(tensor_index, metadata) and "__index_timestep_zero__" not in output_tensors:
        output_tensors["__index_timestep_zero__"] = torch.empty((0,), dtype=torch.float32, device="cpu")
        dtype_counts["float32"] = dtype_counts.get("float32", 0) + 1

    output_metadata = dict(metadata or {})
    output_metadata.update(
        {
            "artifact_target": "comfyui_diffusion_model",
            "artifact_contract": fp8_inference_artifact_contract(resolved_target_dtype),
            "target_dtype": resolved_target_dtype,
            "quant_storage_dtype": spec.torch_dtype_name,
            "scale_granularity": scale_granularity,
            "quantized_tensor_count": quantized,
        }
    )
    if execution_device_obj.type == "cuda":
        torch.cuda.synchronize(execution_device_obj)
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated(execution_device_obj))
        cuda_peak_reserved = int(torch.cuda.max_memory_reserved(execution_device_obj))
        torch.cuda.empty_cache()
    _emit_progress(
        progress,
        stage="save_checkpoint",
        output_checkpoint=str(output_path),
        output_tensor_count=len(output_tensors),
        output_tensor_device="cpu",
    )
    save_file(output_tensors, str(output_path), metadata={str(k): _metadata_value(v) for k, v in output_metadata.items()})

    copied_config_path: str | None = None
    if copy_config:
        config_path = _resolve_model_config_path(source, config_source)
        if config_path is not None:
            destination = output_path.parent / "config.json"
            if config_path.resolve() != destination.resolve():
                shutil.copy2(config_path, destination)
            copied_config_path = str(destination)

    output_hash = ""
    output_hash_state = "not_requested"
    if hash_output:
        _emit_progress(progress, stage="hash_checkpoint", output_checkpoint=str(output_path))
        output_hash = hash_file(output_path)
        output_hash_state = "written"
    output_bytes = output_path.stat().st_size
    written_files = [
        {
            "path": str(output_path),
            "kind": fp8_inference_checkpoint_kind(resolved_target_dtype),
            "state": "written",
            "tensor_count": len(output_tensors),
            "bytes": output_bytes,
            "hash": output_hash,
            "hash_state": output_hash_state,
        }
    ]
    if copied_config_path is not None:
        config_file = Path(copied_config_path)
        written_files.append(
            {
                "path": str(config_file),
                "kind": "model_config",
                "state": "copied",
                "bytes": config_file.stat().st_size,
                "hash": hash_file(config_file),
            }
        )

    return InferenceCheckpointExportReport(
        source_checkpoint=str(source.source_path),
        output_checkpoint=str(output_path),
        quantized_tensor_count=quantized,
        copied_tensor_count=copied,
        output_tensor_count=len(output_tensors),
        requested_device=requested_device,
        execution_device=execution_device,
        output_tensor_device="cpu",
        target_dtype=resolved_target_dtype,
        quant_storage_dtype=spec.torch_dtype_name,
        scale_granularity=scale_granularity,
        scale_axis=scale_axis,
        source_layout=source.layout,
        source_tensor_count=len(source.file_map),
        source_file_count=len(set(source.file_map.values())),
        selected_source_files=source.selected_file_counts(selected_names),
        missing_tensor_count=len(missing),
        missing_tensors=missing,
        quant_metadata_tensor_count=quantized,
        scale_tensor_count=quantized * 2,
        input_scale_tensor_count=quantized,
        output_bytes=output_bytes,
        output_hash=output_hash,
        output_hash_state=output_hash_state,
        config_path=copied_config_path,
        cuda_max_memory_allocated_bytes=cuda_peak_allocated,
        cuda_max_memory_reserved_bytes=cuda_peak_reserved,
        dtype_counts=dict(sorted(dtype_counts.items())),
        written_files=written_files,
    )


def write_fp8_e4m3_inference_checkpoint_from_safetensors(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    tensor_index: dict[str, Any],
    scale_granularity: str = "per_tensor",
    scale_axis: str | int | None = None,
    device: str = "auto",
    strict: bool = True,
    config_source: str | Path | None = None,
    copy_config: bool = True,
    hash_output: bool = False,
    metadata: dict[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> InferenceCheckpointExportReport:
    """Write a full checkpoint with selected weights stored as FP8 E4M3 tensors."""
    return write_fp8_inference_checkpoint_from_safetensors(
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        tensor_index=tensor_index,
        target_dtype="fp8_e4m3",
        scale_granularity=scale_granularity,
        scale_axis=scale_axis,
        device=device,
        strict=strict,
        config_source=config_source,
        copy_config=copy_config,
        hash_output=hash_output,
        metadata=metadata,
        progress=progress,
    )


def write_fp8_e5m2_inference_checkpoint_from_safetensors(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    tensor_index: dict[str, Any],
    scale_granularity: str = "per_tensor",
    scale_axis: str | int | None = None,
    device: str = "auto",
    strict: bool = True,
    config_source: str | Path | None = None,
    copy_config: bool = True,
    hash_output: bool = False,
    metadata: dict[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> InferenceCheckpointExportReport:
    """Write a full checkpoint with selected weights stored as FP8 E5M2 tensors."""
    return write_fp8_inference_checkpoint_from_safetensors(
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        tensor_index=tensor_index,
        target_dtype="fp8_e5m2",
        scale_granularity=scale_granularity,
        scale_axis=scale_axis,
        device=device,
        strict=strict,
        config_source=config_source,
        copy_config=copy_config,
        hash_output=hash_output,
        metadata=metadata,
        progress=progress,
    )
