"""Safetensors NVFP4 (FP4 E2M1 microscaling) inference-checkpoint exporter.

OFFLINE producer of **stock-ComfyUI-native** NVFP4 checkpoints. Mirrors the MXFP8
full-checkpoint writer but stores FP4-E2M1 weights packed 2-per-byte (``uint8``,
shape ``[out, in//2]``) with two-level scaling: a per-block-16 ``float8_e4m3fn``
``weight_scale`` (cuBLAS ``to_blocked`` swizzle, ``uint8``-viewable) and a per-tensor
``float32`` ``weight_scale_2``, plus a per-layer ``comfy_quant`` marker. Emits NO
``input_scale`` (activations are quantized dynamically at runtime).

Loaded by stock ComfyUI via ``QUANT_ALGOS["nvfp4"]`` + ``TensorCoreNVFP4Layout``
(``comfy/ops.py`` ``_load_from_state_dict`` nvfp4 branch). The nvfp4 tensor-core
matmul is Blackwell-gated; on other hardware ComfyUI silently dequantizes the weight
to the compute dtype (loads & correct, no speedup).

Block math + pack + the ``to_blocked`` swizzle live in
:mod:`comfy_quants.formats.nvfp4_blocked` (pure torch, bit-faithful to comfy-kitchen's
eager ``quantize_nvfp4``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# Reuse the FP8 writer's framework-agnostic helpers (source iteration, device
# resolution, config copy, metadata, progress); keeps the load-bearing FP8 writer
# untouched (same pattern as the MXFP8 / INT8 W8A8 writers).
from comfy_quants.backends.inference_model_export import (
    _emit_progress,
    _iter_source_files,
    _layer_name_from_weight,
    _metadata_value,
    _require_safetensors,
    _require_torch,
    _resolve_model_config_path,
    _resolve_torch_device,
    _shape_list,
    _source_name,
)
from comfy_quants.backends.safetensors_source import SafetensorsTensorSource
from comfy_quants.core.errors import PayloadWriteError
from comfy_quants.formats.nvfp4 import NVFP4_FORMAT_NAME, nvfp4_checkpoint_quant_config
from comfy_quants.formats.nvfp4_blocked import BLOCK_SIZE, quantize_nvfp4_block
from comfy_quants.utils.hashing import hash_file


@dataclass
class NvFp4CheckpointExportReport:
    """Summary of a full NVFP4 checkpoint export."""

    source_checkpoint: str
    output_checkpoint: str
    quantized_tensor_count: int
    copied_tensor_count: int
    output_tensor_count: int
    schema_version: str = "nvfp4_checkpoint_export_report.v1"
    status: str = "model_written"
    source_format: str = "safetensors"
    target_format: str = "safetensors"
    requested_device: str = "auto"
    execution_device: str = "cpu"
    output_tensor_device: str = "cpu"
    artifact_target: str = "comfyui_diffusion_model"
    target_dtype: str = NVFP4_FORMAT_NAME
    quant_storage_dtype: str = "uint8"
    scale_dtype: str = "float8_e4m3fn"
    per_tensor_scale_dtype: str = "float32"
    scale_granularity: str = "block"
    scale_axis: str | int | None = "in_features"
    block_size: int = BLOCK_SIZE
    source_layout: str = "single_file"
    source_tensor_count: int = 0
    source_file_count: int = 0
    selected_source_files: dict[str, int] = field(default_factory=dict)
    missing_tensor_count: int = 0
    missing_tensors: list[str] = field(default_factory=list)
    quant_metadata_tensor_count: int = 0
    scale_tensor_count: int = 0
    tensor_scale_count: int = 0
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


def _resolve_nvfp4_target_dtype(tensor_index: dict[str, Any], target_dtype: str | None = None) -> str:
    index_dtype = tensor_index.get("format", {}).get("name")
    resolved = target_dtype or index_dtype
    if resolved != NVFP4_FORMAT_NAME:
        raise PayloadWriteError(f"tensor index is not an {NVFP4_FORMAT_NAME} format: {resolved}")
    if index_dtype != NVFP4_FORMAT_NAME:
        raise PayloadWriteError(f"tensor index format {index_dtype} does not match requested target dtype {resolved}")
    return NVFP4_FORMAT_NAME


def _check_selected_rows(tensor_index: dict[str, Any], *, target_dtype: str) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in list(tensor_index.get("tensors") or []):
        if row.get("quant_dtype") != target_dtype:
            raise PayloadWriteError(f"unsupported quant dtype for NVFP4 export: {row.get('quant_dtype')}")
        if row.get("storage_dtype") != "uint8":
            raise PayloadWriteError(f"unsupported storage dtype for NVFP4 export: {row.get('storage_dtype')}")
        name = _source_name(row)
        if name in selected:
            raise PayloadWriteError(f"duplicate selected tensor in export plan: {name}")
        _layer_name_from_weight(name)
        selected[name] = row
    return selected


def _marker_tensor(conf: dict[str, Any], *, device: str):
    """Encode the comfy_quant marker like ComfyUI's save path (default json.dumps)."""
    torch = _require_torch()
    data = json.dumps(conf).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8, device=device)


def write_nvfp4_inference_checkpoint_from_safetensors(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    tensor_index: dict[str, Any],
    target_dtype: str | None = None,
    device: str = "auto",
    strict: bool = True,
    config_source: str | Path | None = None,
    copy_config: bool = True,
    hash_output: bool = False,
    metadata: dict[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> NvFp4CheckpointExportReport:
    """Write a full checkpoint with selected weights stored as NVFP4 tensors."""
    safe_open, save_file = _require_safetensors()
    torch = _require_torch()
    resolved_target_dtype = _resolve_nvfp4_target_dtype(tensor_index, target_dtype)

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
        block_size=BLOCK_SIZE,
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
                qweight, weight_scale, weight_scale_2 = quantize_nvfp4_block(tensor_for_quant)
                layer = _layer_name_from_weight(tensor_name)
                output_tensors[tensor_name] = qweight.detach().to(device="cpu").contiguous()
                output_tensors[f"{layer}.weight_scale"] = weight_scale.detach().to(device="cpu").contiguous()
                output_tensors[f"{layer}.weight_scale_2"] = weight_scale_2.detach().to(device="cpu").contiguous()
                output_tensors[f"{layer}.comfy_quant"] = _marker_tensor(nvfp4_checkpoint_quant_config(), device="cpu")
                dtype_counts["uint8"] = dtype_counts.get("uint8", 0) + 2  # packed weight + comfy_quant
                dtype_counts["float8_e4m3fn"] = dtype_counts.get("float8_e4m3fn", 0) + 1
                dtype_counts["float32"] = dtype_counts.get("float32", 0) + 1
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
                del qweight, weight_scale, weight_scale_2
                if execution_device_obj.type == "cuda":
                    del tensor_for_quant

    output_metadata = dict(metadata or {})
    output_metadata.update(
        {
            "artifact_target": "comfyui_diffusion_model",
            "artifact_contract": "qwen_image_nvfp4_inference_checkpoint.v1",
            "target_dtype": resolved_target_dtype,
            "quant_storage_dtype": "uint8",
            "scale_dtype": "float8_e4m3fn",
            "per_tensor_scale_dtype": "float32",
            "scale_granularity": "block",
            "block_size": BLOCK_SIZE,
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
                import shutil

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
            "kind": "nvfp4_inference_checkpoint",
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

    return NvFp4CheckpointExportReport(
        source_checkpoint=str(source.source_path),
        output_checkpoint=str(output_path),
        quantized_tensor_count=quantized,
        copied_tensor_count=copied,
        output_tensor_count=len(output_tensors),
        requested_device=requested_device,
        execution_device=execution_device,
        output_tensor_device="cpu",
        source_layout=source.layout,
        source_tensor_count=len(source.file_map),
        source_file_count=len(set(source.file_map.values())),
        selected_source_files=source.selected_file_counts(selected_names),
        missing_tensor_count=len(missing),
        missing_tensors=missing,
        quant_metadata_tensor_count=quantized,
        scale_tensor_count=quantized,
        tensor_scale_count=quantized,
        output_bytes=output_bytes,
        output_hash=output_hash,
        output_hash_state=output_hash_state,
        config_path=copied_config_path,
        cuda_max_memory_allocated_bytes=cuda_peak_allocated,
        cuda_max_memory_reserved_bytes=cuda_peak_reserved,
        dtype_counts=dict(sorted(dtype_counts.items())),
        written_files=written_files,
    )
