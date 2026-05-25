"""Safetensors payload writer for selected FP8 tensors."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from comfy_quants.backends.safetensors_source import SafetensorsTensorSource
from comfy_quants.backends.torch_ref import quantize_tensor_fp8
from comfy_quants.core.errors import PayloadWriteError
from comfy_quants.formats.fp8_common import get_fp8_runtime_spec
from comfy_quants.utils.hashing import hash_file


def _require_safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - exercised only without package deps
        raise PayloadWriteError("safetensors is required for tensor payload writing") from exc
    return safe_open, save_file


@dataclass
class PayloadWriteReport:
    """Summary of tensor payload files written for an artifact."""

    source_checkpoint: str
    artifact_dir: str
    quantized_tensor_count: int
    weight_payload_path: str
    scale_payload_path: str
    schema_version: str = "payload_write_report.v1"
    status: str = "payload_written"
    source_format: str = "safetensors"
    target_dtype: str = "fp8_e4m3"
    scale_dtype: str = "fp32"
    storage_dtype: str = "uint8"
    source_layout: str = "single_file"
    source_tensor_count: int = 0
    source_file_count: int = 0
    selected_source_files: dict[str, int] = field(default_factory=dict)
    missing_tensor_count: int = 0
    missing_tensors: list[str] = field(default_factory=list)
    written_files: list[dict[str, Any]] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _artifact_path(artifact_dir: Path, relative_path: str) -> Path:
    rel = PurePosixPath(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise PayloadWriteError(f"artifact-relative path is invalid: {relative_path}")
    if not rel.parts:
        raise PayloadWriteError("artifact-relative path is empty")
    return artifact_dir.joinpath(*rel.parts)


def _shape_list(tensor: Any) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def _validate_tensor_row(row: dict[str, Any], *, target_dtype: str) -> None:
    if row.get("quant_dtype") != target_dtype:
        raise PayloadWriteError(f"unsupported quant dtype for payload writing: {row.get('quant_dtype')}")
    if row.get("storage_dtype") != "uint8":
        raise PayloadWriteError(f"unsupported storage dtype for payload writing: {row.get('storage_dtype')}")
    if not isinstance(row.get("payload"), dict):
        raise PayloadWriteError(f"tensor row missing payload metadata: {row.get('name')}")
    if not isinstance(row.get("scale"), dict):
        raise PayloadWriteError(f"tensor row missing scale metadata: {row.get('name')}")


def _file_record(path: Path, rel_path: str, kind: str, tensor_count: int) -> dict[str, Any]:
    return {
        "path": rel_path,
        "kind": kind,
        "state": "written",
        "tensor_count": tensor_count,
        "bytes": path.stat().st_size,
        "hash": hash_file(path),
    }


def _ensure_unique(targets: set[str], target: str, *, target_kind: str) -> None:
    if target in targets:
        raise PayloadWriteError(f"duplicate {target_kind} tensor name in payload plan: {target}")
    targets.add(target)


def _source_name(row: dict[str, Any]) -> str:
    name = row.get("source_name") or row.get("name")
    if not isinstance(name, str) or not name:
        raise PayloadWriteError(f"tensor row has no source name: {row.get('name')}")
    return name


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


def write_fp8_payload_from_safetensors(
    *,
    source_checkpoint: str | Path,
    artifact_dir: str | Path,
    tensor_index: dict[str, Any],
    target_dtype: str | None = None,
    strict: bool = True,
    device: str = "cpu",
) -> PayloadWriteReport:
    """Write selected FP8 payload and FP32 scale safetensors files."""
    safe_open, save_file = _require_safetensors()
    source = SafetensorsTensorSource.from_path(source_checkpoint)
    artifact_path = Path(os.path.expandvars(str(artifact_dir))).expanduser()
    resolved_target_dtype = _resolve_target_dtype(tensor_index, target_dtype)

    rows = list(tensor_index.get("tensors") or [])
    for row in rows:
        _validate_tensor_row(row, target_dtype=resolved_target_dtype)

    selected_names = [_source_name(row) for row in rows]
    missing = source.missing_tensors(selected_names)
    if missing and strict:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise PayloadWriteError(f"source checkpoint is missing selected tensors: {preview}{suffix}")

    payload_by_file: dict[str, dict[str, Any]] = {}
    scale_by_file: dict[str, dict[str, Any]] = {}
    payload_targets: set[str] = set()
    scale_targets: set[str] = set()
    quantized_count = 0

    for source_file, file_rows in source.group_rows_by_file(rows).items():
        if not source_file.is_file():
            raise PayloadWriteError(f"source tensor file is missing: {source_file}")
        with safe_open(str(source_file), framework="pt", device=device) as handle:
            available_in_file = set(handle.keys())
            for row in file_rows:
                source_name = _source_name(row)
                if source_name not in available_in_file:
                    raise PayloadWriteError(f"safetensors index maps {source_name} to {source_file}, but the tensor is absent from that file")
                source_tensor = handle.get_tensor(source_name)
                expected_shape = [int(dim) for dim in row.get("shape") or []]
                if _shape_list(source_tensor) != expected_shape:
                    raise PayloadWriteError(
                        f"source tensor shape mismatch for {source_name}: expected {expected_shape}, got {_shape_list(source_tensor)}"
                    )

                scale_meta = row["scale"]
                payload_meta = row["payload"]
                quantized = quantize_tensor_fp8(
                    source_tensor,
                    quant_dtype=resolved_target_dtype,
                    granularity=scale_meta["granularity"],
                    axis=scale_meta.get("axis"),
                    rounding=row.get("rounding", "nearest_even"),
                )

                payload_file = payload_meta["file"]
                scale_file = scale_meta["file"]
                payload_name = payload_meta.get("tensor_name") or row["name"]
                scale_name = scale_meta.get("tensor_name") or f"{row['name']}.scale"
                _ensure_unique(payload_targets, f"{payload_file}:{payload_name}", target_kind="payload")
                _ensure_unique(scale_targets, f"{scale_file}:{scale_name}", target_kind="scale")

                payload_by_file.setdefault(payload_file, {})[payload_name] = quantized.payload.cpu().contiguous()
                scale_by_file.setdefault(scale_file, {})[scale_name] = quantized.scale.cpu().contiguous()
                quantized_count += 1

    written_files: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}

    for rel_path, tensors in sorted(payload_by_file.items()):
        destination = _artifact_path(artifact_path, rel_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            tensors,
            str(destination),
            metadata={"target_dtype": resolved_target_dtype, "storage_dtype": "uint8", "payload_kind": "weight"},
        )
        record = _file_record(destination, rel_path, "fp8_weight_payload", len(tensors))
        written_files.append(record)
        hashes[rel_path] = record["hash"]

    for rel_path, tensors in sorted(scale_by_file.items()):
        destination = _artifact_path(artifact_path, rel_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            tensors,
            str(destination),
            metadata={"target_dtype": resolved_target_dtype, "scale_dtype": "fp32", "payload_kind": "scale"},
        )
        record = _file_record(destination, rel_path, "scale_payload", len(tensors))
        written_files.append(record)
        hashes[rel_path] = record["hash"]

    layout = tensor_index.get("payload_layout") or {}
    return PayloadWriteReport(
        source_checkpoint=str(source.source_path),
        artifact_dir=str(artifact_path),
        quantized_tensor_count=quantized_count,
        weight_payload_path=layout.get("weight_payload_path", "tensors/fp8_weights.safetensors"),
        scale_payload_path=layout.get("scale_payload_path", "scales/fp8_static_scales.safetensors"),
        target_dtype=resolved_target_dtype,
        source_layout=source.layout,
        source_tensor_count=len(source.file_map),
        source_file_count=len(set(source.file_map.values())),
        selected_source_files=source.selected_file_counts(selected_names),
        missing_tensor_count=len(missing),
        missing_tensors=missing,
        written_files=written_files,
        hashes=hashes,
    )


def write_fp8_e4m3_payload_from_safetensors(
    *,
    source_checkpoint: str | Path,
    artifact_dir: str | Path,
    tensor_index: dict[str, Any],
    strict: bool = True,
    device: str = "cpu",
) -> PayloadWriteReport:
    """Write selected FP8 E4M3 payload and FP32 scale safetensors files."""
    return write_fp8_payload_from_safetensors(
        source_checkpoint=source_checkpoint,
        artifact_dir=artifact_dir,
        tensor_index=tensor_index,
        target_dtype="fp8_e4m3",
        strict=strict,
        device=device,
    )


def write_fp8_e5m2_payload_from_safetensors(
    *,
    source_checkpoint: str | Path,
    artifact_dir: str | Path,
    tensor_index: dict[str, Any],
    strict: bool = True,
    device: str = "cpu",
) -> PayloadWriteReport:
    """Write selected FP8 E5M2 payload and FP32 scale safetensors files."""
    return write_fp8_payload_from_safetensors(
        source_checkpoint=source_checkpoint,
        artifact_dir=artifact_dir,
        tensor_index=tensor_index,
        target_dtype="fp8_e5m2",
        strict=strict,
        device=device,
    )
