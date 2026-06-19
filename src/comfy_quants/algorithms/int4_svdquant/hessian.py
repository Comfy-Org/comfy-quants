"""GPTQ Hessian artifact reduction for INT4 SVDQuant calibration.

The reducer consumes safetensors activation dumps produced by an external model
runtime and writes portable per-layer Hessian artifacts.  It deliberately does
not import any model runtime; the only contract is the activation sample
manifest shared with the activation statistics reducer.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from comfy_quants.algorithms.int4_svdquant.calibration import ActivationSampleRef
from comfy_quants.algorithms.int4_svdquant.gptq import GptqHessianStats, build_gptq_hessian_from_activations
from comfy_quants.utils.jsonio import write_json


GPTQ_HESSIAN_STATS_SCHEMA_VERSION = "int4_gptq_hessian_stats.v1"
GPTQ_HESSIAN_REDUCE_REPORT_SCHEMA_VERSION = "int4_gptq_hessian_reduce_report.v1"
DEFAULT_GPTQ_HESSIAN_MANIFEST = "int4_gptq_hessian_stats.json"
DEFAULT_GPTQ_HESSIAN_TENSOR_DIR = "gptq_hessians"

_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _require_safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("safetensors is required for GPTQ Hessian artifact reduction") from exc
    return safe_open, save_file


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("torch is required for GPTQ Hessian artifact reduction") from exc
    return torch


@dataclass(frozen=True)
class GptqHessianLayerRecord:
    """Manifest record for one layer's precomputed GPTQ Hessian tensor."""

    layer_name: str
    file_path: str
    tensor_name: str
    channel_count: int
    sample_count: int
    row_count: int
    normalization_count: int
    channel_dim: int
    dtype: str = "float32"
    shape: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GptqHessianReduceReport:
    """Summary for an activation-dump-to-Hessian reduction run."""

    status: str
    samples: str
    output_dir: str
    manifest_path: str
    hessian_tensor_dir: str
    layer_count: int
    sample_ref_count: int
    row_count: int
    schema_version: str = GPTQ_HESSIAN_REDUCE_REPORT_SCHEMA_VERSION
    written_files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_torch_device(device: str | None):
    torch = _require_torch()
    requested = str(device or "auto")
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(requested)
    if device_obj.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but torch.cuda is not available: {device}")
        index = torch.cuda.current_device() if device_obj.index is None else int(device_obj.index)
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return device_obj


def _safe_relative_dir(value: str) -> str:
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"GPTQ Hessian relative directory is invalid: {value}")
    return str(rel)


def _safe_layer_stem(layer_name: str, *, used: set[str]) -> str:
    base = _SAFE_STEM_RE.sub("_", layer_name.strip()).strip("._-")
    if not base:
        base = "layer"
    if len(base) > 160:
        digest = hashlib.sha1(layer_name.encode("utf-8")).hexdigest()[:12]
        base = f"{base[:140]}-{digest}"
    stem = base
    if stem in used:
        digest = hashlib.sha1(layer_name.encode("utf-8")).hexdigest()[:12]
        stem = f"{base}-{digest}"
    collision_index = 1
    while stem in used:
        collision_index += 1
        stem = f"{base}-{collision_index}"
    used.add(stem)
    return stem


def _file_record(path: Path, *, kind: str, include_bytes: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {"kind": kind, "path": str(path)}
    if include_bytes:
        record["bytes"] = int(path.stat().st_size)
    return record


def _dtype_name(tensor: Any) -> str:
    return str(getattr(tensor, "dtype", "unknown")).replace("torch.", "")


def _shape_of(tensor: Any) -> list[int]:
    return [int(dim) for dim in getattr(tensor, "shape", ())]


def _resolve_ref_file(ref: ActivationSampleRef, *, input_root: str | Path | None) -> Path:
    path = Path(ref.file_path).expanduser()
    if path.is_absolute() or input_root is None:
        return path
    return Path(input_root).expanduser() / path


def _resolve_manifest_relative_file(manifest_path: Path, file_path: str) -> Path:
    rel = PurePosixPath(file_path)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"GPTQ Hessian tensor path must be a safe manifest-relative path: {file_path}")
    return manifest_path.parent.joinpath(*rel.parts)


def _validate_layer_refs(layer_name: str, refs: list[ActivationSampleRef]) -> int:
    if not refs:
        raise ValueError(f"no activation samples for layer {layer_name}")
    channel_dims = {int(ref.channel_dim) for ref in refs}
    if len(channel_dims) != 1:
        raise ValueError(f"activation samples for {layer_name} use multiple channel_dim values: {sorted(channel_dims)}")
    return next(iter(channel_dims))


def reduce_gptq_hessians_from_safetensors(
    sample_refs: Iterable[ActivationSampleRef],
    *,
    output_dir: str | Path,
    input_root: str | Path | None = None,
    samples_path: str | Path | None = None,
    hessian_tensor_dir: str = DEFAULT_GPTQ_HESSIAN_TENSOR_DIR,
    hessian_block_size: int = 512,
    device: str | None = "auto",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> GptqHessianReduceReport:
    """Reduce captured activation tensors into per-layer GPTQ Hessian files.

    Each output Hessian is normalized by the total flattened row count using the
    same ``2 / n`` convention as the layer-level GPTQ helper.  The output
    manifest stores relative tensor file paths so the artifact directory can be
    moved as a unit.
    """
    safe_open, save_file = _require_safetensors()
    torch = _require_torch()
    refs = list(sample_refs)
    if not refs:
        raise ValueError("at least one activation sample reference is required")
    if int(hessian_block_size) == 0:
        raise ValueError("hessian_block_size must be positive or negative")

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    tensor_dir_rel = _safe_relative_dir(hessian_tensor_dir)
    tensor_dir_path = output_path.joinpath(*PurePosixPath(tensor_dir_rel).parts)
    tensor_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / DEFAULT_GPTQ_HESSIAN_MANIFEST
    execution_device = _resolve_torch_device(device)

    refs_by_layer: dict[str, list[ActivationSampleRef]] = defaultdict(list)
    for ref in refs:
        if not isinstance(ref.layer_name, str) or not ref.layer_name:
            raise ValueError("activation sample layer_name must be a non-empty string")
        refs_by_layer[ref.layer_name].append(ref)

    used_stems: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    written_files: list[dict[str, Any]] = []
    total_rows = 0
    layer_items = sorted(refs_by_layer.items())
    for layer_index, (layer_name, layer_refs) in enumerate(layer_items, start=1):
        channel_dim = _validate_layer_refs(layer_name, layer_refs)

        def _samples():
            for sample_index, ref in enumerate(layer_refs, start=1):
                file_path = _resolve_ref_file(ref, input_root=input_root)
                if progress is not None:
                    progress(
                        {
                            "stage": "reduce_gptq_hessian_sample",
                            "layer_name": layer_name,
                            "layer_index": layer_index,
                            "layer_count": len(layer_items),
                            "sample_index": sample_index,
                            "sample_count": len(layer_refs),
                            "file_path": str(file_path),
                            "tensor_name": ref.tensor_name,
                            "execution_device": str(execution_device),
                        }
                    )
                if not file_path.is_file():
                    raise ValueError(f"activation sample file is missing: {file_path}")
                with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                    if ref.tensor_name not in handle.keys():
                        raise ValueError(f"activation sample tensor {ref.tensor_name!r} is missing from {file_path}")
                    yield handle.get_tensor(ref.tensor_name)

        if progress is not None:
            progress(
                {
                    "stage": "reduce_gptq_hessian_layer",
                    "layer_name": layer_name,
                    "layer_index": layer_index,
                    "layer_count": len(layer_items),
                    "sample_count": len(layer_refs),
                    "execution_device": str(execution_device),
                }
            )
        stats: GptqHessianStats = build_gptq_hessian_from_activations(
            _samples(),
            channel_dim=channel_dim,
            hessian_block_size=int(hessian_block_size),
            device=execution_device,
            dtype=torch.float32,
        )
        hessian = stats.hessian.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(int(dim) for dim in hessian.shape) != (int(stats.channel_count), int(stats.channel_count)):
            raise ValueError(f"computed Hessian shape for {layer_name} does not match channel count {stats.channel_count}")
        if not bool(torch.isfinite(hessian).all().item()):
            raise ValueError(f"computed Hessian for {layer_name} contains NaN or Inf values")

        stem = _safe_layer_stem(layer_name, used=used_stems)
        tensor_rel = str(PurePosixPath(tensor_dir_rel) / f"{stem}.safetensors")
        tensor_path = output_path.joinpath(*PurePosixPath(tensor_rel).parts)
        metadata = {
            "schema_version": GPTQ_HESSIAN_STATS_SCHEMA_VERSION,
            "layer_name": layer_name,
            "tensor_name": "hessian",
            "normalization": "two_over_row_count",
        }
        save_file({"hessian": hessian}, str(tensor_path), metadata=metadata)
        written_files.append(_file_record(tensor_path, kind="gptq_hessian_tensor"))
        total_rows += int(stats.row_count)
        records[layer_name] = GptqHessianLayerRecord(
            layer_name=layer_name,
            file_path=tensor_rel,
            tensor_name="hessian",
            channel_count=int(stats.channel_count),
            sample_count=int(stats.sample_count),
            row_count=int(stats.row_count),
            normalization_count=int(stats.normalization_count),
            channel_dim=channel_dim,
            dtype=_dtype_name(hessian),
            shape=_shape_of(hessian),
        ).to_dict()

        if execution_device.type == "cuda":
            del hessian, stats
            torch.cuda.empty_cache()

    manifest = {
        "schema_version": GPTQ_HESSIAN_STATS_SCHEMA_VERSION,
        "normalization": "two_over_row_count",
        "hessian_tensor_dir": tensor_dir_rel,
        "layer_count": len(records),
        "sample_ref_count": len(refs),
        "row_count": total_rows,
        "layers": records,
    }
    write_json(manifest_path, manifest)
    written_files.append(_file_record(manifest_path, kind="gptq_hessian_manifest"))
    return GptqHessianReduceReport(
        status="ok",
        samples=str(Path(samples_path).expanduser()) if samples_path is not None else "",
        output_dir=str(output_path),
        manifest_path=str(manifest_path),
        hessian_tensor_dir=tensor_dir_rel,
        layer_count=len(records),
        sample_ref_count=len(refs),
        row_count=total_rows,
        written_files=written_files,
    )


def load_gptq_hessian_manifest(path: str | Path) -> dict[str, GptqHessianLayerRecord]:
    """Load and validate a GPTQ Hessian manifest without reading tensor payloads."""
    import json

    manifest_path = Path(path).expanduser()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"GPTQ Hessian manifest must contain a JSON object: {manifest_path}")
    schema_version = data.get("schema_version")
    if schema_version != GPTQ_HESSIAN_STATS_SCHEMA_VERSION:
        raise ValueError(f"unsupported GPTQ Hessian manifest schema {schema_version!r}: {manifest_path}")
    layers = data.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError(f"GPTQ Hessian manifest contains no layers: {manifest_path}")
    records: dict[str, GptqHessianLayerRecord] = {}
    for layer_name, record in layers.items():
        if not isinstance(layer_name, str) or not isinstance(record, Mapping):
            raise ValueError("GPTQ Hessian manifest layers must map string names to objects")
        file_path = record.get("file_path", record.get("file"))
        tensor_name = record.get("tensor_name", record.get("tensor", "hessian"))
        if not isinstance(file_path, str) or not file_path:
            raise ValueError(f"GPTQ Hessian manifest layer {layer_name} is missing file_path")
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"GPTQ Hessian manifest layer {layer_name} has an invalid tensor_name")
        records[layer_name] = GptqHessianLayerRecord(
            layer_name=layer_name,
            file_path=file_path,
            tensor_name=tensor_name,
            channel_count=int(record.get("channel_count", 0)),
            sample_count=int(record.get("sample_count", 0)),
            row_count=int(record.get("row_count", 0)),
            normalization_count=int(record.get("normalization_count", 0)),
            channel_dim=int(record.get("channel_dim", -1)),
            dtype=str(record.get("dtype", "")),
            shape=[int(dim) for dim in record.get("shape", [])],
        )
    return records


def resolve_gptq_hessian_tensor_path(record: GptqHessianLayerRecord, *, manifest_path: str | Path) -> Path:
    """Resolve a manifest-relative Hessian tensor file path."""
    return _resolve_manifest_relative_file(Path(manifest_path).expanduser(), record.file_path)


def load_gptq_hessian_tensor(
    record: GptqHessianLayerRecord,
    *,
    manifest_path: str | Path,
    device: str | Any = "cpu",
) -> GptqHessianStats:
    """Load one per-layer GPTQ Hessian tensor referenced by a manifest record."""
    safe_open, _save_file = _require_safetensors()
    torch = _require_torch()
    tensor_path = resolve_gptq_hessian_tensor_path(record, manifest_path=manifest_path)
    if not tensor_path.is_file():
        raise ValueError(f"GPTQ Hessian tensor file is missing: {tensor_path}")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if record.tensor_name not in handle.keys():
            raise ValueError(f"GPTQ Hessian tensor {record.tensor_name!r} is missing from {tensor_path}")
        hessian = handle.get_tensor(record.tensor_name).detach().to(device=torch.device(device), dtype=torch.float32).contiguous()
    expected_k = int(record.channel_count)
    if expected_k <= 0:
        raise ValueError(f"GPTQ Hessian record has invalid channel_count for {record.layer_name}: {record.channel_count}")
    expected_shape = (expected_k, expected_k)
    if tuple(int(dim) for dim in hessian.shape) != expected_shape:
        raise ValueError(f"GPTQ Hessian shape {tuple(hessian.shape)} for {record.layer_name} does not match expected {expected_shape}")
    if record.shape and [int(dim) for dim in record.shape] != [expected_k, expected_k]:
        raise ValueError(f"GPTQ Hessian manifest shape for {record.layer_name} does not match channel_count: {record.shape}")
    if not bool(torch.isfinite(hessian).all().item()):
        raise ValueError(f"GPTQ Hessian for {record.layer_name} contains NaN or Inf values")
    return GptqHessianStats(
        hessian=hessian,
        channel_count=expected_k,
        sample_count=int(record.sample_count),
        row_count=int(record.row_count),
        normalization_count=int(record.normalization_count),
    )
