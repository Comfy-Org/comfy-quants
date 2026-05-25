"""export-model command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from comfy_quants.algorithms.registry import get_algorithm
from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.backends.inference_model_export import write_fp8_inference_checkpoint_from_safetensors
from comfy_quants.backends.safetensors_source import SafetensorsTensorSource
from comfy_quants.cli.common import local_path, print_json
from comfy_quants.core.config import load_quant_config
from comfy_quants.core.errors import ConfigurationError, PayloadWriteError
from comfy_quants.formats.fp8_common import FP8_FORMAT_NAMES, get_fp8_runtime_spec
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter
from comfy_quants.utils.jsonio import write_json


def _default_checkpoint_name(target_dtype: str) -> str:
    spec = get_fp8_runtime_spec(target_dtype)
    return f"diffusion_pytorch_model.{spec.name}.safetensors"


def _expand_local_path(value: str | Path) -> Path:
    return local_path(value)


def _config_base_dir(config_path: str | Path) -> Path:
    return _expand_local_path(config_path).resolve().parent


def register(subparsers):
    parser = subparsers.add_parser("export-model", help="Export a full FP8 inference checkpoint")
    parser.add_argument("--config", required=True, help="Quantization YAML/JSON config")
    parser.add_argument("--source", default=None, help="Local safetensors file, index JSON, or indexed directory")
    parser.add_argument("--out", required=True, help="Output .safetensors path or output directory")
    parser.add_argument("--device", default="auto", help="Torch device for tensor conversion; auto uses cuda:0 when available and falls back to cpu")
    parser.add_argument("--scale-granularity", default="per_tensor", choices=["per_tensor"], help="Weight scale granularity for the exported checkpoint")
    parser.add_argument("--hash-output", action="store_true", help="Compute a SHA256 hash after writing the checkpoint")
    parser.add_argument("--no-progress", action="store_true", help="Do not write export progress events to stderr")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=run)


def _resolve_source(cfg, config_path: str | Path, override: str | None) -> Path:
    if override:
        source = _expand_local_path(override)
    else:
        if cfg.model.source != "local":
            raise ConfigurationError("export-model requires --source unless model.source is local")
        source = _expand_local_path(cfg.model.model_id)
        if not source.is_absolute():
            source = _config_base_dir(config_path) / source
    if not source.exists():
        raise ConfigurationError(f"source checkpoint does not exist: {source}")
    source = source.resolve()
    try:
        SafetensorsTensorSource.from_path(source)
    except PayloadWriteError as exc:
        raise ConfigurationError(f"source checkpoint must be a safetensors file, index JSON, or indexed directory: {source}") from exc
    return source


def _resolve_output(out: str | Path, target_dtype: str) -> tuple[Path, Path]:
    path = _expand_local_path(out)
    if path.exists() and path.is_dir():
        checkpoint = path / _default_checkpoint_name(target_dtype)
        report = path / "export_report.json"
    elif path.suffix == ".safetensors":
        checkpoint = path
        report = path.with_suffix(".export_report.json")
    else:
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = path / _default_checkpoint_name(target_dtype)
        report = path / "export_report.json"
    return checkpoint, report


def _build_tensor_index(cfg, config_path: str | Path, source: Path, scale_granularity: str) -> dict:
    adapter = get_adapter(cfg.model.family)
    model_source = ModelSource(
        family=cfg.model.family,
        model_id=str(source),
        revision=cfg.model.revision,
        dtype=cfg.model.dtype,
        source="local",
    )
    _inspection, graph = adapter.inspect(model_source)
    policy = adapter.default_policy(cfg.quant.target_dtype)
    policy.algorithm = cfg.quant.algorithm
    policy.include = cfg.quant.modules.get("include", policy.include)
    policy.exclude = cfg.quant.modules.get("exclude", policy.exclude)
    algorithm = get_algorithm(cfg.quant.algorithm)
    algorithm.plan(graph, policy)
    return build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm=cfg.quant.algorithm,
            algorithm_version=getattr(algorithm, "version", "0.1.0"),
            target_dtype=cfg.quant.target_dtype,
            scale_granularity=scale_granularity,
            scale_axis=cfg.quant.scale.axis,
            scale_method=cfg.quant.scale.method,
            rounding=cfg.quant.rounding,
            compatibility_level=cfg.artifact.compatibility_target,
            artifact_state="model_export",
            tensor_payload_state="written_in_checkpoint",
        ),
    )


def run(args) -> int:
    cfg = load_quant_config(args.config)
    try:
        get_fp8_runtime_spec(cfg.quant.target_dtype)
    except KeyError as exc:
        supported = ", ".join(FP8_FORMAT_NAMES)
        raise ConfigurationError(f"export-model supports FP8 target dtypes ({supported}), got {cfg.quant.target_dtype}") from exc

    source = _resolve_source(cfg, args.config, args.source)
    checkpoint, report_path = _resolve_output(args.out, cfg.quant.target_dtype)
    tensor_index = _build_tensor_index(cfg, args.config, source, args.scale_granularity)

    def progress(event: dict) -> None:
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)

    report = write_fp8_inference_checkpoint_from_safetensors(
        source_checkpoint=source,
        output_checkpoint=checkpoint,
        tensor_index=tensor_index,
        target_dtype=cfg.quant.target_dtype,
        scale_granularity=args.scale_granularity,
        scale_axis=None,
        device=args.device,
        strict=True,
        config_source=source,
        hash_output=args.hash_output,
        metadata={
            "model_family": cfg.model.family,
            "model_id": cfg.model.model_id,
            "project": cfg.project.name,
        },
        progress=None if args.no_progress else progress,
    ).to_dict()
    report["tensor_index"] = {
        "schema_version": tensor_index.get("schema_version"),
        "selection": tensor_index.get("selection"),
        "format": tensor_index.get("format"),
    }
    write_json(report_path, report)

    result = {
        "status": report["status"],
        "output_checkpoint": report["output_checkpoint"],
        "report": str(report_path),
        "quantized_tensor_count": report["quantized_tensor_count"],
        "copied_tensor_count": report["copied_tensor_count"],
        "output_tensor_count": report["output_tensor_count"],
        "output_bytes": report["output_bytes"],
        "output_hash": report["output_hash"],
        "output_hash_state": report["output_hash_state"],
        "requested_device": report["requested_device"],
        "execution_device": report["execution_device"],
        "output_tensor_device": report["output_tensor_device"],
        "cuda_max_memory_allocated_bytes": report["cuda_max_memory_allocated_bytes"],
        "cuda_max_memory_reserved_bytes": report["cuda_max_memory_reserved_bytes"],
    }
    if args.json:
        print_json(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
