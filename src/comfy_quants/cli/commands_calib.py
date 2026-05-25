"""calib command group."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from comfy_quants.algorithms.int4_svdquant.calibration import (
    load_activation_sample_refs,
    reduce_activation_samples_from_safetensors,
    summarize_activation_stats_map,
)
from comfy_quants.algorithms.int4_svdquant.hessian import (
    DEFAULT_GPTQ_HESSIAN_MANIFEST,
    DEFAULT_GPTQ_HESSIAN_TENSOR_DIR,
    reduce_gptq_hessians_from_safetensors,
)
from comfy_quants.algorithms.int4_svdquant.stats import write_activation_stats_map
from comfy_quants.backends.activation_capture.materialize import materialize_int4_activation_sample_manifest
from comfy_quants.backends.activation_capture.qwen_image_edit import (
    DEFAULT_ACTIVATION_SAMPLES,
    DEFAULT_ACTIVATION_TENSOR_DIR,
    write_qwen_image_edit_int4_activation_capture_plan,
)
from comfy_quants.calibration.datasets import load_calibration_cases, load_calibration_manifest_cases, write_calibration_cases_jsonl
from comfy_quants.cli.common import ensure_dir, local_path, print_json
from comfy_quants.core.errors import ConfigurationError, PayloadWriteError
from comfy_quants.utils.hashing import hash_file
from comfy_quants.utils.jsonio import write_json


def register(subparsers):
    parser = subparsers.add_parser("calib", help="Calibration dataset utilities")
    calib_sub = parser.add_subparsers(dest="calib_command", required=True)
    build = calib_sub.add_parser("build", help="Create a calibration manifest")
    build.add_argument("--family", required=True)
    build.add_argument("--prompt-set")
    build.add_argument("--edit-set")
    build.add_argument("--image-root")
    build.add_argument("--edit-types", default="")
    build.add_argument("--resolutions", default="1024x1024")
    build.add_argument("--timesteps", default="0,10,20,30,40")
    build.add_argument("--scheduler", default="default")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--out", required=True)
    build.add_argument("--json", action="store_true")
    build.set_defaults(func=run_build)

    records = calib_sub.add_parser("records", help="Normalize calibration prompt/edit records")
    source_group = records.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--manifest", help="calibration_manifest.json produced by calib build")
    source_group.add_argument("--input", help="Prompt/edit JSONL file to normalize directly")
    records.add_argument("--image-root", help="Image root used when --input contains relative image paths")
    records.add_argument("--edit-types", default="", help="Comma-separated edit types to keep when --input is used")
    records.add_argument("--limit", type=int)
    records.add_argument("--out", required=True)
    records.add_argument("--json", action="store_true")
    records.set_defaults(func=run_records)

    reduce = calib_sub.add_parser("reduce-int4-activations", help="Reduce captured activation tensors to an INT4 stats JSON")
    reduce.add_argument("--samples", required=True, help="JSONL/JSON manifest of captured safetensors activation samples")
    reduce.add_argument("--input-root", help="Base directory for relative sample file paths")
    reduce.add_argument("--out", required=True, help="Output stats JSON path or directory")
    reduce.add_argument("--default-tensor", default="activation", help="Tensor name used when a sample row omits tensor/tensor_name")
    reduce.add_argument("--channel-dim", type=int, default=-1, help="Activation channel dimension")
    reduce.add_argument("--schema-version", default="int4_activation_stats.v1")
    reduce.add_argument("--no-progress", action="store_true")
    reduce.add_argument("--json", action="store_true")
    reduce.set_defaults(func=run_reduce_int4_activations)

    reduce_hessian = calib_sub.add_parser(
        "reduce-int4-gptq-hessians",
        help="Reduce captured activation tensors to per-layer GPTQ Hessian artifacts",
    )
    reduce_hessian.add_argument("--samples", required=True, help="JSONL/JSON manifest of captured safetensors activation samples")
    reduce_hessian.add_argument("--input-root", help="Base directory for relative sample file paths")
    reduce_hessian.add_argument("--out", required=True, help="Output directory for int4_gptq_hessian_stats.json and Hessian tensors")
    reduce_hessian.add_argument("--default-tensor", default="activation", help="Tensor name used when a sample row omits tensor/tensor_name")
    reduce_hessian.add_argument("--channel-dim", type=int, default=-1, help="Activation channel dimension")
    reduce_hessian.add_argument("--hessian-tensor-dir", default=DEFAULT_GPTQ_HESSIAN_TENSOR_DIR)
    reduce_hessian.add_argument("--hessian-block-size", type=int, default=512)
    reduce_hessian.add_argument("--device", default="auto", help="Torch device for Hessian accumulation; auto uses cuda:0 when available")
    reduce_hessian.add_argument("--no-progress", action="store_true")
    reduce_hessian.add_argument("--json", action="store_true")
    reduce_hessian.set_defaults(func=run_reduce_int4_gptq_hessians)

    plan_capture = calib_sub.add_parser("plan-int4-capture", help="Write a plan-only INT4 activation capture target list")
    plan_capture.add_argument("--family", required=True, help="Model family; currently qwen_image_edit")
    plan_capture.add_argument("--source", required=True, help="Dense safetensors checkpoint, index JSON, or local shard directory")
    plan_capture.add_argument("--records", required=True, help="Normalized calibration records JSONL")
    plan_capture.add_argument("--out", required=True, help="Output directory for capture_plan.json and templates")
    plan_capture.add_argument("--activation-tensor-dir", default=DEFAULT_ACTIVATION_TENSOR_DIR)
    plan_capture.add_argument("--activation-samples", default=DEFAULT_ACTIVATION_SAMPLES)
    plan_capture.add_argument("--channel-dim", type=int, default=-1)
    plan_capture.add_argument("--json", action="store_true")
    plan_capture.set_defaults(func=run_plan_int4_capture)

    materialize_capture = calib_sub.add_parser(
        "materialize-int4-capture",
        help="Write reducer-ready activation sample references from an INT4 capture plan",
    )
    materialize_capture.add_argument("--plan", required=True, help="capture_plan.json produced by calib plan-int4-capture")
    materialize_capture.add_argument("--records", help="Calibration records JSONL override; defaults to records_path from the plan")
    materialize_capture.add_argument("--out", help="Output directory for activation_samples.jsonl and materialization report")
    materialize_capture.add_argument("--activation-tensor-dir", help="Relative directory containing captured activation safetensors")
    materialize_capture.add_argument("--activation-samples", help="Relative output path for the activation sample manifest")
    materialize_capture.add_argument("--json", action="store_true")
    materialize_capture.set_defaults(func=run_materialize_int4_capture)


def _hash_optional(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    return hash_file(p) if p.exists() and p.is_file() else None


def run_build(args) -> int:
    out = ensure_dir(args.out)
    manifest = {
        "schema_version": "0.1.0",
        "family": args.family,
        "prompt_set": args.prompt_set,
        "prompt_set_hash": _hash_optional(args.prompt_set),
        "edit_set": args.edit_set,
        "edit_set_hash": _hash_optional(args.edit_set),
        "image_root": args.image_root,
        "edit_types": [x for x in args.edit_types.split(",") if x],
        "resolutions": [x for x in args.resolutions.split(",") if x],
        "timesteps": [int(x) for x in args.timesteps.split(",") if x],
        "scheduler": args.scheduler,
        "seed": args.seed,
        "manifest_kind": "calibration_dataset",
    }
    write_json(out / "calibration_manifest.json", manifest)
    if args.json:
        print_json({"status": "ok", "out": str(out), "manifest": manifest})
    else:
        print(f"calibration manifest written to {out / 'calibration_manifest.json'}")
    return 0


def _resolve_output_file(out: str | Path, *, default_name: str) -> Path:
    path = local_path(out)
    if path.exists() and path.is_dir():
        return path / default_name
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / default_name


def run_records(args) -> int:
    try:
        if args.manifest:
            cases = load_calibration_manifest_cases(args.manifest, limit=args.limit)
            source = str(local_path(args.manifest))
        else:
            edit_types = [value for value in args.edit_types.split(",") if value]
            cases = load_calibration_cases(args.input, image_root=args.image_root, edit_types=edit_types, limit=args.limit)
            source = str(local_path(args.input))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    output = _resolve_output_file(args.out, default_name="calibration_records.jsonl")
    write_calibration_cases_jsonl(output, cases)
    result = {
        "status": "ok",
        "source": source,
        "output": str(output),
        "record_count": len(cases),
        "schema_version": "calibration_records_report.v1",
    }
    if args.json:
        print_json(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_reduce_int4_activations(args) -> int:
    try:
        refs = load_activation_sample_refs(
            args.samples,
            input_root=args.input_root,
            default_tensor_name=args.default_tensor,
            default_channel_dim=args.channel_dim,
        )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    def progress(event: dict) -> None:
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)

    try:
        stats = reduce_activation_samples_from_safetensors(refs, progress=None if args.no_progress else progress)
    except (ImportError, ValueError) as exc:
        raise PayloadWriteError(str(exc)) from exc

    output = _resolve_output_file(args.out, default_name="int4_activation_stats.json")
    write_activation_stats_map(output, stats, schema_version=args.schema_version)
    summary = summarize_activation_stats_map(stats)
    result = {
        "status": "ok",
        "samples": str(local_path(args.samples)),
        "output": str(output),
        "sample_ref_count": len(refs),
        "layer_count": summary["layer_count"],
        "sample_count": summary["sample_count"],
        "element_count": summary["element_count"],
        "summary": summary,
        "schema_version": "int4_activation_reduce_report.v1",
    }
    if args.json:
        print_json(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0



def run_reduce_int4_gptq_hessians(args) -> int:
    try:
        refs = load_activation_sample_refs(
            args.samples,
            input_root=args.input_root,
            default_tensor_name=args.default_tensor,
            default_channel_dim=args.channel_dim,
        )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    def progress(event: dict) -> None:
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)

    try:
        report = reduce_gptq_hessians_from_safetensors(
            refs,
            output_dir=args.out,
            samples_path=args.samples,
            hessian_tensor_dir=args.hessian_tensor_dir,
            hessian_block_size=args.hessian_block_size,
            device=args.device,
            progress=None if args.no_progress else progress,
        )
    except (ImportError, ValueError) as exc:
        raise PayloadWriteError(str(exc)) from exc

    result = report.to_dict()
    result["output"] = str(Path(report.manifest_path))
    result["manifest"] = str(Path(report.manifest_path))
    result["default_manifest_name"] = DEFAULT_GPTQ_HESSIAN_MANIFEST
    if args.json:
        print_json(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_plan_int4_capture(args) -> int:
    if args.family != "qwen_image_edit":
        raise ConfigurationError(f"unsupported INT4 activation-capture family: {args.family}")
    report = write_qwen_image_edit_int4_activation_capture_plan(
        source_checkpoint=args.source,
        records=args.records,
        out_dir=args.out,
        channel_dim=args.channel_dim,
        activation_tensor_dir=args.activation_tensor_dir,
        activation_samples=args.activation_samples,
    ).to_dict()
    if args.json:
        print_json(report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_materialize_int4_capture(args) -> int:
    report = materialize_int4_activation_sample_manifest(
        plan=args.plan,
        records=args.records,
        out_dir=args.out,
        activation_tensor_dir=args.activation_tensor_dir,
        activation_samples=args.activation_samples,
    ).to_dict()
    if args.json:
        print_json(report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
