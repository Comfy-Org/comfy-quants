"""quantize-int4 command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from comfy_quants.algorithms.int4_svdquant.config import (
    CALIBRATED_SVDQUANT_MODE,
    Int4SvdquantPipelineConfig,
    LOWRANK_CALIBRATION_OUTPUT_ERROR,
    LOWRANK_CALIBRATION_WEIGHT_RESIDUAL,
    SVDQUANT_GPTQ_EXPERIMENTAL_MODE,
    SUPPORTED_LOWRANK_CALIBRATION_MODES,
    SUPPORTED_QUANTIZATION_MODES,
)
from comfy_quants.backends.int4_full_pipeline_export import (
    plan_qwen_image_edit_svdquant_w4a4_pipeline,
    write_qwen_image_edit_svdquant_w4a4_pipeline_checkpoint,
)
from comfy_quants.cli.common import local_path, print_json
from comfy_quants.core.errors import ConfigurationError
from comfy_quants.formats.kitchen_tilepack import SVDQUANT_W4A4_FORMAT_NAME
from comfy_quants.utils.jsonio import write_json


SUPPORTED_INT4_FAMILIES = ("qwen_image_edit",)
SUPPORTED_INT4_FORMATS = (SVDQUANT_W4A4_FORMAT_NAME,)


def register(subparsers):
    parser = subparsers.add_parser("quantize-int4", help="Quantize a dense checkpoint directly to an INT4 tile-packed checkpoint")
    parser.add_argument("--family", default="qwen_image_edit", choices=SUPPORTED_INT4_FAMILIES)
    parser.add_argument("--format", default=SVDQUANT_W4A4_FORMAT_NAME, choices=SUPPORTED_INT4_FORMATS)
    parser.add_argument("--source", required=True, help="Input dense safetensors file, safetensors index JSON, or local shard directory")
    parser.add_argument("--out", required=True, help="Output .safetensors path or output directory")
    parser.add_argument("--calibration", help="Calibration JSON/JSONL path reserved for solver modes that need activation statistics")
    parser.add_argument(
        "--activation-stats",
        help="Per-layer activation stats JSON used by calibrated_svdquant and svdquant_gptq_experimental",
    )
    parser.add_argument(
        "--gptq-hessian-stats",
        help="GPTQ Hessian manifest produced by calib reduce-int4-gptq-hessians for svdquant_gptq_experimental",
    )
    parser.add_argument(
        "--activation-samples",
        help=(
            "Activation sample manifest for experimental output-error low-rank calibration. "
            "Required when --lowrank-calibration=output_error."
        ),
    )
    parser.add_argument(
        "--activation-samples-input-root",
        help="Optional root directory used to resolve relative activation sample file paths in the manifest",
    )
    parser.add_argument("--quantization-mode", default="weight_only_initialization", choices=SUPPORTED_QUANTIZATION_MODES)
    parser.add_argument("--rank", type=int, default=64, help="Low-rank branch rank for emitted SVDQuant side tensors")
    parser.add_argument("--scale-dtype", default="source", choices=("source", "float16", "bfloat16", "float32"))
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.5,
        help="Smoothing balance for calibrated_svdquant and svdquant_gptq_experimental",
    )
    parser.add_argument("--smooth-min", type=float, default=1.0 / 64.0, help="Minimum per-channel smooth factor")
    parser.add_argument("--smooth-max", type=float, default=64.0, help="Maximum per-channel smooth factor")
    parser.add_argument("--gptq-damp-percentage", type=float, default=0.01, help="GPTQ Hessian damping percentage")
    parser.add_argument("--gptq-block-size", type=int, default=128, help="GPTQ column block size")
    parser.add_argument("--gptq-num-inv-tries", type=int, default=250, help="Maximum damped Hessian factorization attempts")
    parser.add_argument("--gptq-hessian-block-size", type=int, default=512, help="Activation-to-Hessian reduction block size recorded in reports")
    parser.add_argument(
        "--lowrank-branch-input-basis",
        default="raw",
        choices=("post_smoothing", "raw"),
        help=(
            "Low-rank branch input basis for emitted SVDQuant tensors. "
            "Default raw folds smooth_factor into proj_down for Kitchen/Nunchaku-style runtimes that compute the branch from raw x. "
            "post_smoothing is retained for internal reference experiments only."
        ),
    )
    parser.add_argument(
        "--lowrank-calibration",
        default=LOWRANK_CALIBRATION_WEIGHT_RESIDUAL,
        choices=SUPPORTED_LOWRANK_CALIBRATION_MODES,
        help=(
            "Low-rank branch calibration mode. weight_residual is the default. "
            "output_error is experimental, only valid for svdquant_gptq_experimental, and requires --activation-samples."
        ),
    )
    parser.add_argument("--lowrank-ridge", type=float, default=1.0e-6, help="Ridge regularization for output-error low-rank calibration")
    parser.add_argument("--device", default="auto", help="Torch device for quantization; auto uses cuda:0 when available and falls back to cpu")
    parser.add_argument("--hash-output", action="store_true", help="Compute SHA256 for the written checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Plan selected layers and write a report without writing a checkpoint")
    parser.add_argument("--no-progress", action="store_true", help="Do not write progress events to stderr")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=run)


def _resolve_output(out: str | Path, target_format: str, *, dry_run: bool = False) -> tuple[Path, Path]:
    path = local_path(out)
    if path.exists() and path.is_dir():
        checkpoint = path / f"diffusion_pytorch_model.{target_format}.safetensors"
        report = path / "quantization_report.json"
    elif path.suffix == ".safetensors" and not dry_run:
        checkpoint = path
        report = path.with_suffix(".quantization_report.json")
    else:
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = path / f"diffusion_pytorch_model.{target_format}.safetensors"
        report = path / "quantization_report.json"
    return checkpoint, report


def _build_config(args) -> Int4SvdquantPipelineConfig:
    calibration = str(local_path(args.calibration)) if args.calibration else None
    activation_stats = str(local_path(args.activation_stats)) if args.activation_stats else None
    gptq_hessian_stats = str(local_path(args.gptq_hessian_stats)) if args.gptq_hessian_stats else None
    activation_samples = str(local_path(args.activation_samples)) if args.activation_samples else None
    activation_samples_input_root = str(local_path(args.activation_samples_input_root)) if args.activation_samples_input_root else None
    try:
        cfg = Int4SvdquantPipelineConfig(
            model_family=args.family,
            target_format=args.format,
            rank=args.rank,
            scale_dtype=args.scale_dtype,
            calibration_path=calibration,
            activation_stats_path=activation_stats,
            gptq_hessian_stats_path=gptq_hessian_stats,
            quantization_mode=args.quantization_mode,
            smooth_alpha=args.smooth_alpha,
            smooth_min=args.smooth_min,
            smooth_max=args.smooth_max,
            gptq_damp_percentage=args.gptq_damp_percentage,
            gptq_block_size=args.gptq_block_size,
            gptq_num_inv_tries=args.gptq_num_inv_tries,
            gptq_hessian_block_size=args.gptq_hessian_block_size,
            lowrank_branch_input_basis=args.lowrank_branch_input_basis,
            activation_samples_path=activation_samples,
            activation_samples_input_root=activation_samples_input_root,
            lowrank_calibration=args.lowrank_calibration,
            lowrank_ridge=args.lowrank_ridge,
        )
        cfg.validate()
        return cfg
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def run(args) -> int:
    if args.format != SVDQUANT_W4A4_FORMAT_NAME:
        raise ConfigurationError(f"unsupported INT4 format: {args.format}")
    source = local_path(args.source)
    if not source.exists():
        raise ConfigurationError(f"source checkpoint does not exist: {source}")
    checkpoint, report_path = _resolve_output(args.out, args.format, dry_run=args.dry_run)
    cfg = _build_config(args)

    def progress(event: dict) -> None:
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)

    if args.dry_run:
        plan = plan_qwen_image_edit_svdquant_w4a4_pipeline(source_checkpoint=source, config=cfg)
        activation_stats_coverage = plan.get("activation_stats_coverage") or {}
        gptq_hessian_coverage = plan.get("gptq_hessian_coverage") or {}
        activation_samples_coverage = plan.get("activation_samples_coverage") or {}
        dry_run_status = "dry_run_planned"
        exit_code = 0
        if cfg.quantization_mode == CALIBRATED_SVDQUANT_MODE and activation_stats_coverage.get("state") != "valid":
            dry_run_status = "dry_run_validation_failed"
            exit_code = 2
        if cfg.quantization_mode == SVDQUANT_GPTQ_EXPERIMENTAL_MODE and (
            activation_stats_coverage.get("state") != "valid" or gptq_hessian_coverage.get("state") != "valid"
        ):
            dry_run_status = "dry_run_validation_failed"
            exit_code = 2
        if cfg.lowrank_calibration == LOWRANK_CALIBRATION_OUTPUT_ERROR and activation_samples_coverage.get("state") != "valid":
            dry_run_status = "dry_run_validation_failed"
            exit_code = 2
        plan.update(
            {
                "status": dry_run_status,
                "output_checkpoint": str(checkpoint),
                "report": str(report_path),
                "requested_device": str(args.device or "auto"),
            }
        )
        write_json(report_path, plan)
        result = {
            "status": dry_run_status,
            "family": args.family,
            "format": args.format,
            "source": str(source),
            "output_checkpoint": str(checkpoint),
            "report": str(report_path),
            "selected_layer_count": plan["selected_layer_count"],
            "awq_modulation_layer_count": plan["awq_modulation_layer_count"],
            "pipeline_kind": plan["pipeline_kind"],
            "quantization_mode": plan["quantization_mode"],
            "algorithm_state": plan["algorithm_state"],
            "publishable_svdquant_gptq": plan["publishable_svdquant_gptq"],
            "gptq_state": plan["gptq_state"],
            "runtime_contract_state": plan["runtime_contract_state"],
            "runtime_reference_state": plan["runtime_reference_state"],
            "lowrank_branch_input_basis": plan["lowrank_branch_input_basis"],
            "proj_down_smooth_folded": plan["proj_down_smooth_folded"],
            "lowrank_calibration": plan["lowrank_calibration"],
            "lowrank_ridge": plan["lowrank_ridge"],
            "mixed_quantization_state": plan["mixed_quantization_state"],
            "algorithm_notes": plan["algorithm_notes"],
            "activation_stats_path": plan["activation_stats_path"],
            "activation_stats_state": plan["activation_stats_state"],
            "activation_stats_coverage_state": activation_stats_coverage.get("state", ""),
            "activation_stats_missing_layer_count": activation_stats_coverage.get("missing_layer_count", 0),
            "activation_stats_shape_mismatch_count": activation_stats_coverage.get("shape_mismatch_count", 0),
            "gptq_hessian_stats_path": plan["gptq_hessian_stats_path"],
            "gptq_hessian_stats_state": plan["gptq_hessian_stats_state"],
            "gptq_hessian_coverage_state": gptq_hessian_coverage.get("state", ""),
            "gptq_hessian_missing_layer_count": gptq_hessian_coverage.get("missing_layer_count", 0),
            "gptq_hessian_shape_mismatch_count": gptq_hessian_coverage.get("shape_mismatch_count", 0),
            "gptq_hessian_file_error_count": gptq_hessian_coverage.get("file_error_count", 0),
            "activation_samples_path": plan["activation_samples_path"],
            "activation_samples_state": plan["activation_samples_state"],
            "activation_samples_layer_count": plan["activation_samples_layer_count"],
            "activation_sample_ref_count": plan["activation_sample_ref_count"],
            "activation_samples_coverage_state": activation_samples_coverage.get("state", ""),
            "activation_samples_missing_layer_count": activation_samples_coverage.get("missing_layer_count", 0),
            "activation_samples_shape_mismatch_count": activation_samples_coverage.get("shape_mismatch_count", 0),
            "activation_samples_file_error_count": activation_samples_coverage.get("file_error_count", 0),
        }
    else:
        exit_code = 0
        report_obj = write_qwen_image_edit_svdquant_w4a4_pipeline_checkpoint(
            source_checkpoint=source,
            output_checkpoint=checkpoint,
            config=cfg,
            device=args.device,
            hash_output=args.hash_output,
            progress=None if args.no_progress else progress,
            report_path=report_path,
        )
        report = report_obj.to_dict()
        result = {
            "status": report["status"],
            "family": report["model_family"],
            "format": report["target_format"],
            "storage_layout": report["storage_layout"],
            "source": report["source_checkpoint"],
            "output_checkpoint": report["output_checkpoint"],
            "report": str(report_path),
            "pipeline_kind": report["pipeline_kind"],
            "quantization_mode": report["quantization_mode"],
            "algorithm_state": report["algorithm_state"],
            "publishable_svdquant_gptq": report["publishable_svdquant_gptq"],
            "gptq_state": report["gptq_state"],
            "runtime_contract_state": report["runtime_contract_state"],
            "runtime_reference_state": report["runtime_reference_state"],
            "lowrank_branch_input_basis": report["lowrank_branch_input_basis"],
            "proj_down_smooth_folded": report["proj_down_smooth_folded"],
            "lowrank_calibration": report["lowrank_calibration"],
            "lowrank_ridge": report["lowrank_ridge"],
            "mixed_quantization_state": report["mixed_quantization_state"],
            "algorithm_notes": report["algorithm_notes"],
            "selected_layer_count": report["selected_layer_count"],
            "quantized_layer_count": report["quantized_layer_count"],
            "awq_modulation_layer_count": report["awq_modulation_layer_count"],
            "copied_tensor_count": report["copied_tensor_count"],
            "output_tensor_count": report["output_tensor_count"],
            "output_bytes": report["output_bytes"],
            "output_hash": report["output_hash"],
            "output_hash_state": report["output_hash_state"],
            "requested_device": report["requested_device"],
            "execution_device": report["execution_device"],
            "cuda_max_memory_allocated_bytes": report["cuda_max_memory_allocated_bytes"],
            "cuda_max_memory_reserved_bytes": report["cuda_max_memory_reserved_bytes"],
            "activation_stats_path": report["activation_stats_path"],
            "activation_stats_state": report["activation_stats_state"],
            "activation_stats_layer_count": report["activation_stats_layer_count"],
            "gptq_hessian_stats_path": report["gptq_hessian_stats_path"],
            "gptq_hessian_stats_state": report["gptq_hessian_stats_state"],
            "gptq_hessian_layer_count": report["gptq_hessian_layer_count"],
            "activation_samples_path": report["activation_samples_path"],
            "activation_samples_state": report["activation_samples_state"],
            "activation_samples_layer_count": report["activation_samples_layer_count"],
            "activation_sample_ref_count": report["activation_sample_ref_count"],
            "gptq_config": report["gptq_config"],
        }

    if args.json:
        print_json(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code
