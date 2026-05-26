"""One-step Qwen-Image-Edit-2511 INT4 tile-pack command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from comfy_quants.backends.qwen_image_edit_int4_pipeline import (
    DEFAULT_AWQ_GROUP_SIZE,
    DEFAULT_CALIBRATION_RELATIVE_PATH,
    DEFAULT_CALIBRATION_SAMPLES,
    DEFAULT_DEEPCOMPRESSOR_ROOT,
    DEFAULT_GPUS,
    DEFAULT_MODEL_ID,
    DEFAULT_NUNCHAKU_ROOT,
    DEFAULT_ROUTE,
    DEFAULT_SEARCH_STRENGTH,
    ROUTES,
    SEARCH_STRENGTHS,
    QwenImageEditInt4PipelineConfig,
    QwenImageEditInt4TilepackPipeline,
)
from comfy_quants.cli.common import local_path, print_json


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "qwen-image-edit-2511-int4",
        help="Run Qwen-Image-Edit-2511 INT4 search/PTQ and export one ComfyUI tile-pack safetensors file",
    )
    parser.add_argument("--out", "-o", required=True, help="Final single-file INT4 tile-pack .safetensors output")
    parser.add_argument(
        "--base-checkpoint",
        dest="base_checkpoint",
        metavar="BASE_CHECKPOINT",
        default=None,
        help="BF16 transformer checkpoint used to assemble the final tile-pack artifact",
    )
    parser.add_argument("--base-comfy", dest="base_checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--model", "--model-id", dest="model_id", default=DEFAULT_MODEL_ID, help="HF id or local model path")
    parser.add_argument(
        "--deepcompressor-root",
        default=str(DEFAULT_DEEPCOMPRESSOR_ROOT),
        help="Local DeepCompressor checkout",
    )
    parser.add_argument(
        "--nunchaku-root",
        default=str(DEFAULT_NUNCHAKU_ROOT),
        help="Local Nunchaku checkout with tools/kitchen_native helpers",
    )
    parser.add_argument(
        "--search-strength",
        "--candidate",
        dest="search_strength",
        default=DEFAULT_SEARCH_STRENGTH,
        choices=SEARCH_STRENGTHS,
        help="Search preset. Default: quality-r64.",
    )
    parser.add_argument(
        "--calibration-path",
        "--search-calib-path",
        dest="calibration_path",
        default=None,
        help=(
            "Calibration dataset/cache path. Default: <deepcompressor-root>/"
            f"{DEFAULT_CALIBRATION_RELATIVE_PATH.as_posix()}"
        ),
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help="Number of calibration samples used by search/PTQ. Default: 128.",
    )
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help="CUDA_VISIBLE_DEVICES for DeepCompressor PTQ. Default: 0")
    parser.add_argument("--python-bin", default="python", help="Python executable used for external tools")
    parser.add_argument("--micromamba-env", default=None, help="Optional micromamba prefix for external tools")
    parser.add_argument("--runs-root", default=None, help="DeepCompressor runs root used to locate the produced PTQ model")
    parser.add_argument("--export-root", default=None, help="Intermediate split/raw export directory")
    parser.add_argument("--export-name", default=None, help="Intermediate Nunchaku split checkpoint directory name")
    parser.add_argument(
        "--quant-path",
        default=None,
        help="Existing DeepCompressor PTQ artifact directory. If omitted, the command runs PTQ first.",
    )
    parser.add_argument(
        "--ptq-output-dirname",
        default=None,
        help="Override DeepCompressor output.dirname for the PTQ run. Usually leave unset.",
    )
    parser.add_argument("--route", choices=ROUTES, default=DEFAULT_ROUTE, help="Export route. Default: nunchaku-bridge")
    parser.add_argument("--raw-nunchaku", default=None, help="Optional path for the intermediate raw Nunchaku safetensors")
    parser.add_argument("--awq-group-size", type=int, default=DEFAULT_AWQ_GROUP_SIZE, help="AWQ modulation group size")
    parser.add_argument("--no-awq-modulation", action="store_true", help="Disable AWQ modulation bridge mode")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing split/raw/final artifacts when present")
    parser.add_argument("--hash-output", action="store_true", help="Compute SHA256 for the final output")
    parser.add_argument("--no-inspect", action="store_true", help="Skip strict structural inspection after writing the artifact")
    parser.add_argument("--no-strict-inspect", action="store_true", help="Run inspection without the Qwen-Image-Edit-2511 strict counts")
    parser.add_argument("--report", default=None, help="Pipeline report JSON path. Default: <out>.pipeline_report.json")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan without running the export")
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    parser.set_defaults(func=run)


def _config_from_args(args) -> QwenImageEditInt4PipelineConfig:
    return QwenImageEditInt4PipelineConfig(
        output=local_path(args.out),
        base_checkpoint=local_path(args.base_checkpoint) if args.base_checkpoint else None,
        model_id=args.model_id,
        deepcompressor_root=local_path(args.deepcompressor_root),
        nunchaku_root=local_path(args.nunchaku_root),
        search_strength=args.search_strength,
        calibration_path=local_path(args.calibration_path) if args.calibration_path else None,
        calibration_samples=int(args.calibration_samples),
        gpus=args.gpus,
        python_bin=args.python_bin,
        micromamba_env=local_path(args.micromamba_env) if args.micromamba_env else None,
        runs_root=local_path(args.runs_root) if args.runs_root else None,
        export_root=local_path(args.export_root) if args.export_root else None,
        export_name=args.export_name,
        quant_path=local_path(args.quant_path) if args.quant_path else None,
        ptq_output_dirname=args.ptq_output_dirname,
        route=args.route,
        raw_nunchaku=local_path(args.raw_nunchaku) if args.raw_nunchaku else None,
        awq_group_size=int(args.awq_group_size),
        no_awq_modulation=bool(args.no_awq_modulation),
        reuse=bool(args.reuse),
        hash_output=bool(args.hash_output),
        inspect_output=not bool(args.no_inspect),
        strict_inspect=not bool(args.no_strict_inspect),
        dry_run=bool(args.dry_run),
        report=local_path(args.report) if args.report else None,
    )


def run(args) -> int:
    config = _config_from_args(args)

    def progress(event: dict) -> None:
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)

    runner = QwenImageEditInt4TilepackPipeline(config)
    result = runner.plan() if args.dry_run else runner.run(progress=progress)
    result_dict = result.to_dict()
    if args.json:
        print_json(result_dict)
    else:
        print(
            "Qwen-Image-Edit-2511 INT4 tile-pack "
            f"status={result.status} output={Path(result.output)} report={Path(result.report)}"
        )
        print(f"search_strength={result_dict['config']['search_strength']} calibration_samples={result_dict['config']['calibration_samples']}")
        print(f"calibration_path={result_dict['config']['calibration_path']}")
        if result.quant_path:
            print(f"quant_path={result.quant_path}")
        if args.dry_run:
            print("planned_commands:")
            for command in result.commands:
                print(f"- [{command['label']}] (cd {command['cwd']} && {command['shell']})")
    return 0 if result.status in {"ok", "dry_run_planned"} else 2
