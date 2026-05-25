"""export-int4 command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from comfy_quants.backends.deepcompressor_import import (
    write_qwen_image_edit_deepcompressor_svdquant_kitchen_checkpoint,
)
from comfy_quants.backends.int4_kitchen_export import write_svdquant_w4a4_kitchen_checkpoint_from_safetensors
from comfy_quants.cli.common import local_path, print_json
from comfy_quants.core.errors import ConfigurationError
from comfy_quants.formats.kitchen_tilepack import SVDQUANT_W4A4_FORMAT_NAME
from comfy_quants.utils.jsonio import write_json


SUPPORTED_INT4_EXPORT_FORMATS = (SVDQUANT_W4A4_FORMAT_NAME,)
SUPPORTED_SOURCE_FORMATS = ("natural-safetensors", "deepcompressor-qwen-image-edit")


def register(subparsers):
    parser = subparsers.add_parser("export-int4", help="Export an INT4 checkpoint artifact")
    parser.add_argument("--format", default=SVDQUANT_W4A4_FORMAT_NAME, choices=SUPPORTED_INT4_EXPORT_FORMATS)
    parser.add_argument(
        "--source-format",
        default="natural-safetensors",
        choices=SUPPORTED_SOURCE_FORMATS,
        help="Input artifact format. natural-safetensors reads an already-natural SVDQuant checkpoint; "
        "deepcompressor-qwen-image-edit reads model.pt/scale.pt/smooth.pt/branch.pt.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Input path: safetensors file/index/directory for natural-safetensors, or PTQ artifact directory for DeepCompressor",
    )
    parser.add_argument("--out", required=True, help="Output .safetensors path or output directory")
    parser.add_argument("--device", default="auto", help="Torch device for layout transforms; auto uses cuda:0 when available and falls back to cpu")
    parser.add_argument("--hash-output", action="store_true", help="Compute a SHA256 hash after writing the checkpoint")
    parser.add_argument("--allow-empty", action="store_true", help="Allow an input with no SVDQuant W4A4 layers")
    parser.add_argument("--no-progress", action="store_true", help="Do not write export progress events to stderr")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=run)


def _resolve_output(out: str | Path, target_format: str) -> tuple[Path, Path]:
    path = local_path(out)
    if path.exists() and path.is_dir():
        checkpoint = path / f"diffusion_pytorch_model.{target_format}.safetensors"
        report = path / "export_report.json"
    elif path.suffix == ".safetensors":
        checkpoint = path
        report = path.with_suffix(".export_report.json")
    else:
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = path / f"diffusion_pytorch_model.{target_format}.safetensors"
        report = path / "export_report.json"
    return checkpoint, report


def run(args) -> int:
    if args.format != SVDQUANT_W4A4_FORMAT_NAME:
        raise ConfigurationError(f"unsupported INT4 export format: {args.format}")

    source = local_path(args.source)
    if not source.exists():
        raise ConfigurationError(f"source checkpoint does not exist: {source}")
    checkpoint, report_path = _resolve_output(args.out, args.format)

    def progress(event: dict) -> None:
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)

    if args.source_format == "natural-safetensors":
        report_obj = write_svdquant_w4a4_kitchen_checkpoint_from_safetensors(
            source_checkpoint=source,
            output_checkpoint=checkpoint,
            device=args.device,
            require_svdquant=not args.allow_empty,
            hash_output=args.hash_output,
            metadata={"export_format": args.format, "source_format": args.source_format},
            progress=None if args.no_progress else progress,
        )
    elif args.source_format == "deepcompressor-qwen-image-edit":
        if not source.is_dir():
            raise ConfigurationError(f"DeepCompressor source must be a directory containing model.pt and scale.pt: {source}")
        report_obj = write_qwen_image_edit_deepcompressor_svdquant_kitchen_checkpoint(
            quant_path=source,
            output_checkpoint=checkpoint,
            device=args.device,
            require_svdquant=not args.allow_empty,
            hash_output=args.hash_output,
            metadata={"export_format": args.format, "source_format": args.source_format},
            progress=None if args.no_progress else progress,
        )
    else:  # pragma: no cover - argparse choices prevent this
        raise ConfigurationError(f"unsupported INT4 source format: {args.source_format}")

    report = report_obj.to_dict()
    write_json(report_path, report)

    result = {
        "status": report["status"],
        "format": args.format,
        "source_format": args.source_format,
        "output_checkpoint": report["output_checkpoint"],
        "report": str(report_path),
        "repacked_layer_count": report["repacked_layer_count"],
        "repacked_tensor_count": report["repacked_tensor_count"],
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
    if report.get("source_import"):
        result["source_import"] = {
            "source_format": report["source_import"].get("source_format"),
            "model_family": report["source_import"].get("model_family"),
            "imported_layer_count": report["source_import"].get("imported_layer_count"),
            "skipped_no_scale_count": report["source_import"].get("skipped_no_scale_count"),
            "execution_device": report["source_import"].get("execution_device"),
        }
    if args.json:
        print_json(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
