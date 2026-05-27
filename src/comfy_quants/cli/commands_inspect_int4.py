"""inspect-int4 command."""

from __future__ import annotations

from comfy_quants.backends.int4_artifact_inspect import inspect_svdquant_w4a4_artifact
from comfy_quants.cli.common import local_path, print_json
from comfy_quants.formats.kitchen_tilepack import SVDQUANT_W4A4_FORMAT_NAME
from comfy_quants.utils.jsonio import write_json


SUPPORTED_INT4_INSPECT_FORMATS = (SVDQUANT_W4A4_FORMAT_NAME,)


def register(subparsers):
    parser = subparsers.add_parser("inspect-int4", help="Inspect an exported INT4 checkpoint artifact")
    parser.add_argument("--artifact", required=True, help="Input .safetensors artifact")
    parser.add_argument("--family", default="qwen_image_edit", help="Model family contract to apply")
    parser.add_argument("--format", default=SVDQUANT_W4A4_FORMAT_NAME, choices=SUPPORTED_INT4_INSPECT_FORMATS)
    parser.add_argument(
        "--strict-qwen-image-edit-2511",
        action="store_true",
        help="Require the verified Qwen-Image-Edit-2511 tile-pack structure: 720 SVDQuant layers and split QKV branches",
    )
    parser.add_argument("--expected-svdquant-layers", type=int, default=None, help="Expected number of SVDQuant W4A4 layers")
    parser.add_argument("--require-all-lowrank", action="store_true", help="Fail if any SVDQuant layer has rank-0 low-rank tensors")
    parser.add_argument("--check-qkv-splits", action="store_true", help="Check Qwen grouped-QKV split low-rank branches")
    parser.add_argument("--example-limit", type=int, default=20, help="Maximum issue/prefix examples kept in the report")
    parser.add_argument("--out", default=None, help="Optional JSON report path")
    parser.add_argument("--json", action="store_true", help="Print inspection JSON")
    parser.set_defaults(func=run)


def run(args) -> int:
    artifact = local_path(args.artifact)
    report = inspect_svdquant_w4a4_artifact(
        artifact,
        family=args.family,
        requested_format=args.format,
        expected_svdquant_layers=args.expected_svdquant_layers,
        require_all_lowrank=bool(args.require_all_lowrank),
        check_qkv_splits=bool(args.check_qkv_splits),
        strict_qwen_image_edit_2511=bool(args.strict_qwen_image_edit_2511),
        example_limit=max(0, int(args.example_limit)),
    )
    result = report.to_dict()

    if args.out:
        write_json(local_path(args.out), result)

    if args.json:
        print_json(result)
    else:
        print(
            "INT4 inspection "
            f"status={report.status} format={args.format} "
            f"svdquant_layers={report.svdquant_w4a4_count} "
            f"lowrank_layers={report.svdquant_lowrank_count} "
            f"qkv_groups={report.qkv_group_count} "
            f"artifact={artifact}"
        )
        if args.out:
            print(f"report={local_path(args.out)}")
        if report.errors:
            print_json({"errors": report.errors, "examples": report.examples})

    return 0 if report.status == "ok" else 2
