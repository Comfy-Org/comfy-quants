"""validate command."""

from __future__ import annotations

from pathlib import Path

from comfy_quants.backends.artifact_verify import verify_artifact
from comfy_quants.cli.common import ensure_dir, print_json
from comfy_quants.utils.jsonio import write_json


def register(subparsers):
    parser = subparsers.add_parser("validate", help="Validate an artifact")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--smoke-set")
    parser.add_argument("--edit-set")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=run)


def run(args) -> int:
    artifact = Path(args.artifact)
    out = ensure_dir(args.out)
    report = verify_artifact(artifact, strict=not args.no_strict).to_dict()
    report["baseline"] = args.baseline
    report["smoke_set"] = args.smoke_set
    report["edit_set"] = args.edit_set
    report["device"] = args.device
    write_json(out / "validation_report.json", report)
    if args.json:
        print_json(report)
    else:
        print(f"validation report written to {out / 'validation_report.json'}; status={report['status']}")
    return 0 if report["status"] == "valid" else 2
