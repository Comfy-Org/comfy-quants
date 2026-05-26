"""export command."""

from __future__ import annotations

from pathlib import Path

from comfy_quants.cli.common import ensure_dir, local_path, print_json
from comfy_quants.core.manifest import ArtifactManifest
from comfy_quants.utils.jsonio import write_json


def register(subparsers):
    parser = subparsers.add_parser("export", help="Create an export manifest for an artifact")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--format", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=run)


def run(args) -> int:
    artifact = local_path(args.artifact)
    out = ensure_dir(args.out)
    manifest = ArtifactManifest.load(artifact / "manifest.json")
    export_manifest = {
        "schema_version": "0.1.0",
        "status": "export_manifest_created",
        "artifact_id": manifest.artifact_id,
        "source_artifact": str(artifact),
        "format": args.format,
        "backend": args.backend,
        "compatibility": "source_manifest_only",
    }
    write_json(out / "export_manifest.json", export_manifest)
    if args.json:
        print_json(export_manifest)
    else:
        print(f"export manifest written to {out / 'export_manifest.json'}")
    return 0
