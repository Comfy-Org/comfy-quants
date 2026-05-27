"""inspect command."""

from __future__ import annotations

import csv
from pathlib import Path

from comfy_quants.cli.common import ensure_dir, print_json
from comfy_quants.core.provenance import build_provenance
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter
from comfy_quants.utils.jsonio import write_json, write_yaml


def register(subparsers):
    parser = subparsers.add_parser("inspect", help="Inspect a model using a registered model adapter")
    parser.add_argument("--model", required=True, help="Model id or local path")
    parser.add_argument("--family", required=True, help="Model family, e.g. qwen_image or qwen_image_edit")
    parser.add_argument("--revision", default=None, help="Pinned model revision/commit")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True, help="Output inspect directory")
    parser.add_argument("--json", action="store_true", help="Print inspection JSON")
    parser.set_defaults(func=run)


def _write_tables(out: Path, graph) -> None:
    with (out / "module_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "module_type", "component", "quantizable", "default_action", "parameter_count", "notes"])
        writer.writeheader()
        for module in graph.modules:
            writer.writerow({
                "name": module.name,
                "module_type": module.module_type,
                "component": module.component,
                "quantizable": module.quantizable,
                "default_action": module.default_action,
                "parameter_count": sum(t.parameter_count for t in module.tensors),
                "notes": module.notes,
            })
    with (out / "tensor_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "name", "shape", "dtype", "parameter_count", "role", "scale_axis"])
        writer.writeheader()
        for module in graph.modules:
            for tensor in module.tensors:
                writer.writerow({"module": module.name, **tensor.to_dict(), "shape": "x".join(map(str, tensor.shape))})


def run(args) -> int:
    out = ensure_dir(args.out)
    adapter = get_adapter(args.family)
    source = ModelSource(family=args.family, model_id=args.model, revision=args.revision, dtype=args.dtype)
    inspection, graph = adapter.inspect(source)
    policy = adapter.default_policy()
    memory_estimate = {
        "device": args.device,
        "gpu_profile": "rtx_pro_6000_blackwell_96gb",
        "max_vram_gb_default": 88,
        "estimated_parameter_bytes_bf16": graph.total_parameters * 2,
        "graph_kind": graph.metadata.get("graph_kind", "static_model_contract"),
        "tensor_coverage": graph.metadata.get("tensor_coverage", "declared_tensors"),
        "contract_source": graph.metadata.get("contract_source", "unknown"),
        "estimate_basis": "static_adapter_contract",
    }
    write_json(out / "model_inspection.json", inspection.to_dict())
    write_json(out / "model_graph.json", graph.to_dict())
    write_yaml(out / "default_policy.yaml", policy.to_dict())
    write_json(out / "memory_estimate.json", memory_estimate)
    write_json(out / "provenance.json", build_provenance({"command": "inspect", "model": args.model, "family": args.family}))
    _write_tables(out, graph)
    result = {"status": "ok", "out": str(out), "inspection": inspection.to_dict()}
    if args.json:
        print_json(result)
    else:
        print(f"inspection written to {out}")
        print(f"family={inspection.family} quantizable_modules={inspection.quantizable_modules} kept_high_precision={inspection.kept_high_precision_modules}")
    return 0
