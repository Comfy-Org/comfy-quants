"""quantize command."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from comfy_quants.algorithms.registry import get_algorithm
from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.backends.safetensors_payload import write_fp8_payload_from_safetensors
from comfy_quants.backends.safetensors_source import SafetensorsTensorSource
from comfy_quants.cli.common import ensure_dir, local_path, print_json
from comfy_quants.core.artifact_layout import DEFAULT_ARTIFACT_PAYLOAD_LAYOUT
from comfy_quants.core.config import load_quant_config
from comfy_quants.core.errors import ConfigurationError, PayloadWriteError
from comfy_quants.core.manifest import create_minimal_manifest
from comfy_quants.core.provenance import build_provenance
from comfy_quants.jobs.store import JobStore
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter
from comfy_quants.utils.hashing import hash_file
from comfy_quants.utils.jsonio import write_json, write_yaml
from comfy_quants.utils.system_info import collect_system_info


def register(subparsers):
    parser = subparsers.add_parser("quantize", help="Create or run an offline quantization job")
    parser.add_argument("--config", required=True, help="Quantization YAML/JSON config")
    parser.add_argument("--work-dir", required=True, help="Job working directory")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not run numeric quantization")
    parser.add_argument("--device", default="cpu", help="Torch device for numeric tensor conversion, for example cpu or cuda:0")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=run)


def _local_safetensors_source(model_id: str, source_kind: str, config_path: str | Path) -> Path:
    if source_kind != "local":
        raise ConfigurationError("non-dry-run payload writing requires model.source: local")
    path = local_path(model_id)
    if not path.is_absolute():
        path = local_path(config_path).resolve().parent / path
    if not path.exists():
        raise ConfigurationError(f"non-dry-run payload writing requires model.model_id to point to a local safetensors source: {path}")
    path = path.resolve()
    try:
        SafetensorsTensorSource.from_path(path)
    except PayloadWriteError as exc:
        raise ConfigurationError(f"non-dry-run payload writing requires a local safetensors source: {path}") from exc
    return path


def _apply_manifest_payload_report(manifest, tensor_index: dict, payload_report: dict, artifact_dir: Path) -> None:
    tensor_index["artifact_state"] = "payload_written"
    tensor_index["tensor_payload_state"] = "written"
    write_json(artifact_dir / "quant_tensor_index.json", tensor_index)
    write_json(artifact_dir / "payload_report.json", payload_report)

    manifest.compatibility["artifact_state"] = "payload_written"
    manifest.compatibility["tensor_payload_state"] = "written"
    manifest.quantization["payload_report"] = "payload_report.json"
    manifest.files = [
        {
            "path": DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.tensor_index_path,
            "kind": "quant_tensor_index",
            "state": "written",
        },
        *payload_report["written_files"],
        {
            "path": "payload_report.json",
            "kind": "payload_write_report",
            "state": "written",
        },
    ]
    manifest.hashes = dict(payload_report["hashes"])
    manifest.hashes[DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.tensor_index_path] = hash_file(artifact_dir / "quant_tensor_index.json")
    manifest.hashes["payload_report.json"] = hash_file(artifact_dir / "payload_report.json")


def run(args) -> int:
    cfg = load_quant_config(args.config)
    work_dir = ensure_dir(args.work_dir)
    store = JobStore(work_dir)
    adapter = get_adapter(cfg.model.family)
    source = ModelSource(family=cfg.model.family, model_id=cfg.model.model_id, revision=cfg.model.revision, dtype=cfg.model.dtype, source=cfg.model.source)
    inspection, graph = adapter.inspect(source)
    policy = adapter.default_policy(cfg.quant.target_dtype)
    policy.algorithm = cfg.quant.algorithm
    policy.include = cfg.quant.modules.get("include", policy.include)
    policy.exclude = cfg.quant.modules.get("exclude", policy.exclude)
    algorithm = get_algorithm(cfg.quant.algorithm)
    steps = algorithm.plan(graph, policy)
    status = "dry_run_planned" if args.dry_run else "payload_writing"
    job = store.create(job_id=cfg.project.name, status=status, config_path=str(Path(args.config)), dry_run=args.dry_run)
    write_yaml(work_dir / "config.yaml", cfg.to_dict())
    write_json(work_dir / "system_info.json", collect_system_info())
    write_json(work_dir / "provenance.json", build_provenance({"command": "quantize", "config": str(args.config)}))
    write_json(work_dir / "model_inspection.json", inspection.to_dict())
    write_json(work_dir / "model_graph.json", graph.to_dict())
    write_json(work_dir / "plan.json", {"algorithm": cfg.quant.algorithm, "target_dtype": cfg.quant.target_dtype, "steps": [asdict(s) for s in steps]})
    artifact_dir = work_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = create_minimal_manifest(
        artifact_id=cfg.project.name,
        family=cfg.model.family,
        model_id=cfg.model.model_id,
        revision=cfg.model.revision,
        algorithm=cfg.quant.algorithm,
        target_dtype=cfg.quant.target_dtype,
        compatibility_level="L0",
        hardware={"gpu_profile": cfg.hardware.gpu_profile, "max_vram_gb": cfg.hardware.max_vram_gb},
    )
    manifest.compatibility["target_level"] = cfg.artifact.compatibility_target
    manifest.quantization["payload_layout"] = DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.to_dict()
    tensor_index = build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm=cfg.quant.algorithm,
            algorithm_version=getattr(algorithm, "version", "0.1.0"),
            target_dtype=cfg.quant.target_dtype,
            scale_granularity=cfg.quant.scale.granularity,
            scale_axis=cfg.quant.scale.axis,
            scale_method=cfg.quant.scale.method,
            rounding=cfg.quant.rounding,
            compatibility_level=cfg.artifact.compatibility_target,
        ),
    )
    payload_result = None
    if args.dry_run:
        manifest.compatibility["artifact_state"] = "metadata_only"
        manifest.compatibility["tensor_payload_state"] = "pending_export"
        manifest.files.append(DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.manifest_index_record())
        write_json(artifact_dir / "quant_tensor_index.json", tensor_index)
    else:
        source_checkpoint = _local_safetensors_source(cfg.model.model_id, cfg.model.source, args.config)
        try:
            payload_report = write_fp8_payload_from_safetensors(
                source_checkpoint=source_checkpoint,
                artifact_dir=artifact_dir,
                tensor_index=tensor_index,
                target_dtype=cfg.quant.target_dtype,
                strict=True,
                device=args.device,
            ).to_dict()
        except Exception as exc:
            store.set_status("payload_failed", str(exc), current_step="payload_writer")
            raise
        _apply_manifest_payload_report(manifest, tensor_index, payload_report, artifact_dir)
        payload_result = {
            "status": payload_report["status"],
            "quantized_tensor_count": payload_report["quantized_tensor_count"],
            "weight_payload": payload_report["weight_payload_path"],
            "scale_payload": payload_report["scale_payload_path"],
            "report": "artifact/payload_report.json",
        }
        status = "payload_written"
    manifest.save(artifact_dir / "manifest.json")
    if not args.dry_run:
        job = store.set_status(status, "selected tensor payload files written", current_step="payload_writer")
    report = work_dir / "report.md"
    report.write_text(
        f"# Comfy Quants Job Report\n\n"
        f"- job_id: `{cfg.project.name}`\n"
        f"- status: `{status}`\n"
        f"- family: `{cfg.model.family}`\n"
        f"- model: `{cfg.model.model_id}`\n"
        f"- algorithm: `{cfg.quant.algorithm}`\n"
        f"- target_dtype: `{cfg.quant.target_dtype}`\n"
        f"- compatibility: `L0 schema-valid`\n"
        f"- compatibility_target: `{cfg.artifact.compatibility_target}`\n"
        f"- artifact_state: `{manifest.compatibility['artifact_state']}`\n"
        f"- tensor_payload_state: `{manifest.compatibility['tensor_payload_state']}`\n",
        encoding="utf-8",
    )
    result = {
        "status": status,
        "job": job.to_dict(),
        "work_dir": str(work_dir),
        "steps": len(steps),
        "quantized_tensors": tensor_index["selection"]["quantized_tensor_count"],
    }
    if payload_result is not None:
        result["payload"] = payload_result
    if args.json:
        print_json(result)
    else:
        print(f"job {cfg.project.name} {status}; work_dir={work_dir}; steps={len(steps)}")
    return 0
