"""Qwen-Image-Edit INT4 activation-capture planning."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from comfy_quants.algorithms.int4_svdquant.layer_selection import (
    Int4LinearSelection,
    activation_stats_lookup_candidates,
    select_qwen_image_edit_svdquant_linears,
)
from comfy_quants.backends.safetensors_source import SafetensorsTensorSource
from comfy_quants.core.errors import PayloadWriteError
from comfy_quants.utils.jsonio import write_json


CAPTURE_PLAN_SCHEMA_VERSION = "int4_activation_capture_plan.v1"
CAPTURE_REPORT_SCHEMA_VERSION = "int4_activation_capture_report.v1"
DEFAULT_ACTIVATION_TENSOR_DIR = "activation_tensors"
DEFAULT_ACTIVATION_SAMPLES = "activation_samples.jsonl"
DEFAULT_ACTIVATION_SAMPLES_TEMPLATE = "activation_samples.template.jsonl"


def _require_safetensors():
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise PayloadWriteError("safetensors is required for activation-capture planning") from exc
    return safe_open


@dataclass(frozen=True)
class Int4ActivationCaptureTarget:
    """One linear input tensor that should be sampled for INT4 calibration."""

    output_prefix: str
    source_prefix: str
    weight_name: str
    bias_name: str | None
    output_channels: int
    input_channels: int
    capture_tensor_name: str
    channel_dim: int
    act_unsigned: bool
    stats_lookup_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Int4ActivationCapturePlan:
    """Plan-only description of activation tensors required by calibrated INT4."""

    family: str
    source_checkpoint: str
    source: dict[str, Any]
    records_path: str
    record_count: int
    targets: list[Int4ActivationCaptureTarget]
    activation_tensor_dir: str = DEFAULT_ACTIVATION_TENSOR_DIR
    activation_samples: str = DEFAULT_ACTIVATION_SAMPLES
    activation_samples_template: str = DEFAULT_ACTIVATION_SAMPLES_TEMPLATE
    channel_dim: int = -1
    schema_version: str = CAPTURE_PLAN_SCHEMA_VERSION
    capture_mode: str = "plan_only"
    runtime_state: str = "not_executed"
    target_format: str = "svdquant_w4a4"
    quantization_mode: str = "calibrated_svdquant"

    @property
    def selected_layer_count(self) -> int:
        return len(self.targets)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["selected_layer_count"] = self.selected_layer_count
        return data


@dataclass(frozen=True)
class Int4ActivationCapturePlanReport:
    """Files and counts produced by the capture-plan command."""

    status: str
    family: str
    source_checkpoint: str
    source_layout: str
    records_path: str
    record_count: int
    out_dir: str
    capture_plan: str
    activation_samples_template: str
    capture_report: str
    selected_layer_count: int
    capture_mode: str = "plan_only"
    runtime_state: str = "not_executed"
    schema_version: str = CAPTURE_REPORT_SCHEMA_VERSION
    note: str = "This command writes a static capture plan only; it does not run model forward passes."
    written_files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count_json_records(path: str | Path) -> int:
    records_path = Path(path).expanduser()
    if not records_path.is_file():
        raise PayloadWriteError(f"calibration records file does not exist: {records_path}")
    count = 0
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise PayloadWriteError(f"invalid calibration records JSONL at {records_path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise PayloadWriteError(f"calibration record must be a JSON object at {records_path}:{line_number}")
            count += 1
    if count <= 0:
        raise PayloadWriteError(f"calibration records file is empty: {records_path}")
    return count


def _selection_with_source_shapes(source: SafetensorsTensorSource, selection: Iterable[Int4LinearSelection]) -> list[Int4LinearSelection]:
    safe_open = _require_safetensors()
    grouped: dict[Path, list[tuple[Int4LinearSelection, str]]] = {}
    for item in selection:
        weight_name = f"{item.source_prefix}.weight"
        if weight_name not in source.file_map:
            raise PayloadWriteError(f"selected linear weight is absent from safetensors source: {weight_name}")
        grouped.setdefault(source.file_path_for(weight_name), []).append((item, weight_name))

    shapes_by_output: dict[str, tuple[int, int]] = {}
    for file_path, rows in sorted(grouped.items(), key=lambda value: str(value[0])):
        if not file_path.is_file():
            raise PayloadWriteError(f"source tensor file is missing: {file_path}")
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for item, weight_name in rows:
                if weight_name not in available:
                    raise PayloadWriteError(f"safetensors index maps {weight_name} to {file_path}, but the tensor is absent from that file")
                shape = [int(dim) for dim in handle.get_slice(weight_name).get_shape()]
                if len(shape) != 2:
                    raise PayloadWriteError(f"selected linear weight must be rank 2: {weight_name} has shape {tuple(shape)}")
                shapes_by_output[item.output_prefix] = (shape[0], shape[1])

    shaped: list[Int4LinearSelection] = []
    for item in selection:
        shaped.append(
            Int4LinearSelection(
                output_prefix=item.output_prefix,
                source_prefix=item.source_prefix,
                smooth_lookup_suffix=item.smooth_lookup_suffix,
                branch_lookup_suffix=item.branch_lookup_suffix,
                act_unsigned=item.act_unsigned,
                has_bias=item.has_bias,
                shape=shapes_by_output[item.output_prefix],
            )
        )
    return shaped


def _target_from_selection(item: Int4LinearSelection, *, channel_dim: int) -> Int4ActivationCaptureTarget:
    if item.shape is None:
        raise PayloadWriteError(f"selected linear has no known shape: {item.output_prefix}")
    output_channels, input_channels = (int(item.shape[0]), int(item.shape[1]))
    return Int4ActivationCaptureTarget(
        output_prefix=item.output_prefix,
        source_prefix=item.source_prefix,
        weight_name=f"{item.source_prefix}.weight",
        bias_name=f"{item.source_prefix}.bias" if item.has_bias else None,
        output_channels=output_channels,
        input_channels=input_channels,
        capture_tensor_name=f"{item.output_prefix}.input",
        channel_dim=int(channel_dim),
        act_unsigned=bool(item.act_unsigned),
        stats_lookup_candidates=activation_stats_lookup_candidates(item),
    )


def build_qwen_image_edit_int4_activation_capture_plan(
    *,
    source_checkpoint: str | Path,
    records: str | Path,
    channel_dim: int = -1,
    activation_tensor_dir: str = DEFAULT_ACTIVATION_TENSOR_DIR,
    activation_samples: str = DEFAULT_ACTIVATION_SAMPLES,
) -> Int4ActivationCapturePlan:
    """Build a static activation-capture target list for Qwen-Image-Edit INT4."""
    source = SafetensorsTensorSource.from_path(source_checkpoint)
    record_count = _count_json_records(records)
    selection = select_qwen_image_edit_svdquant_linears(source.keys())
    if not selection:
        raise PayloadWriteError("no Qwen-Image-Edit SVDQuant W4A4 candidate layers were found in the checkpoint")
    shaped = _selection_with_source_shapes(source, selection)
    targets = [_target_from_selection(item, channel_dim=channel_dim) for item in shaped]
    return Int4ActivationCapturePlan(
        family="qwen_image_edit",
        source_checkpoint=str(source.source_path),
        source=source.describe([target.weight_name for target in targets]),
        records_path=str(Path(records).expanduser()),
        record_count=record_count,
        targets=targets,
        activation_tensor_dir=str(activation_tensor_dir),
        activation_samples=str(activation_samples),
        channel_dim=int(channel_dim),
    )


def _template_rows(plan: Int4ActivationCapturePlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in plan.targets:
        rows.append(
            {
                "layer": target.output_prefix,
                "file": f"{plan.activation_tensor_dir}/{{case_id}}.safetensors",
                "tensor": target.capture_tensor_name,
                "channel_dim": target.channel_dim,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _file_record(path: Path, *, kind: str, include_bytes: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {"kind": kind, "path": str(path)}
    if include_bytes:
        record["bytes"] = int(path.stat().st_size)
    return record


def write_qwen_image_edit_int4_activation_capture_plan(
    *,
    source_checkpoint: str | Path,
    records: str | Path,
    out_dir: str | Path,
    channel_dim: int = -1,
    activation_tensor_dir: str = DEFAULT_ACTIVATION_TENSOR_DIR,
    activation_samples: str = DEFAULT_ACTIVATION_SAMPLES,
) -> Int4ActivationCapturePlanReport:
    """Write capture-plan JSON, sample-template JSONL, and command report."""
    output_dir = Path(out_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_qwen_image_edit_int4_activation_capture_plan(
        source_checkpoint=source_checkpoint,
        records=records,
        channel_dim=channel_dim,
        activation_tensor_dir=activation_tensor_dir,
        activation_samples=activation_samples,
    )
    plan_path = output_dir / "capture_plan.json"
    samples_template_path = output_dir / plan.activation_samples_template
    report_path = output_dir / "capture_report.json"

    write_json(plan_path, plan.to_dict())
    _write_jsonl(samples_template_path, _template_rows(plan))
    written_files = [
        _file_record(plan_path, kind="capture_plan"),
        _file_record(samples_template_path, kind="activation_samples_template"),
        _file_record(report_path, kind="capture_report", include_bytes=False),
    ]
    report = Int4ActivationCapturePlanReport(
        status="capture_plan_written",
        family=plan.family,
        source_checkpoint=plan.source_checkpoint,
        source_layout=str(plan.source.get("layout", "")),
        records_path=plan.records_path,
        record_count=plan.record_count,
        out_dir=str(output_dir),
        capture_plan=str(plan_path),
        activation_samples_template=str(samples_template_path),
        capture_report=str(report_path),
        selected_layer_count=plan.selected_layer_count,
        written_files=written_files,
    )
    write_json(report_path, report.to_dict())
    return report
