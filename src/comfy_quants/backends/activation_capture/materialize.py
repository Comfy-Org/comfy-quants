"""Runtime-independent INT4 activation-capture manifest and writer helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from comfy_quants.core.errors import PayloadWriteError
from comfy_quants.utils.jsonio import read_json, write_json


CAPTURE_MATERIALIZATION_REPORT_SCHEMA_VERSION = "int4_activation_capture_materialization_report.v1"
CAPTURE_CASE_WRITE_REPORT_SCHEMA_VERSION = "int4_activation_case_write_report.v1"
CAPTURE_PLAN_SCHEMA_VERSION = "int4_activation_capture_plan.v1"
DEFAULT_CAPTURE_MATERIALIZATION_REPORT = "capture_materialization_report.json"

_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _require_save_file():
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise PayloadWriteError("safetensors is required for activation sample writing") from exc
    return save_file


@dataclass(frozen=True)
class CaptureCase:
    """Normalized calibration case identity used for activation dump paths."""

    case_id: str
    file_stem: str
    source_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationSampleManifestReport:
    """Summary for materialized activation sample references."""

    status: str
    plan: str
    records: str
    out_dir: str
    activation_samples: str
    activation_tensor_dir: str
    case_count: int
    target_count: int
    sample_ref_count: int
    schema_version: str = CAPTURE_MATERIALIZATION_REPORT_SCHEMA_VERSION
    written_files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationCaseWriteReport:
    """Summary for one written activation safetensors file."""

    status: str
    case_id: str
    output_file: str
    tensor_count: int
    missing_tensor_count: int
    schema_version: str = CAPTURE_CASE_WRITE_REPORT_SCHEMA_VERSION
    written_tensors: list[dict[str, Any]] = field(default_factory=list)
    missing_tensors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_capture_plan(plan: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, Mapping):
        data = dict(plan)
        plan_label = "<mapping>"
    else:
        plan_path = Path(plan).expanduser()
        data = read_json(plan_path)
        plan_label = str(plan_path)
    if not isinstance(data, dict):
        raise PayloadWriteError(f"activation capture plan must be a JSON object: {plan_label}")
    schema_version = data.get("schema_version")
    if schema_version != CAPTURE_PLAN_SCHEMA_VERSION:
        raise PayloadWriteError(f"unsupported activation capture plan schema {schema_version!r}: {plan_label}")
    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        raise PayloadWriteError(f"activation capture plan contains no targets: {plan_label}")
    for index, target in enumerate(targets, start=1):
        if not isinstance(target, Mapping):
            raise PayloadWriteError(f"activation capture plan target {index} must be a JSON object: {plan_label}")
        for field_name in ("output_prefix", "capture_tensor_name", "input_channels", "channel_dim"):
            if field_name not in target:
                raise PayloadWriteError(f"activation capture plan target {index} is missing {field_name!r}: {plan_label}")
    return data


def _read_cases(records: str | Path) -> list[dict[str, Any]]:
    records_path = Path(records).expanduser()
    if not records_path.is_file():
        raise PayloadWriteError(f"calibration records file does not exist: {records_path}")
    rows: list[dict[str, Any]] = []
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
            case_id = row.get("case_id", row.get("id"))
            if not isinstance(case_id, str) or not case_id:
                raise PayloadWriteError(f"calibration record requires a non-empty case_id or id at {records_path}:{line_number}")
            rows.append(row)
    if not rows:
        raise PayloadWriteError(f"calibration records file is empty: {records_path}")
    return rows


def _safe_case_stem(case_id: str, *, source_index: int, used: set[str]) -> str:
    base = _SAFE_STEM_RE.sub("_", case_id.strip()).strip("._-")
    if not base:
        base = f"case-{source_index:06d}"
    stem = base
    if stem in used:
        suffix = hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:10]
        stem = f"{base}-{source_index:06d}-{suffix}"
    collision_index = 1
    while stem in used:
        collision_index += 1
        stem = f"{base}-{source_index:06d}-{collision_index}"
    used.add(stem)
    return stem


def _cases_from_records(rows: list[dict[str, Any]]) -> list[CaptureCase]:
    used: set[str] = set()
    cases: list[CaptureCase] = []
    for index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id", row.get("id")))
        cases.append(CaptureCase(case_id=case_id, file_stem=_safe_case_stem(case_id, source_index=index, used=used), source_index=index))
    return cases


def _safe_relative_path(value: str) -> str:
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise PayloadWriteError(f"activation relative path is invalid: {value}")
    return str(rel)


def _target_sample_row(*, case: CaptureCase, target: Mapping[str, Any], activation_tensor_dir: str) -> dict[str, Any]:
    output_prefix = str(target["output_prefix"])
    capture_tensor_name = str(target["capture_tensor_name"])
    channel_dim = int(target.get("channel_dim", -1))
    return {
        "sample_id": f"{case.case_id}:{output_prefix}",
        "case_id": case.case_id,
        "layer": output_prefix,
        "file": _safe_relative_path(f"{activation_tensor_dir}/{case.file_stem}.safetensors"),
        "tensor": capture_tensor_name,
        "channel_dim": channel_dim,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _file_record(path: Path, *, kind: str, include_bytes: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {"kind": kind, "path": str(path)}
    if include_bytes:
        record["bytes"] = int(path.stat().st_size)
    return record


def materialize_int4_activation_sample_manifest(
    *,
    plan: str | Path,
    records: str | Path | None = None,
    out_dir: str | Path | None = None,
    activation_tensor_dir: str | None = None,
    activation_samples: str | None = None,
    report_name: str = DEFAULT_CAPTURE_MATERIALIZATION_REPORT,
) -> ActivationSampleManifestReport:
    """Write a reducer-ready activation sample manifest from a capture plan."""
    plan_path = Path(plan).expanduser()
    plan_data = _load_capture_plan(plan_path)
    records_value = records if records is not None else plan_data.get("records_path")
    if not records_value:
        raise PayloadWriteError("activation capture plan has no records_path; pass records explicitly")
    records_path = Path(records_value).expanduser()
    output_dir = Path(out_dir).expanduser() if out_dir is not None else plan_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_dir = str(activation_tensor_dir or plan_data.get("activation_tensor_dir") or "activation_tensors")
    samples_name = str(activation_samples or plan_data.get("activation_samples") or "activation_samples.jsonl")
    tensor_dir = _safe_relative_path(tensor_dir)
    output_dir.joinpath(*PurePosixPath(tensor_dir).parts).mkdir(parents=True, exist_ok=True)
    samples_rel = _safe_relative_path(samples_name)
    samples_path = output_dir.joinpath(*PurePosixPath(samples_rel).parts)
    report_rel = _safe_relative_path(report_name)
    report_path = output_dir.joinpath(*PurePosixPath(report_rel).parts)

    cases = _cases_from_records(_read_cases(records_path))
    targets = list(plan_data["targets"])
    rows = [_target_sample_row(case=case, target=target, activation_tensor_dir=tensor_dir) for case in cases for target in targets]
    _write_jsonl(samples_path, rows)

    written_files = [
        _file_record(samples_path, kind="activation_samples"),
        _file_record(report_path, kind="capture_materialization_report", include_bytes=False),
    ]
    report = ActivationSampleManifestReport(
        status="activation_sample_manifest_written",
        plan=str(plan_path),
        records=str(records_path),
        out_dir=str(output_dir),
        activation_samples=str(samples_path),
        activation_tensor_dir=tensor_dir,
        case_count=len(cases),
        target_count=len(targets),
        sample_ref_count=len(rows),
        written_files=written_files,
    )
    write_json(report_path, report.to_dict())
    return report


def _case_file_stem_from_records(plan_data: Mapping[str, Any], case_id: str) -> str:
    records_path = plan_data.get("records_path")
    if isinstance(records_path, str) and records_path:
        cases = _cases_from_records(_read_cases(records_path))
        for case in cases:
            if case.case_id == case_id:
                return case.file_stem
    return _safe_case_stem(case_id, source_index=1, used=set())


def _shape_of(tensor: Any) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise PayloadWriteError("activation tensor object has no shape attribute")
    return tuple(int(dim) for dim in shape)


def _dtype_name(tensor: Any) -> str:
    return str(getattr(tensor, "dtype", "unknown"))


def _prepare_tensor(tensor: Any) -> Any:
    value = tensor.detach() if hasattr(tensor, "detach") else tensor
    value = value.contiguous() if hasattr(value, "contiguous") else value
    value = value.cpu() if hasattr(value, "cpu") else value
    return value


def _normalize_dim(dim: int, rank: int) -> int:
    normalized = int(dim)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise PayloadWriteError(f"activation channel_dim {dim} is out of range for rank {rank}")
    return normalized


def write_int4_activation_case_safetensors(
    *,
    plan: str | Path | Mapping[str, Any],
    case_id: str,
    tensors: Mapping[str, Any],
    out_dir: str | Path | None = None,
    activation_tensor_dir: str | None = None,
    output_file: str | Path | None = None,
    allow_missing: bool = False,
) -> ActivationCaseWriteReport:
    """Write one case's captured activation tensors in the planned layout."""
    save_file = _require_save_file()
    plan_data = _load_capture_plan(plan)
    if not isinstance(case_id, str) or not case_id:
        raise PayloadWriteError("case_id must be a non-empty string")
    if not isinstance(tensors, Mapping):
        raise PayloadWriteError("tensors must be a mapping from capture tensor names to tensor objects")

    if output_file is not None:
        output_path = Path(output_file).expanduser()
    else:
        base_dir = Path(out_dir).expanduser() if out_dir is not None else Path.cwd()
        tensor_dir = str(activation_tensor_dir or plan_data.get("activation_tensor_dir") or "activation_tensors")
        file_stem = _case_file_stem_from_records(plan_data, case_id)
        rel_file = _safe_relative_path(f"{tensor_dir}/{file_stem}.safetensors")
        output_path = base_dir.joinpath(*PurePosixPath(rel_file).parts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, Any] = {}
    written: list[dict[str, Any]] = []
    missing: list[str] = []
    for target in plan_data["targets"]:
        tensor_name = str(target["capture_tensor_name"])
        output_prefix = str(target["output_prefix"])
        tensor = tensors.get(tensor_name, tensors.get(output_prefix))
        if tensor is None:
            missing.append(tensor_name)
            continue
        shape = _shape_of(tensor)
        channel_dim = _normalize_dim(int(target.get("channel_dim", -1)), len(shape))
        expected_channels = int(target["input_channels"])
        actual_channels = int(shape[channel_dim])
        if actual_channels != expected_channels:
            raise PayloadWriteError(
                f"activation tensor {tensor_name!r} channel count mismatch: expected {expected_channels}, got {actual_channels} at dim {channel_dim}"
            )
        prepared[tensor_name] = _prepare_tensor(tensor)
        written.append(
            {
                "tensor": tensor_name,
                "layer": output_prefix,
                "shape": list(shape),
                "dtype": _dtype_name(tensor),
                "channel_dim": channel_dim,
            }
        )

    if missing and not allow_missing:
        first = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise PayloadWriteError(f"missing activation tensors for case {case_id}: {first}{suffix}")
    if not prepared:
        raise PayloadWriteError(f"no activation tensors were provided for case {case_id}")

    metadata = {
        "schema_version": "int4_activation_case_safetensors.v1",
        "case_id": case_id,
        "family": str(plan_data.get("family", "")),
        "target_format": str(plan_data.get("target_format", "")),
    }
    save_file(prepared, str(output_path), metadata=metadata)
    return ActivationCaseWriteReport(
        status="activation_case_written",
        case_id=case_id,
        output_file=str(output_path),
        tensor_count=len(prepared),
        missing_tensor_count=len(missing),
        written_tensors=written,
        missing_tensors=missing,
    )
