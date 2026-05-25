"""Artifact payload verification helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from comfy_quants.core.errors import ManifestError
from comfy_quants.core.manifest import ArtifactManifest
from comfy_quants.formats.fp8_common import FP8_FORMAT_NAMES, is_fp8_format_name
from comfy_quants.utils.hashing import hash_file
from comfy_quants.utils.jsonio import read_json


def _require_safetensors():
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency should be present in normal installs
        raise ManifestError("safetensors is required for artifact verification") from exc
    return safe_open


@dataclass
class ArtifactVerificationReport:
    artifact_dir: str
    status: str
    schema_version: str = "artifact_verification_report.v1"
    manifest_checked: bool = False
    tensor_index_checked: bool = False
    payload_checked: bool = False
    hash_checked_count: int = 0
    tensor_count: int = 0
    payload_tensor_count: int = 0
    scale_tensor_count: int = 0
    errors: list[str] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    payload_files: dict[str, dict[str, Any]] = field(default_factory=dict)
    scale_files: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _artifact_path(artifact_dir: Path, relative_path: str) -> Path:
    rel = PurePosixPath(relative_path)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ManifestError(f"artifact-relative path is invalid: {relative_path}")
    return artifact_dir.joinpath(*rel.parts)


def _safe_shape(value: Any) -> list[int]:
    return [int(dim) for dim in value]


def _record_error(errors: list[str], message: str, *, strict: bool) -> None:
    errors.append(message)
    if strict:
        raise ManifestError(message)


def _file_summary(path: Path, relative_path: str, expected_hash: str | None, *, strict: bool, errors: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative_path,
        "exists": path.is_file(),
    }
    if not path.is_file():
        _record_error(errors, f"artifact file is missing: {relative_path}", strict=strict)
        return record
    digest = hash_file(path)
    record["bytes"] = path.stat().st_size
    record["hash"] = digest
    if expected_hash is not None:
        record["expected_hash"] = expected_hash
        record["hash_match"] = digest == expected_hash
        if digest != expected_hash:
            _record_error(errors, f"artifact file hash mismatch: {relative_path}", strict=strict)
    return record


def _check_unique_tensor(targets: set[str], target: str, *, strict: bool, errors: list[str]) -> None:
    if target in targets:
        _record_error(errors, f"duplicate tensor reference in artifact index: {target}", strict=strict)
    targets.add(target)


def verify_artifact(artifact_dir: str | Path, *, strict: bool = True) -> ArtifactVerificationReport:
    """Verify manifest, tensor index, payload files, hashes, dtypes, and shapes."""
    safe_open = _require_safetensors()
    artifact = Path(artifact_dir)
    errors: list[str] = []

    manifest = ArtifactManifest.load(artifact / "manifest.json")
    index_path = artifact / "quant_tensor_index.json"
    tensor_index = read_json(index_path)
    rows = list(tensor_index.get("tensors") or [])
    payload_report_path = artifact / "payload_report.json"
    payload_report = read_json(payload_report_path) if payload_report_path.is_file() else None

    report = ArtifactVerificationReport(
        artifact_dir=str(artifact),
        status="valid",
        manifest_checked=True,
        tensor_index_checked=True,
        tensor_count=len(rows),
    )

    manifest_hashes = dict(manifest.hashes or {})
    file_paths = {"manifest.json", "quant_tensor_index.json"}
    if payload_report_path.is_file():
        file_paths.add("payload_report.json")
    for file_record in manifest.files or []:
        rel_path = file_record.get("path")
        if isinstance(rel_path, str) and rel_path:
            file_paths.add(rel_path)
    if payload_report:
        for file_record in payload_report.get("written_files") or []:
            rel_path = file_record.get("path")
            if isinstance(rel_path, str) and rel_path:
                file_paths.add(rel_path)

    for rel_path in sorted(file_paths):
        expected = manifest_hashes.get(rel_path)
        try:
            path = _artifact_path(artifact, rel_path)
        except ManifestError as exc:
            _record_error(errors, str(exc), strict=strict)
            continue
        summary = _file_summary(path, rel_path, expected, strict=strict, errors=errors)
        report.files.append(summary)
        if "expected_hash" in summary:
            report.hash_checked_count += 1

    if not is_fp8_format_name(tensor_index.get("format", {}).get("name")):
        supported = ", ".join(FP8_FORMAT_NAMES)
        _record_error(errors, f"unsupported artifact tensor format: {tensor_index.get('format', {}).get('name')}; supported: {supported}", strict=strict)
    if tensor_index.get("format", {}).get("storage_dtype") != "uint8":
        _record_error(errors, f"unsupported artifact storage dtype: {tensor_index.get('format', {}).get('storage_dtype')}", strict=strict)

    payload_files: dict[str, set[str]] = {}
    scale_files: dict[str, set[str]] = {}
    payload_targets: set[str] = set()
    scale_targets: set[str] = set()

    for row in rows:
        payload_meta = row.get("payload") or {}
        scale_meta = row.get("scale") or {}
        payload_file = payload_meta.get("file")
        scale_file = scale_meta.get("file")
        payload_name = payload_meta.get("tensor_name") or row.get("name")
        scale_name = scale_meta.get("tensor_name")
        if not isinstance(payload_file, str) or not isinstance(payload_name, str):
            _record_error(errors, f"tensor row has invalid payload metadata: {row.get('name')}", strict=strict)
            continue
        if not isinstance(scale_file, str) or not isinstance(scale_name, str):
            _record_error(errors, f"tensor row has invalid scale metadata: {row.get('name')}", strict=strict)
            continue
        _check_unique_tensor(payload_targets, f"{payload_file}:{payload_name}", strict=strict, errors=errors)
        _check_unique_tensor(scale_targets, f"{scale_file}:{scale_name}", strict=strict, errors=errors)
        payload_files.setdefault(payload_file, set()).add(payload_name)
        scale_files.setdefault(scale_file, set()).add(scale_name)

    for rel_path, expected_names in sorted(payload_files.items()):
        path = _artifact_path(artifact, rel_path)
        if not path.is_file():
            _record_error(errors, f"payload file is missing: {rel_path}", strict=strict)
            continue
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            missing = sorted(expected_names - keys)
            if missing:
                _record_error(errors, f"payload tensors are missing in {rel_path}: {missing[:8]}", strict=strict)
            for row in rows:
                payload_meta = row.get("payload") or {}
                if payload_meta.get("file") != rel_path:
                    continue
                name = payload_meta.get("tensor_name") or row.get("name")
                if name not in keys:
                    continue
                tensor = handle.get_tensor(name)
                if str(tensor.dtype) != "torch.uint8":
                    _record_error(errors, f"payload tensor dtype mismatch for {name}: {tensor.dtype}", strict=strict)
                expected_shape = _safe_shape(row.get("shape") or [])
                got_shape = _safe_shape(tensor.shape)
                if got_shape != expected_shape:
                    _record_error(errors, f"payload tensor shape mismatch for {name}: expected {expected_shape}, got {got_shape}", strict=strict)
            report.payload_files[rel_path] = {"tensor_count": len(keys), "referenced_tensor_count": len(expected_names)}
            report.payload_tensor_count += len(keys)

    for rel_path, expected_names in sorted(scale_files.items()):
        path = _artifact_path(artifact, rel_path)
        if not path.is_file():
            _record_error(errors, f"scale file is missing: {rel_path}", strict=strict)
            continue
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            missing = sorted(expected_names - keys)
            if missing:
                _record_error(errors, f"scale tensors are missing in {rel_path}: {missing[:8]}", strict=strict)
            for row in rows:
                scale_meta = row.get("scale") or {}
                if scale_meta.get("file") != rel_path:
                    continue
                name = scale_meta.get("tensor_name")
                if name not in keys:
                    continue
                tensor = handle.get_tensor(name)
                if str(tensor.dtype) != "torch.float32":
                    _record_error(errors, f"scale tensor dtype mismatch for {name}: {tensor.dtype}", strict=strict)
                expected_shape = _safe_shape(scale_meta.get("shape") or [])
                got_shape = _safe_shape(tensor.shape)
                if got_shape != expected_shape:
                    _record_error(errors, f"scale tensor shape mismatch for {name}: expected {expected_shape}, got {got_shape}", strict=strict)
            report.scale_files[rel_path] = {"tensor_count": len(keys), "referenced_tensor_count": len(expected_names)}
            report.scale_tensor_count += len(keys)

    if payload_report:
        expected_count = int(payload_report.get("quantized_tensor_count") or -1)
        if expected_count != len(rows):
            _record_error(errors, f"payload report tensor count mismatch: expected {len(rows)}, got {expected_count}", strict=strict)
    selection = tensor_index.get("selection") or {}
    selected_count = int(selection.get("quantized_tensor_count") or -1)
    if selected_count != len(rows):
        _record_error(errors, f"tensor index selection count mismatch: expected {len(rows)}, got {selected_count}", strict=strict)

    if report.payload_tensor_count != len(payload_targets):
        _record_error(errors, f"payload tensor count mismatch: expected {len(payload_targets)}, got {report.payload_tensor_count}", strict=strict)
    if report.scale_tensor_count != len(scale_targets):
        _record_error(errors, f"scale tensor count mismatch: expected {len(scale_targets)}, got {report.scale_tensor_count}", strict=strict)

    report.payload_checked = True
    report.errors = errors
    report.status = "valid" if not errors else "invalid"
    return report
