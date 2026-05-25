"""Calibration dataset schemas and JSONL loading helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalibrationCase:
    """A text-to-image or image-edit calibration case descriptor."""

    case_id: str
    prompt: str
    image: str | None = None
    edit_type: str | None = None
    language: str | None = None
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    input_path = Path(path).expanduser()
    rows: list[Mapping[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {input_path}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"calibration row must be a JSON object at {input_path}:{line_number}")
            rows.append(row)
    return rows


def _optional_string(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"calibration field {key!r} must be a string when provided")
    return value


def _resolve_image(image: str | None, image_root: str | Path | None) -> str | None:
    if not image:
        return None
    image_path = Path(image)
    if image_root is not None and not image_path.is_absolute():
        return str(Path(image_root).expanduser() / image_path)
    return image


def calibration_case_from_mapping(row: Mapping[str, Any], *, image_root: str | Path | None = None) -> CalibrationCase:
    """Build a normalized calibration case from one prompt/edit JSON object."""
    case_id = row.get("case_id", row.get("id"))
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("calibration row requires a non-empty string id or case_id")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"calibration row {case_id!r} requires a non-empty prompt")
    known_keys = {"id", "case_id", "prompt", "image", "edit_type", "language", "category"}
    metadata = {str(key): value for key, value in row.items() if key not in known_keys}
    return CalibrationCase(
        case_id=case_id,
        prompt=prompt,
        image=_resolve_image(_optional_string(row, "image"), image_root),
        edit_type=_optional_string(row, "edit_type"),
        language=_optional_string(row, "language"),
        category=_optional_string(row, "category"),
        metadata=metadata,
    )


def load_calibration_cases(
    path: str | Path,
    *,
    image_root: str | Path | None = None,
    edit_types: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[CalibrationCase]:
    """Load prompt or edit calibration cases from a JSONL file."""
    allowed_edit_types = {value for value in (edit_types or []) if value}
    cases: list[CalibrationCase] = []
    for row in _read_jsonl(path):
        case = calibration_case_from_mapping(row, image_root=image_root)
        if allowed_edit_types and case.edit_type not in allowed_edit_types:
            continue
        cases.append(case)
        if limit is not None and len(cases) >= int(limit):
            break
    return cases


def _manifest_path(manifest_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    manifest_relative = manifest_dir / path
    if manifest_relative.exists() or not path.exists():
        return manifest_relative
    return path


def load_calibration_manifest_cases(path: str | Path, *, limit: int | None = None) -> list[CalibrationCase]:
    """Load normalized calibration cases referenced by a calibration manifest."""
    manifest_path = Path(path).expanduser()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"calibration manifest is invalid JSON: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"calibration manifest must be a JSON object: {manifest_path}")
    manifest_dir = manifest_path.parent
    image_root = _manifest_path(manifest_dir, manifest.get("image_root"))
    edit_types = manifest.get("edit_types") if isinstance(manifest.get("edit_types"), list) else []
    cases: list[CalibrationCase] = []
    prompt_set = _manifest_path(manifest_dir, manifest.get("prompt_set"))
    edit_set = _manifest_path(manifest_dir, manifest.get("edit_set"))
    remaining = None if limit is None else int(limit)
    if prompt_set is not None:
        prompt_cases = load_calibration_cases(prompt_set, limit=remaining)
        cases.extend(prompt_cases)
        if remaining is not None:
            remaining = max(0, remaining - len(prompt_cases))
    if edit_set is not None and (remaining is None or remaining > 0):
        cases.extend(load_calibration_cases(edit_set, image_root=image_root, edit_types=edit_types, limit=remaining))
    if not cases:
        raise ValueError(f"calibration manifest references no usable cases: {manifest_path}")
    return cases


def write_calibration_cases_jsonl(path: str | Path, cases: Iterable[CalibrationCase]) -> None:
    """Write normalized calibration cases as JSONL."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
