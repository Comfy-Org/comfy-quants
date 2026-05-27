"""INT4 runtime readiness gate aggregation.

This module aggregates validation reports produced outside the quantization
writer path.  It deliberately does not import ComfyUI, comfy-kitchen,
DeepCompressor, Nunchaku, or any fused runtime.  The report is a checklist for
runtime parity work; it must not be treated as proof of publishability by
itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_quants.backends.svdquant_runtime_like_validation import (
    SVDQUANT_RUNTIME_LIKE_VALIDATION_PASSED,
    SVDQUANT_RUNTIME_LIKE_VALIDATION_SCOPE,
)
from comfy_quants.utils.jsonio import read_json


INT4_RUNTIME_READINESS_SCHEMA_VERSION = "int4_runtime_readiness_report.v1"
DEFAULT_INT4_RUNTIME_READINESS_REPORT_FILENAME = "int4_runtime_readiness_report.json"
SINGLE_LAYER_FIXTURE_VALIDATION_SCOPE = "single_layer_runtime_fixture_output_only"
SINGLE_LAYER_FIXTURE_VALIDATION_STATE = "single_layer_fixture_output_passed"
MIXED_DISPATCH_VALIDATION_SCOPE = "mixed_svdquant_w4a4_awq_w4a16_dispatch"
FULL_INFERENCE_VALIDATION_SCOPE = "full_qwen_image_edit_png_inference"


def _path_string(path: str | Path | None) -> str | None:
    return None if path is None else str(Path(path).expanduser())


def _new_gate(name: str, *, required: bool, report_path: str | Path | None, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "required": bool(required),
        "description": description,
        "report_path": _path_string(report_path),
        "status": "missing" if report_path is None else "failed",
        "passed": False,
        "errors": [],
        "evidence": {},
    }


def _read_report(path: str | Path | None, gate: dict[str, Any]) -> dict[str, Any] | None:
    if path is None:
        gate["errors"].append("report path was not provided")
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        gate["status"] = "missing"
        gate["errors"].append(f"report file is missing: {p}")
        return None
    try:
        value = read_json(p)
    except Exception as exc:  # noqa: BLE001 - reports are external inputs
        gate["errors"].append(f"failed to read report JSON: {exc}")
        return None
    if not isinstance(value, dict):
        gate["errors"].append("report JSON must be an object")
        return None
    return value


def _finish_gate(gate: dict[str, Any]) -> dict[str, Any]:
    if gate["errors"]:
        gate["passed"] = False
        gate["status"] = "missing" if gate["status"] == "missing" else "failed"
    else:
        gate["passed"] = True
        gate["status"] = "passed"
    return gate


def _single_layer_report_gate(
    name: str,
    *,
    report_path: str | Path | None,
    expected_format_hint: str,
    expected_scope: str,
    expected_validation_state: str,
) -> dict[str, Any]:
    gate = _new_gate(
        name,
        required=True,
        report_path=report_path,
        description=f"{expected_format_hint} external single-layer runtime report satisfies its parity gate",
    )
    report = _read_report(report_path, gate)
    if report is None:
        return _finish_gate(gate)

    gate["evidence"] = {
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "validation_scope": report.get("validation_scope"),
        "external_runtime_validation": report.get("external_runtime_validation"),
        "publishable_svdquant_gptq": report.get("publishable_svdquant_gptq"),
        "max_abs_error": report.get("max_abs_error"),
        "max_relative_error": report.get("max_relative_error"),
        "metrics": report.get("metrics"),
    }
    if report.get("status") != "passed":
        gate["errors"].append(f"expected report status 'passed', got {report.get('status')!r}")
    if report.get("validation_scope") != expected_scope:
        gate["errors"].append(f"expected validation_scope {expected_scope!r}, got {report.get('validation_scope')!r}")
    if report.get("external_runtime_validation") != expected_validation_state:
        gate["errors"].append(
            "expected external_runtime_validation "
            f"{expected_validation_state!r}, got {report.get('external_runtime_validation')!r}"
        )
    if report.get("publishable_svdquant_gptq") is not False:
        gate["errors"].append("single-layer fixture validation reports must keep publishable_svdquant_gptq false")
    return _finish_gate(gate)


def _external_scope_gate(
    name: str,
    *,
    report_path: str | Path | None,
    expected_scope: str,
    description: str,
) -> dict[str, Any]:
    gate = _new_gate(name, required=True, report_path=report_path, description=description)
    report = _read_report(report_path, gate)
    if report is None:
        return _finish_gate(gate)

    gate["evidence"] = {
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "validation_scope": report.get("validation_scope"),
        "external_runtime_validation": report.get("external_runtime_validation"),
        "png_path": report.get("png_path"),
        "image_path": report.get("image_path"),
    }
    if report.get("status") != "passed":
        gate["errors"].append(f"expected report status 'passed', got {report.get('status')!r}")
    if report.get("validation_scope") != expected_scope:
        gate["errors"].append(f"expected validation_scope {expected_scope!r}, got {report.get('validation_scope')!r}")
    return _finish_gate(gate)


def build_int4_runtime_readiness_report(
    *,
    svdquant_report_path: str | Path | None = None,
    awq_report_path: str | Path | None = None,
    mixed_dispatch_report_path: str | Path | None = None,
    full_inference_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable INT4 runtime readiness checklist report."""

    gates = [
        _single_layer_report_gate(
            "svdquant_w4a4_single_layer_runtime_parity",
            report_path=svdquant_report_path,
            expected_format_hint="SVDQuant W4A4",
            expected_scope=SVDQUANT_RUNTIME_LIKE_VALIDATION_SCOPE,
            expected_validation_state=SVDQUANT_RUNTIME_LIKE_VALIDATION_PASSED,
        ),
        _single_layer_report_gate(
            "awq_w4a16_single_layer_runtime_parity",
            report_path=awq_report_path,
            expected_format_hint="AWQ W4A16",
            expected_scope=SINGLE_LAYER_FIXTURE_VALIDATION_SCOPE,
            expected_validation_state=SINGLE_LAYER_FIXTURE_VALIDATION_STATE,
        ),
        _external_scope_gate(
            "mixed_svdquant_w4a4_awq_w4a16_dispatch",
            report_path=mixed_dispatch_report_path,
            expected_scope=MIXED_DISPATCH_VALIDATION_SCOPE,
            description="target runtime dispatches SVDQuant W4A4 and AWQ W4A16 layers in one mixed checkpoint",
        ),
        _external_scope_gate(
            "full_qwen_image_edit_png_inference",
            report_path=full_inference_report_path,
            expected_scope=FULL_INFERENCE_VALIDATION_SCOPE,
            description="target runtime loads the exported single checkpoint and produces a PNG image",
        ),
    ]
    required_gates = [gate for gate in gates if gate["required"]]
    passed_required = [gate for gate in required_gates if gate["passed"]]
    missing_gates = [gate["name"] for gate in required_gates if gate["status"] == "missing"]
    failed_gates = [gate["name"] for gate in required_gates if gate["status"] == "failed"]
    all_required_passed = len(passed_required) == len(required_gates)
    return {
        "schema_version": INT4_RUNTIME_READINESS_SCHEMA_VERSION,
        "status": "passed" if all_required_passed else "blocked",
        "publishable_svdquant_gptq": False,
        "publishable_candidate_after_manual_review": all_required_passed,
        "manual_publishable_review_required": True,
        "required_gate_count": len(required_gates),
        "passed_required_gate_count": len(passed_required),
        "missing_gates": missing_gates,
        "failed_gates": failed_gates,
        "gates": gates,
        "does_not_validate": [
            "the quantization writer itself",
            "that external reports were produced by a trusted environment",
            "image quality beyond the supplied full-inference report",
            "safe promotion of publishable_svdquant_gptq without manual review",
        ],
    }
