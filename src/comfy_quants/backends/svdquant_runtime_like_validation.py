"""Validation for SVDQuant W4A4 fused-runtime parity reports.

This module reads a JSON report produced by an external single-layer harness and
checks the runtime-like component metrics that matter for SVDQuant W4A4 fused
execution.  It does not import or call a model runtime.  Passing this validator
is still only a single-layer signal; full checkpoint export and image inference
must be validated separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from comfy_quants.utils.jsonio import read_json


SVDQUANT_RUNTIME_LIKE_VALIDATION_SCHEMA_VERSION = "svdquant_w4a4_runtime_like_validation_report.v1"
SVDQUANT_RUNTIME_LIKE_VALIDATION_SCOPE = "single_layer_svdquant_w4a4_runtime_like_parity"
SVDQUANT_RUNTIME_LIKE_VALIDATION_PASSED = "single_layer_svdquant_w4a4_runtime_like_passed"
SVDQUANT_RUNTIME_LIKE_VALIDATION_FAILED = "single_layer_svdquant_w4a4_runtime_like_failed"
DEFAULT_SVDQUANT_RUNTIME_LIKE_VALIDATION_REPORT_FILENAME = "svdquant_runtime_like_validation_report.json"


_REQUIRED_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("forward_replay", ("forward_vs_quantize_forward_quant",)),
    (
        "main_runtime_like",
        ("dense_main_replay", "main_vs_decoded_activation_group_dtype_fma_runtime_like"),
    ),
    (
        "lowrank_runtime_like",
        ("lowrank_runtime_like_replay", "lowrank_vs_natural_runtime_dtype_down_up"),
    ),
    (
        "bias_runtime_like",
        ("bias_runtime_like_replay", "bias_vs_runtime_dtype_bias_broadcast"),
    ),
    (
        "full_runtime_like",
        ("full_runtime_like_replay", "full_vs_decoded_main_bias_lowrank_runtime_dtype_epilogue"),
    ),
)


@dataclass
class SVDQuantRuntimeLikeValidationReport:
    """JSON-serializable SVDQuant W4A4 runtime-like report validation."""

    status: str
    harness_report_path: str
    schema_version: str = SVDQUANT_RUNTIME_LIKE_VALIDATION_SCHEMA_VERSION
    validation_scope: str = SVDQUANT_RUNTIME_LIKE_VALIDATION_SCOPE
    external_runtime_validation: str = SVDQUANT_RUNTIME_LIKE_VALIDATION_FAILED
    publishable_svdquant_gptq: bool = False
    atol: float = 1.0e-6
    rtol: float = 1.0e-6
    expected_dtype: str | None = "bfloat16"
    dtype: str | None = None
    device: str | None = None
    assignment_layout: str | None = None
    metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)
    source_status: str | None = None
    source_validation_scope: str | None = None
    source_external_runtime_validation: str | None = None
    errors: list[str] = field(default_factory=list)
    does_not_validate: list[str] = field(
        default_factory=lambda: [
            "full Qwen-Image/Edit model load",
            "mixed SVDQuant W4A4 plus AWQ W4A16 dispatch",
            "full image inference PNG quality",
            "publishable SVDQuant+GPTQ checkpoint status",
            "that the external harness came from a trusted environment",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _new_report(
    harness_report_path: Path,
    *,
    atol: float,
    rtol: float,
    expected_dtype: str | None,
) -> SVDQuantRuntimeLikeValidationReport:
    return SVDQuantRuntimeLikeValidationReport(
        status="failed",
        harness_report_path=str(harness_report_path),
        atol=float(atol),
        rtol=float(rtol),
        expected_dtype=expected_dtype,
    )


def _lookup(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _metric_value(metric: dict[str, Any], key: str) -> float | None:
    value = metric.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_svdquant_runtime_like_harness_report(
    harness_report_path: str | Path,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
    expected_dtype: str | None = "bfloat16",
    require_packed_layout: bool = True,
) -> SVDQuantRuntimeLikeValidationReport:
    """Validate single-layer SVDQuant W4A4 runtime-like metrics from a harness report."""

    path = Path(harness_report_path).expanduser()
    report = _new_report(path, atol=float(atol), rtol=float(rtol), expected_dtype=expected_dtype)
    if not path.is_file():
        report.errors.append(f"harness report file is missing: {path}")
        return report
    if float(atol) < 0:
        report.errors.append(f"atol must be non-negative, got {atol}")
        return report
    if float(rtol) < 0:
        report.errors.append(f"rtol must be non-negative, got {rtol}")
        return report

    try:
        source = read_json(path)
    except Exception as exc:  # noqa: BLE001 - external reports are untrusted inputs
        report.errors.append(f"failed to read harness report JSON: {exc}")
        return report
    if not isinstance(source, dict):
        report.errors.append("harness report JSON must be an object")
        return report

    report.source_status = None if source.get("status") is None else str(source.get("status"))
    report.source_validation_scope = None if source.get("validation_scope") is None else str(source.get("validation_scope"))
    report.source_external_runtime_validation = (
        None if source.get("external_runtime_validation") is None else str(source.get("external_runtime_validation"))
    )
    report.dtype = None if source.get("dtype") is None else str(source.get("dtype"))
    report.device = None if source.get("device") is None else str(source.get("device"))
    report.assignment_layout = None if source.get("assignment_layout") is None else str(source.get("assignment_layout"))

    if source.get("status") != "runtime_output_written":
        report.errors.append(f"expected harness status 'runtime_output_written', got {source.get('status')!r}")
    if source.get("publishable_svdquant_gptq") is not False:
        report.errors.append("source harness report must keep publishable_svdquant_gptq false")
    if expected_dtype is not None and source.get("dtype") != expected_dtype:
        report.errors.append(f"expected dtype {expected_dtype!r}, got {source.get('dtype')!r}")
    if require_packed_layout:
        layout = str(source.get("assignment_layout") or "")
        if not layout.endswith("-packed"):
            report.errors.append(f"expected a packed assignment_layout, got {source.get('assignment_layout')!r}")

    component_diagnostics = source.get("component_diagnostics")
    if not isinstance(component_diagnostics, dict):
        report.errors.append("component_diagnostics object is missing from harness report")
    else:
        for name, metric_path in _REQUIRED_METRICS:
            metric = _lookup(component_diagnostics, metric_path)
            if not isinstance(metric, dict):
                report.metrics[name] = {
                    "max_abs_error": None,
                    "max_relative_error": None,
                    "mean_abs_error": None,
                    "rmse": None,
                }
                report.errors.append(f"required metric is missing: {'.'.join(metric_path)}")
                continue

            max_abs = _metric_value(metric, "max_abs_error")
            max_rel = _metric_value(metric, "max_relative_error")
            mean_abs = _metric_value(metric, "mean_abs_error")
            rmse = _metric_value(metric, "rmse")
            report.metrics[name] = {
                "max_abs_error": max_abs,
                "max_relative_error": max_rel,
                "mean_abs_error": mean_abs,
                "rmse": rmse,
            }
            if max_abs is None:
                report.errors.append(f"metric {'.'.join(metric_path)} has no numeric max_abs_error")
            elif max_abs > float(atol):
                report.errors.append(
                    f"metric {'.'.join(metric_path)} max_abs_error={max_abs} exceeds atol={float(atol)}"
                )
            if max_rel is None:
                report.errors.append(f"metric {'.'.join(metric_path)} has no numeric max_relative_error")
            elif max_rel > float(rtol):
                report.errors.append(
                    f"metric {'.'.join(metric_path)} max_relative_error={max_rel} exceeds rtol={float(rtol)}"
                )

    if not report.errors:
        report.status = "passed"
        report.external_runtime_validation = SVDQUANT_RUNTIME_LIKE_VALIDATION_PASSED
    return report
