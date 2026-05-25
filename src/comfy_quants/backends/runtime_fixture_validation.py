"""Runtime fixture output validation helpers.

These helpers compare an external runtime's saved output tensor against a
repository-generated fixture oracle.  They deliberately do not import ComfyUI,
Nunchaku, comfy-kitchen, or any model runtime.  A passing report is only a
single-layer fixture parity signal; it is not a publishable mixed-checkpoint
validation claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


RUNTIME_FIXTURE_OUTPUT_VALIDATION_SCHEMA_VERSION = "runtime_fixture_output_validation_report.v1"
RUNTIME_FIXTURE_OUTPUT_VALIDATION_SCOPE = "single_layer_runtime_fixture_output_only"
DEFAULT_RUNTIME_FIXTURE_OUTPUT_VALIDATION_REPORT_FILENAME = "runtime_fixture_output_validation_report.json"


@dataclass
class RuntimeFixtureOutputValidationReport:
    """JSON-serializable report for one external fixture output comparison."""

    status: str
    fixture_path: str
    output_path: str
    expected_tensor: str
    actual_tensor: str
    schema_version: str = RUNTIME_FIXTURE_OUTPUT_VALIDATION_SCHEMA_VERSION
    validation_scope: str = RUNTIME_FIXTURE_OUTPUT_VALIDATION_SCOPE
    publishable_svdquant_gptq: bool = False
    external_runtime_validation: str = "single_layer_fixture_not_publishable"
    atol: float = 1.0e-4
    rtol: float = 1.0e-4
    expected_shape: list[int] | None = None
    actual_shape: list[int] | None = None
    expected_dtype: str | None = None
    actual_dtype: str | None = None
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    max_relative_error: float | None = None
    fixture_metadata: dict[str, str] = field(default_factory=dict)
    fixture_tensor_count: int | None = None
    output_tensor_count: int | None = None
    errors: list[str] = field(default_factory=list)
    does_not_validate: list[str] = field(
        default_factory=lambda: [
            "full Qwen-Image/Edit model load",
            "mixed SVDQuant W4A4 plus AWQ W4A16 dispatch",
            "ComfyUI node/runtime registration",
            "full image inference PNG quality",
            "publishable SVDQuant+GPTQ checkpoint status",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImportError("torch is required to validate runtime fixture outputs") from exc
    return torch


def _require_safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - dependency should be installed by package metadata
        raise ImportError("safetensors is required to validate runtime fixture outputs") from exc
    return safe_open, load_file


def _shape(value: Any) -> list[int]:
    return [int(dim) for dim in value.shape]


def _metadata_and_keys(path: Path) -> tuple[dict[str, str], list[str]]:
    safe_open, _load_file = _require_safetensors()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = sorted(str(key) for key in handle.keys())
    return metadata, keys


def _new_report(
    *,
    fixture_path: Path,
    output_path: Path,
    expected_tensor: str,
    actual_tensor: str,
    atol: float,
    rtol: float,
) -> RuntimeFixtureOutputValidationReport:
    return RuntimeFixtureOutputValidationReport(
        status="failed",
        fixture_path=str(fixture_path),
        output_path=str(output_path),
        expected_tensor=str(expected_tensor),
        actual_tensor=str(actual_tensor),
        atol=float(atol),
        rtol=float(rtol),
    )


def validate_runtime_fixture_output(
    fixture_path: str | Path,
    output_path: str | Path,
    *,
    expected_tensor: str = "fixture.expected_output",
    actual_tensor: str = "runtime.output",
    atol: float = 1.0e-4,
    rtol: float = 1.0e-4,
) -> RuntimeFixtureOutputValidationReport:
    """Compare an external runtime output tensor against a fixture oracle.

    ``fixture_path`` is a safetensors fixture produced by
    ``make-int4-runtime-fixture`` or ``make-awq-runtime-fixture``.  ``output_path``
    is a safetensors file written by an external harness.  The external file is
    expected to contain ``actual_tensor``; by default that name is
    ``runtime.output``.  The fixture is expected to contain ``expected_tensor``;
    by default that name is ``fixture.expected_output``.
    """

    torch = _require_torch()
    _safe_open, load_file = _require_safetensors()
    fixture = Path(fixture_path).expanduser()
    output = Path(output_path).expanduser()
    report = _new_report(
        fixture_path=fixture,
        output_path=output,
        expected_tensor=expected_tensor,
        actual_tensor=actual_tensor,
        atol=float(atol),
        rtol=float(rtol),
    )

    if not fixture.is_file():
        report.errors.append(f"fixture file is missing: {fixture}")
        return report
    if not output.is_file():
        report.errors.append(f"runtime output file is missing: {output}")
        return report
    if float(atol) < 0:
        report.errors.append(f"atol must be non-negative, got {atol}")
        return report
    if float(rtol) < 0:
        report.errors.append(f"rtol must be non-negative, got {rtol}")
        return report

    fixture_metadata, fixture_keys = _metadata_and_keys(fixture)
    output_metadata, output_keys = _metadata_and_keys(output)
    _ = output_metadata  # currently reserved for future external harness metadata checks
    report.fixture_metadata = fixture_metadata
    report.fixture_tensor_count = len(fixture_keys)
    report.output_tensor_count = len(output_keys)

    if expected_tensor not in fixture_keys:
        report.errors.append(f"expected tensor is missing from fixture: {expected_tensor}")
        return report
    if actual_tensor not in output_keys:
        report.errors.append(f"actual tensor is missing from runtime output: {actual_tensor}")
        return report

    fixture_tensors = load_file(str(fixture))
    output_tensors = load_file(str(output))
    expected = fixture_tensors[expected_tensor]
    actual = output_tensors[actual_tensor]
    report.expected_shape = _shape(expected)
    report.actual_shape = _shape(actual)
    report.expected_dtype = str(expected.dtype)
    report.actual_dtype = str(actual.dtype)

    if report.expected_shape != report.actual_shape:
        report.errors.append(f"shape mismatch: expected {report.expected_shape}, got {report.actual_shape}")
        return report

    expected_f = expected.detach().to(dtype=torch.float32)
    actual_f = actual.detach().to(dtype=torch.float32)
    if bool((~torch.isfinite(actual_f)).any().item()):
        report.errors.append("actual tensor contains NaN or Inf values")
        return report
    if bool((~torch.isfinite(expected_f)).any().item()):
        report.errors.append("expected tensor contains NaN or Inf values")
        return report

    diff = (actual_f - expected_f).abs()
    report.max_abs_error = float(diff.max().item()) if int(diff.numel()) else 0.0
    report.mean_abs_error = float(diff.mean().item()) if int(diff.numel()) else 0.0
    denom = expected_f.abs().clamp_min(1.0e-12)
    rel = diff / denom
    report.max_relative_error = float(rel.max().item()) if int(rel.numel()) else 0.0
    passed = bool(torch.allclose(actual_f, expected_f, atol=float(atol), rtol=float(rtol)))
    if passed:
        report.status = "passed"
        report.external_runtime_validation = "single_layer_fixture_output_passed"
    else:
        report.status = "failed"
        report.external_runtime_validation = "single_layer_fixture_output_failed"
        report.errors.append(
            "runtime output differs from fixture oracle: "
            f"max_abs_error={report.max_abs_error}, max_relative_error={report.max_relative_error}, "
            f"atol={float(atol)}, rtol={float(rtol)}"
        )
    return report
