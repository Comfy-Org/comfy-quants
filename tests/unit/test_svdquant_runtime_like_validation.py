import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.cli.main import main
from comfy_quants.utils.jsonio import write_json


def _metric(max_abs_error=0.0, max_relative_error=0.0):
    return {
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "mean_abs_error": max_abs_error,
        "rmse": max_abs_error,
    }


def _write_harness_report(path: Path, *, status="runtime_output_written", full_error=0.0, layout="runtime-packed"):
    write_json(
        path,
        {
            "status": status,
            "assignment_layout": layout,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "validation_scope": "single_layer_svdquant_w4a4_linear_forward",
            "external_runtime_validation": "not_validated_by_this_script",
            "publishable_svdquant_gptq": False,
            "component_diagnostics": {
                "forward_vs_quantize_forward_quant": _metric(),
                "dense_main_replay": {
                    "main_vs_decoded_activation_group_dtype_fma_runtime_like": _metric(),
                },
                "lowrank_runtime_like_replay": {
                    "lowrank_vs_natural_runtime_dtype_down_up": _metric(),
                },
                "bias_runtime_like_replay": {
                    "bias_vs_runtime_dtype_bias_broadcast": _metric(),
                },
                "full_runtime_like_replay": {
                    "full_vs_decoded_main_bias_lowrank_runtime_dtype_epilogue": _metric(
                        max_abs_error=full_error,
                        max_relative_error=full_error,
                    ),
                },
            },
        },
    )


class TestSVDQuantRuntimeLikeValidation(unittest.TestCase):
    def test_validator_passes_exact_runtime_like_harness_report_and_cli_writes_report(self):
        from comfy_quants.backends.svdquant_runtime_like_validation import (
            SVDQUANT_RUNTIME_LIKE_VALIDATION_PASSED,
            SVDQUANT_RUNTIME_LIKE_VALIDATION_SCOPE,
            validate_svdquant_runtime_like_harness_report,
        )

        with tempfile.TemporaryDirectory() as tmp:
            harness_report = Path(tmp) / "harness_report.json"
            _write_harness_report(harness_report)

            report = validate_svdquant_runtime_like_harness_report(harness_report)
            self.assertEqual(report.status, "passed")
            self.assertEqual(report.validation_scope, SVDQUANT_RUNTIME_LIKE_VALIDATION_SCOPE)
            self.assertEqual(report.external_runtime_validation, SVDQUANT_RUNTIME_LIKE_VALIDATION_PASSED)
            self.assertIs(report.publishable_svdquant_gptq, False)
            self.assertEqual(report.metrics["full_runtime_like"]["max_abs_error"], 0.0)

            out_dir = Path(tmp) / "validation"
            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "validate-svdquant-runtime-like-report",
                        "--harness-report",
                        str(harness_report),
                        "--out",
                        str(out_dir),
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            cli_report = json.loads(captured.getvalue())
            self.assertEqual(cli_report["status"], "passed")
            self.assertEqual(cli_report["external_runtime_validation"], SVDQUANT_RUNTIME_LIKE_VALIDATION_PASSED)
            self.assertTrue((out_dir / "svdquant_runtime_like_validation_report.json").exists())

    def test_validator_fails_when_full_runtime_like_metric_exceeds_tolerance(self):
        from comfy_quants.backends.svdquant_runtime_like_validation import (
            SVDQUANT_RUNTIME_LIKE_VALIDATION_FAILED,
            validate_svdquant_runtime_like_harness_report,
        )

        with tempfile.TemporaryDirectory() as tmp:
            harness_report = Path(tmp) / "harness_report.json"
            _write_harness_report(harness_report, full_error=0.25)

            report = validate_svdquant_runtime_like_harness_report(harness_report, atol=1.0e-6, rtol=1.0e-6)

            self.assertEqual(report.status, "failed")
            self.assertEqual(report.external_runtime_validation, SVDQUANT_RUNTIME_LIKE_VALIDATION_FAILED)
            self.assertTrue(any("full_runtime_like_replay" in error for error in report.errors))

    def test_validator_rejects_non_packed_layout_by_default(self):
        from comfy_quants.backends.svdquant_runtime_like_validation import validate_svdquant_runtime_like_harness_report

        with tempfile.TemporaryDirectory() as tmp:
            harness_report = Path(tmp) / "harness_report.json"
            _write_harness_report(harness_report, layout="natural")

            report = validate_svdquant_runtime_like_harness_report(harness_report)

            self.assertEqual(report.status, "failed")
            self.assertTrue(any("packed assignment_layout" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
