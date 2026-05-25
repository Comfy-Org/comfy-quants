import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.cli.main import main
from comfy_quants.utils.jsonio import write_json


def _metric():
    return {
        "max_abs_error": 0.0,
        "max_relative_error": 0.0,
        "mean_abs_error": 0.0,
        "rmse": 0.0,
    }


def _deps():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        return None
    return load_file, save_file


class TestInt4RuntimeReadiness(unittest.TestCase):
    def setUp(self):
        deps = _deps()
        if deps is None:
            self.skipTest("safetensors is required")
        self.load_file, self.save_file = deps

    def _write_validation_report(self, tmp: str, *, kind: str) -> Path:
        from comfy_quants.algorithms.awq_w4a16.runtime_fixture import (
            AwqW4A16RuntimeFixtureConfig,
            write_awq_w4a16_runtime_fixture,
        )
        from comfy_quants.backends.runtime_fixture_validation import validate_runtime_fixture_output
        from comfy_quants.backends.svdquant_runtime_like_validation import validate_svdquant_runtime_like_harness_report

        fixture_dir = Path(tmp) / f"{kind}_fixture"
        output_path = Path(tmp) / f"{kind}_runtime_output.safetensors"
        validation_path = Path(tmp) / f"{kind}_validation_report.json"
        if kind == "svdquant":
            harness_report = Path(tmp) / "svdquant_harness_report.json"
            write_json(
                harness_report,
                {
                    "status": "runtime_output_written",
                    "assignment_layout": "runtime-packed",
                    "dtype": "bfloat16",
                    "device": "cuda:0",
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
                            "full_vs_decoded_main_bias_lowrank_runtime_dtype_epilogue": _metric(),
                        },
                    },
                },
            )
            report = validate_svdquant_runtime_like_harness_report(harness_report).to_dict()
            write_json(validation_path, report)
            return validation_path
        elif kind == "awq":
            written = write_awq_w4a16_runtime_fixture(
                fixture_dir,
                config=AwqW4A16RuntimeFixtureConfig(seed=7201, n=8, batch=2, scale_dtype="float32"),
                hash_fixture=False,
            )
        else:  # pragma: no cover - test helper misuse
            raise ValueError(kind)
        fixture_tensors = self.load_file(str(written.fixture_path))
        self.save_file({"runtime.output": fixture_tensors["fixture.expected_output"].clone()}, str(output_path))
        report = validate_runtime_fixture_output(written.fixture_path, output_path).to_dict()
        write_json(validation_path, report)
        return validation_path

    def test_readiness_blocks_when_only_single_layer_reports_pass(self):
        from comfy_quants.backends.int4_runtime_readiness import (
            INT4_RUNTIME_READINESS_SCHEMA_VERSION,
            build_int4_runtime_readiness_report,
        )

        with tempfile.TemporaryDirectory() as tmp:
            svd_report = self._write_validation_report(tmp, kind="svdquant")
            awq_report = self._write_validation_report(tmp, kind="awq")

            report = build_int4_runtime_readiness_report(
                svdquant_report_path=svd_report,
                awq_report_path=awq_report,
            )

            self.assertEqual(report["schema_version"], INT4_RUNTIME_READINESS_SCHEMA_VERSION)
            self.assertEqual(report["status"], "blocked")
            self.assertIs(report["publishable_svdquant_gptq"], False)
            self.assertIs(report["publishable_candidate_after_manual_review"], False)
            self.assertEqual(report["passed_required_gate_count"], 2)
            self.assertEqual(report["required_gate_count"], 4)
            self.assertIn("mixed_svdquant_w4a4_awq_w4a16_dispatch", report["missing_gates"])
            self.assertIn("full_qwen_image_edit_png_inference", report["missing_gates"])
            gate_status = {gate["name"]: gate["status"] for gate in report["gates"]}
            self.assertEqual(gate_status["svdquant_w4a4_single_layer_runtime_parity"], "passed")
            self.assertEqual(gate_status["awq_w4a16_single_layer_runtime_parity"], "passed")

            out_dir = Path(tmp) / "readiness"
            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "validate-int4-runtime-readiness",
                        "--svdquant-report",
                        str(svd_report),
                        "--awq-report",
                        str(awq_report),
                        "--out",
                        str(out_dir),
                        "--json",
                    ]
                )

            self.assertEqual(rc, 2)
            cli_report = json.loads(captured.getvalue())
            self.assertEqual(cli_report["status"], "blocked")
            self.assertTrue((out_dir / "int4_runtime_readiness_report.json").exists())

    def test_readiness_can_mark_all_required_gates_passed_but_not_publishable(self):
        from comfy_quants.backends.int4_runtime_readiness import (
            FULL_INFERENCE_VALIDATION_SCOPE,
            MIXED_DISPATCH_VALIDATION_SCOPE,
            build_int4_runtime_readiness_report,
        )

        with tempfile.TemporaryDirectory() as tmp:
            svd_report = self._write_validation_report(tmp, kind="svdquant")
            awq_report = self._write_validation_report(tmp, kind="awq")
            mixed_report = Path(tmp) / "mixed_dispatch_report.json"
            full_report = Path(tmp) / "full_inference_report.json"
            write_json(
                mixed_report,
                {
                    "schema_version": "external_mixed_dispatch_report.v1",
                    "status": "passed",
                    "validation_scope": MIXED_DISPATCH_VALIDATION_SCOPE,
                },
            )
            write_json(
                full_report,
                {
                    "schema_version": "external_full_inference_report.v1",
                    "status": "passed",
                    "validation_scope": FULL_INFERENCE_VALIDATION_SCOPE,
                    "png_path": "outputs/qwen_image_edit_smoke.png",
                },
            )

            report = build_int4_runtime_readiness_report(
                svdquant_report_path=svd_report,
                awq_report_path=awq_report,
                mixed_dispatch_report_path=mixed_report,
                full_inference_report_path=full_report,
            )

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["passed_required_gate_count"], 4)
            self.assertEqual(report["missing_gates"], [])
            self.assertEqual(report["failed_gates"], [])
            self.assertIs(report["publishable_candidate_after_manual_review"], True)
            self.assertIs(report["manual_publishable_review_required"], True)
            self.assertIs(report["publishable_svdquant_gptq"], False)

    def test_readiness_rejects_failed_single_layer_report(self):
        from comfy_quants.backends.int4_runtime_readiness import build_int4_runtime_readiness_report

        with tempfile.TemporaryDirectory() as tmp:
            failed_report = Path(tmp) / "failed_svd_report.json"
            write_json(
                failed_report,
                {
                    "schema_version": "runtime_fixture_output_validation_report.v1",
                    "status": "failed",
                    "validation_scope": "single_layer_runtime_fixture_output_only",
                    "external_runtime_validation": "single_layer_fixture_output_failed",
                    "publishable_svdquant_gptq": False,
                },
            )

            report = build_int4_runtime_readiness_report(svdquant_report_path=failed_report)
            self.assertEqual(report["status"], "blocked")
            self.assertIn("svdquant_w4a4_single_layer_runtime_parity", report["failed_gates"])
            svd_gate = next(gate for gate in report["gates"] if gate["name"] == "svdquant_w4a4_single_layer_runtime_parity")
            self.assertTrue(svd_gate["errors"])


if __name__ == "__main__":
    unittest.main()
