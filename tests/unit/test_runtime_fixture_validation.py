import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.cli.main import main


def _deps():
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError:
        return None
    return torch, load_file, save_file


class TestRuntimeFixtureValidation(unittest.TestCase):
    def setUp(self):
        deps = _deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.load_file, self.save_file = deps

    def _write_svd_fixture_and_matching_output(self, tmp: str):
        from comfy_quants.algorithms.int4_svdquant.runtime_fixture import (
            SVDQuantW4A4RuntimeFixtureConfig,
            write_svdquant_w4a4_runtime_fixture,
        )

        fixture_dir = Path(tmp) / "fixture"
        output_path = Path(tmp) / "runtime_output.safetensors"
        written = write_svdquant_w4a4_runtime_fixture(
            fixture_dir,
            config=SVDQuantW4A4RuntimeFixtureConfig(seed=6101, batch=2, rank=2),
            hash_fixture=False,
        )
        fixture_tensors = self.load_file(str(written.fixture_path))
        self.save_file({"runtime.output": fixture_tensors["fixture.expected_output"].clone()}, str(output_path))
        return written.fixture_path, output_path

    def test_validate_runtime_fixture_output_passes_and_cli_writes_report(self):
        from comfy_quants.backends.runtime_fixture_validation import (
            RUNTIME_FIXTURE_OUTPUT_VALIDATION_SCHEMA_VERSION,
            validate_runtime_fixture_output,
        )

        with tempfile.TemporaryDirectory() as tmp:
            fixture_path, output_path = self._write_svd_fixture_and_matching_output(tmp)

            report = validate_runtime_fixture_output(fixture_path, output_path)
            self.assertEqual(report.status, "passed")
            self.assertEqual(report.schema_version, RUNTIME_FIXTURE_OUTPUT_VALIDATION_SCHEMA_VERSION)
            self.assertEqual(report.external_runtime_validation, "single_layer_fixture_output_passed")
            self.assertIs(report.publishable_svdquant_gptq, False)
            self.assertEqual(report.max_abs_error, 0.0)
            self.assertEqual(report.mean_abs_error, 0.0)
            self.assertIn("full Qwen-Image/Edit model load", report.does_not_validate)

            validation_dir = Path(tmp) / "validation"
            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "validate-runtime-fixture-output",
                        "--fixture",
                        str(fixture_path),
                        "--output",
                        str(output_path),
                        "--out",
                        str(validation_dir),
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            cli_report = json.loads(captured.getvalue())
            self.assertEqual(cli_report["status"], "passed")
            self.assertEqual(cli_report["external_runtime_validation"], "single_layer_fixture_output_passed")
            self.assertIs(cli_report["publishable_svdquant_gptq"], False)
            report_path = validation_dir / "runtime_fixture_output_validation_report.json"
            self.assertTrue(report_path.exists())
            on_disk = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["status"], "passed")

    def test_validate_runtime_fixture_output_fails_on_value_mismatch_and_cli_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path, output_path = self._write_svd_fixture_and_matching_output(tmp)
            output_tensors = self.load_file(str(output_path))
            perturbed = output_tensors["runtime.output"].clone()
            perturbed.reshape(-1)[0] += 0.25
            self.save_file({"runtime.output": perturbed}, str(output_path))

            validation_dir = Path(tmp) / "validation_fail"
            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "validate-runtime-fixture-output",
                        "--fixture",
                        str(fixture_path),
                        "--output",
                        str(output_path),
                        "--out",
                        str(validation_dir),
                        "--atol",
                        "1e-6",
                        "--rtol",
                        "1e-6",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 2)
            cli_report = json.loads(captured.getvalue())
            self.assertEqual(cli_report["status"], "failed")
            self.assertEqual(cli_report["external_runtime_validation"], "single_layer_fixture_output_failed")
            self.assertIs(cli_report["publishable_svdquant_gptq"], False)
            self.assertGreater(cli_report["max_abs_error"], 0.0)
            self.assertTrue(cli_report["errors"])

    def test_validate_runtime_fixture_output_reports_missing_tensor(self):
        from comfy_quants.backends.runtime_fixture_validation import validate_runtime_fixture_output

        with tempfile.TemporaryDirectory() as tmp:
            fixture_path, output_path = self._write_svd_fixture_and_matching_output(tmp)
            self.save_file({"other.output": self.torch.zeros((1, 1), dtype=self.torch.float32)}, str(output_path))

            report = validate_runtime_fixture_output(fixture_path, output_path)
            self.assertEqual(report.status, "failed")
            self.assertIs(report.publishable_svdquant_gptq, False)
            self.assertTrue(any("actual tensor is missing" in error for error in report.errors))

    def test_validate_runtime_fixture_output_reports_shape_mismatch(self):
        from comfy_quants.backends.runtime_fixture_validation import validate_runtime_fixture_output

        with tempfile.TemporaryDirectory() as tmp:
            fixture_path, output_path = self._write_svd_fixture_and_matching_output(tmp)
            self.save_file({"runtime.output": self.torch.zeros((1, 1), dtype=self.torch.float32)}, str(output_path))

            report = validate_runtime_fixture_output(fixture_path, output_path)
            self.assertEqual(report.status, "failed")
            self.assertEqual(report.actual_shape, [1, 1])
            self.assertTrue(any("shape mismatch" in error for error in report.errors))

    def test_validate_awq_runtime_fixture_output_uses_same_single_layer_gate(self):
        from comfy_quants.algorithms.awq_w4a16.runtime_fixture import (
            AwqW4A16RuntimeFixtureConfig,
            write_awq_w4a16_runtime_fixture,
        )
        from comfy_quants.backends.runtime_fixture_validation import validate_runtime_fixture_output

        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "awq_fixture"
            output_path = Path(tmp) / "awq_runtime_output.safetensors"
            written = write_awq_w4a16_runtime_fixture(
                fixture_dir,
                config=AwqW4A16RuntimeFixtureConfig(seed=6201, n=8, batch=2, scale_dtype="float32"),
                hash_fixture=False,
            )
            fixture_tensors = self.load_file(str(written.fixture_path))
            self.save_file({"runtime.output": fixture_tensors["fixture.expected_output"].clone()}, str(output_path))

            report = validate_runtime_fixture_output(written.fixture_path, output_path)
            self.assertEqual(report.status, "passed")
            self.assertEqual(report.external_runtime_validation, "single_layer_fixture_output_passed")
            self.assertIs(report.publishable_svdquant_gptq, False)
            self.assertIn("mixed SVDQuant W4A4 plus AWQ W4A16 dispatch", report.does_not_validate)


if __name__ == "__main__":
    unittest.main()
