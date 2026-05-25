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
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError:
        return None
    return torch, safe_open, load_file


class TestAwqW4A16RuntimeFixture(unittest.TestCase):
    def setUp(self):
        deps = _deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.safe_open, self.load_file = deps

    def test_writes_kitchen_native_awq_fixture(self):
        torch = self.torch
        from comfy_quants.algorithms.awq_w4a16.reference import AWQ_W4A16_REFERENCE_STATE, reference_awq_w4a16_linear
        from comfy_quants.algorithms.awq_w4a16.runtime_fixture import (
            AWQ_W4A16_RUNTIME_FIXTURE_SCHEMA_VERSION,
            AwqW4A16RuntimeFixtureConfig,
            write_awq_w4a16_runtime_fixture,
        )
        from comfy_quants.formats.int4_common import decode_quant_config_tensor

        with tempfile.TemporaryDirectory() as tmp:
            written = write_awq_w4a16_runtime_fixture(
                tmp,
                config=AwqW4A16RuntimeFixtureConfig(seed=5101, n=14, batch=2, scale_dtype="float16"),
                hash_fixture=False,
            )

            self.assertTrue(written.fixture_path.exists())
            self.assertTrue(written.report_path.exists())
            self.assertEqual(written.report["status"], "fixture_written")
            self.assertEqual(written.report["schema_version"], AWQ_W4A16_RUNTIME_FIXTURE_SCHEMA_VERSION)
            self.assertEqual(written.report["runtime_reference_state"], AWQ_W4A16_REFERENCE_STATE)
            self.assertIs(written.report["publishable_svdquant_gptq"], False)
            self.assertEqual(written.report["external_runtime_validation"], "not_run")
            self.assertEqual(written.report["format"], "awq_w4a16")
            self.assertEqual(written.report["storage_layout"], "kitchen_native_awq_w4a16")
            self.assertEqual(written.report["local_self_check"]["status"], "passed")
            self.assertIsNone(written.report["fixture_hash_sha256"])
            harness = written.report["external_harness_contract"]
            self.assertEqual(harness["scope"], "single_layer_awq_w4a16_linear_forward")
            self.assertEqual(harness["validation_command"], "validate-runtime-fixture-output")
            self.assertEqual(harness["forward_input_tensor"], "fixture.input")
            self.assertEqual(harness["expected_output_tensor"], "fixture.expected_output")
            self.assertEqual(harness["external_output_tensor"], "runtime.output")
            self.assertEqual(harness["group_size"], 64)
            self.assertIn("fixture_layer.weight", harness["required_layer_tensors"])
            self.assertIn("fixture_layer.weight_zero", harness["required_layer_tensors"])
            self.assertIn("fixture_layer.bias", harness["optional_layer_tensors"])

            report_on_disk = json.loads(written.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report_on_disk["status"], "fixture_written")
            self.assertEqual(report_on_disk["external_harness_contract"]["external_output_tensor"], "runtime.output")

            tensors = self.load_file(str(written.fixture_path))
            self.assertEqual(tuple(tensors["fixture_layer.weight"].shape), (14, 64))
            self.assertEqual(tuple(tensors["fixture_layer.weight_scale"].shape), (2, 14))
            self.assertEqual(tuple(tensors["fixture_layer.weight_zero"].shape), (2, 14))
            self.assertEqual(tuple(tensors["fixture_layer.bias"].shape), (14,))
            self.assertEqual(tuple(tensors["fixture.input"].shape), (2, 128))
            self.assertEqual(tuple(tensors["fixture.expected_output"].shape), (2, 14))
            self.assertEqual(tensors["fixture_layer.weight"].dtype, torch.int8)
            self.assertEqual(tensors["fixture_layer.weight_scale"].dtype, torch.float16)
            self.assertEqual(tensors["fixture_layer.weight_zero"].dtype, torch.float16)

            quant_config = decode_quant_config_tensor(tensors["fixture_layer.comfy_quant"])
            self.assertEqual(quant_config, {"format": "awq_w4a16", "group_size": 64})

            recomputed = reference_awq_w4a16_linear(
                tensors["fixture.input"],
                tensors["fixture_layer.weight"],
                tensors["fixture_layer.weight_scale"],
                tensors["fixture_layer.weight_zero"],
                bias=tensors["fixture_layer.bias"],
            )
            self.assertTrue(torch.allclose(recomputed, tensors["fixture.expected_output"], atol=1e-5, rtol=1e-5))
            self.assertTrue(
                torch.allclose(
                    tensors["fixture.expected_output_from_dequantized_weight"],
                    tensors["fixture.expected_output"],
                    atol=1e-5,
                    rtol=1e-5,
                )
            )

            with self.safe_open(str(written.fixture_path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            self.assertEqual(metadata["artifact_contract"], AWQ_W4A16_RUNTIME_FIXTURE_SCHEMA_VERSION)
            self.assertEqual(metadata["artifact_state"], "local_awq_runtime_fixture_external_runtime_unverified")
            self.assertEqual(metadata["publishable_svdquant_gptq"], "false")

    def test_cli_make_awq_runtime_fixture_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "make-awq-runtime-fixture",
                        "--out",
                        tmp,
                        "--seed",
                        "5201",
                        "--n",
                        "10",
                        "--batch",
                        "1",
                        "--scale-dtype",
                        "float32",
                        "--json",
                        "--no-hash",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "fixture_written")
            self.assertEqual(result["format"], "awq_w4a16")
            self.assertEqual(result["scale_dtype"], "float32")
            self.assertIs(result["publishable_svdquant_gptq"], False)
            self.assertEqual(result["external_runtime_validation"], "not_run")
            self.assertEqual(result["local_self_check"]["status"], "passed")
            self.assertTrue(Path(result["fixture_path"]).exists())
            self.assertTrue(Path(result["report_path"]).exists())


if __name__ == "__main__":
    unittest.main()
