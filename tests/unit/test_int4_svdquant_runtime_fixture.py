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


class TestInt4SvdquantRuntimeFixture(unittest.TestCase):
    def setUp(self):
        deps = _deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.safe_open, self.load_file = deps

    def test_writes_post_smoothing_kitchen_tilepacked_fixture(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_fixture import (
            SVDQUANT_W4A4_RUNTIME_FIXTURE_SCHEMA_VERSION,
            SVDQUANT_W4A4_RUNTIME_REFERENCE_STATE,
            SVDQuantW4A4RuntimeFixtureConfig,
            write_svdquant_w4a4_runtime_fixture,
        )
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import reference_svdquant_w4a4_linear_runtime
        from comfy_quants.formats.int4_common import decode_quant_config_tensor

        with tempfile.TemporaryDirectory() as tmp:
            written = write_svdquant_w4a4_runtime_fixture(
                tmp,
                config=SVDQuantW4A4RuntimeFixtureConfig(
                    seed=4101,
                    batch=2,
                    rank=3,
                    lowrank_branch_input_basis="post_smoothing",
                ),
                hash_fixture=False,
            )

            self.assertTrue(written.fixture_path.exists())
            self.assertTrue(written.report_path.exists())
            self.assertEqual(written.report["status"], "fixture_written")
            self.assertEqual(written.report["schema_version"], SVDQUANT_W4A4_RUNTIME_FIXTURE_SCHEMA_VERSION)
            self.assertEqual(written.report["runtime_reference_state"], SVDQUANT_W4A4_RUNTIME_REFERENCE_STATE)
            self.assertIs(written.report["publishable_svdquant_gptq"], False)
            self.assertEqual(written.report["external_runtime_validation"], "not_run")
            self.assertEqual(written.report["lowrank_branch_input_basis"], "post_smoothing")
            self.assertIs(written.report["proj_down_smooth_folded"], False)
            self.assertEqual(written.report["local_self_check"]["status"], "passed")
            self.assertIsNone(written.report["fixture_hash_sha256"])
            harness = written.report["external_harness_contract"]
            self.assertEqual(harness["scope"], "single_layer_svdquant_w4a4_linear_forward")
            self.assertEqual(harness["validation_command"], "validate-runtime-fixture-output")
            self.assertEqual(harness["forward_input_tensor"], "fixture.input")
            self.assertEqual(harness["expected_output_tensor"], "fixture.expected_output")
            self.assertEqual(harness["external_output_tensor"], "runtime.output")
            self.assertEqual(harness["lowrank_branch_input_basis"], "post_smoothing")
            self.assertIs(harness["proj_down_smooth_folded"], False)
            self.assertIn("fixture_layer.weight", harness["required_layer_tensors"])
            self.assertIn("fixture_layer.bias", harness["optional_layer_tensors"])

            report_on_disk = json.loads(written.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report_on_disk["status"], "fixture_written")
            self.assertEqual(report_on_disk["external_harness_contract"]["external_output_tensor"], "runtime.output")

            tensors = self.load_file(str(written.fixture_path))
            self.assertEqual(tuple(tensors["fixture_layer.weight"].shape), (1, 2, 32, 128))
            self.assertEqual(tuple(tensors["fixture_layer.weight_scale"].shape), (1, 2, 128))
            self.assertEqual(tuple(tensors["fixture_layer.proj_up"].shape), (1, 3, 128))
            self.assertEqual(tuple(tensors["fixture.input"].shape), (2, 128))
            self.assertEqual(tuple(tensors["fixture.expected_output"].shape), (2, 128))
            self.assertEqual(tensors["fixture_layer.weight"].dtype, torch.int8)

            quant_config = decode_quant_config_tensor(tensors["fixture_layer.comfy_quant"])
            self.assertEqual(quant_config["format"], "svdquant_w4a4")
            self.assertEqual(quant_config["layout"], "kitchen_tile_packed_w4a4")
            self.assertEqual(quant_config["lowrank_branch_input_basis"], "post_smoothing")
            self.assertIs(quant_config["proj_down_smooth_folded"], False)

            recomputed = reference_svdquant_w4a4_linear_runtime(
                tensors["fixture.input"],
                tensors["fixture_layer.weight"],
                tensors["fixture_layer.weight_scale"],
                tensors["fixture_layer.smooth_factor"],
                tensors["fixture_layer.proj_down"],
                tensors["fixture_layer.proj_up"],
                bias=tensors["fixture_layer.bias"],
                branch_input_basis="post_smoothing",
            )
            self.assertTrue(torch.allclose(recomputed, tensors["fixture.expected_output"], atol=1e-5, rtol=1e-5))

            with self.safe_open(str(written.fixture_path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            self.assertEqual(metadata["artifact_contract"], SVDQUANT_W4A4_RUNTIME_FIXTURE_SCHEMA_VERSION)
            self.assertEqual(metadata["artifact_state"], "local_runtime_fixture_external_runtime_unverified")
            self.assertEqual(metadata["publishable_svdquant_gptq"], "false")

    def test_raw_basis_fixture_stores_smooth_folded_proj_down(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.branch_basis import fold_proj_down_for_raw_branch
        from comfy_quants.algorithms.int4_svdquant.runtime_fixture import (
            SVDQuantW4A4RuntimeFixtureConfig,
            build_svdquant_w4a4_runtime_fixture,
        )
        from comfy_quants.formats.int4_common import decode_quant_config_tensor

        fixture = build_svdquant_w4a4_runtime_fixture(
            SVDQuantW4A4RuntimeFixtureConfig(
                seed=4201,
                batch=2,
                rank=2,
                lowrank_branch_input_basis="raw",
            )
        )

        self.assertEqual(fixture.report["status"], "fixture_built")
        self.assertEqual(fixture.report["lowrank_branch_input_basis"], "raw")
        self.assertIs(fixture.report["proj_down_smooth_folded"], True)
        self.assertEqual(fixture.report["external_harness_contract"]["lowrank_branch_input_basis"], "raw")
        self.assertIs(fixture.report["external_harness_contract"]["proj_down_smooth_folded"], True)
        self.assertEqual(fixture.report["local_self_check"]["status"], "passed")
        self.assertLessEqual(fixture.report["local_self_check"]["branch_basis_equivalence_max_abs_error"], 1e-5)

        expected_raw = fold_proj_down_for_raw_branch(
            fixture.tensors["fixture.proj_down_post_smoothing_reference"],
            fixture.tensors["fixture_layer.smooth_factor"],
        )
        self.assertTrue(torch.allclose(fixture.tensors["fixture_layer.proj_down"], expected_raw, atol=1e-6, rtol=1e-6))
        self.assertTrue(
            torch.allclose(
                fixture.tensors["fixture.expected_output"],
                fixture.tensors["fixture.expected_output_post_smoothing_basis"],
                atol=1e-5,
                rtol=1e-5,
            )
        )
        quant_config = decode_quant_config_tensor(fixture.tensors["fixture_layer.comfy_quant"])
        self.assertEqual(quant_config["lowrank_branch_input_basis"], "raw")
        self.assertIs(quant_config["proj_down_smooth_folded"], True)

    def test_unsigned_fixture_records_shifted_main_path_and_raw_lowrank_input(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_fixture import (
            SVDQuantW4A4RuntimeFixtureConfig,
            build_svdquant_w4a4_runtime_fixture,
        )
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import (
            GELU_UNSIGNED_SHIFT,
            quantize_activation_w4_unsigned,
        )

        fixture = build_svdquant_w4a4_runtime_fixture(
            SVDQuantW4A4RuntimeFixtureConfig(
                seed=4251,
                batch=2,
                rank=2,
                activation_signedness="unsigned",
                lowrank_branch_input_basis="raw",
            )
        )

        inputs = fixture.tensors["fixture.input"]
        smooth = fixture.tensors["fixture_layer.smooth_factor"].reshape(1, -1)
        expected_main_input = inputs + GELU_UNSIGNED_SHIFT
        expected_main_post = expected_main_input / smooth
        expected_activation = quantize_activation_w4_unsigned(expected_main_post, group_size=64)

        self.assertEqual(fixture.report["activation_signedness"], "unsigned")
        self.assertIs(fixture.report["act_unsigned"], True)
        self.assertEqual(fixture.report["external_harness_contract"]["main_input_tensor"], "fixture.main_input")
        self.assertEqual(
            fixture.report["external_harness_contract"]["main_post_smoothing_input_tensor"],
            "fixture.main_post_smoothing_input",
        )
        self.assertEqual(fixture.report["external_harness_contract"]["lowrank_input_tensor"], "fixture.lowrank_input")
        self.assertEqual(fixture.report["external_harness_contract"]["unsigned_activation_shift"], GELU_UNSIGNED_SHIFT)
        self.assertTrue(torch.allclose(fixture.tensors["fixture.main_input"], expected_main_input))
        self.assertTrue(torch.allclose(fixture.tensors["fixture.main_post_smoothing_input"], expected_main_post))
        self.assertTrue(torch.allclose(fixture.tensors["fixture.post_smoothing_input"], expected_main_post))
        self.assertTrue(torch.equal(fixture.tensors["fixture.lowrank_input"], inputs))
        self.assertTrue(torch.equal(fixture.tensors["fixture.activation_q_values"], expected_activation.q_values))
        self.assertTrue(torch.allclose(fixture.tensors["fixture.activation_scale"], expected_activation.scale))

    def test_cli_make_int4_runtime_fixture_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "make-int4-runtime-fixture",
                        "--out",
                        tmp,
                        "--seed",
                        "4301",
                        "--batch",
                        "1",
                        "--rank",
                        "2",
                        "--activation-signedness",
                        "unsigned",
                        "--lowrank-branch-input-basis",
                        "raw",
                        "--json",
                        "--no-hash",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "fixture_written")
            self.assertEqual(result["activation_signedness"], "unsigned")
            self.assertIs(result["act_unsigned"], True)
            self.assertEqual(result["lowrank_branch_input_basis"], "raw")
            self.assertIs(result["proj_down_smooth_folded"], True)
            self.assertIs(result["publishable_svdquant_gptq"], False)
            self.assertEqual(result["local_self_check"]["status"], "passed")
            self.assertTrue(Path(result["fixture_path"]).exists())
            self.assertTrue(Path(result["report_path"]).exists())


if __name__ == "__main__":
    unittest.main()
