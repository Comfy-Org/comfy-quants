import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.cli.main import main
from comfy_quants.formats.int4_common import decode_quant_config_tensor


def _torch_safetensors_deps():
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError:
        return None
    return torch, load_file, save_file


class TestInt4FullPipeline(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.load_file, self.save_file = deps

    def test_weight_quantizer_builds_natural_svdquant_tensors(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.weight_quant import quantize_linear_weight_to_natural_svdquant
        from comfy_quants.formats.int4_common import unpack_signed_int4_pairs

        weight = torch.linspace(-2.0, 2.0, 128 * 128, dtype=torch.float32).view(128, 128).to(torch.float16)
        natural = quantize_linear_weight_to_natural_svdquant(weight, rank=8, scale_dtype="float16")

        self.assertEqual(tuple(natural.weight.shape), (128, 64))
        self.assertEqual(tuple(unpack_signed_int4_pairs(natural.weight).shape), (128, 128))
        self.assertEqual(tuple(natural.weight_scale.shape), (2, 128))
        self.assertEqual(tuple(natural.smooth_factor.shape), (128,))
        self.assertEqual(tuple(natural.proj_down.shape), (128, 8))
        self.assertEqual(tuple(natural.proj_up.shape), (128, 8))
        self.assertEqual(natural.weight.dtype, torch.int8)
        self.assertEqual(natural.weight_scale.dtype, torch.float16)

    def test_cli_quantize_int4_writes_single_tilepacked_checkpoint(self):
        torch = self.torch
        from comfy_quants.formats.int4_common import decode_quant_config_tensor
        from comfy_quants.formats.kitchen_tilepack import unpack_weight_scale, unpack_weight_tile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output_dir = root / "out"
            prefix = "transformer_blocks.0.attn.to_q"
            source_tensors = {
                f"{prefix}.weight": torch.linspace(-1.0, 1.0, 128 * 128, dtype=torch.float32).view(128, 128).to(torch.float16),
                f"{prefix}.bias": torch.arange(128, dtype=torch.float16),
                "transformer_blocks.0.norm.weight": torch.ones((128,), dtype=torch.float16),
            }
            self.save_file(source_tensors, str(source))

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--family",
                        "qwen_image_edit",
                        "--format",
                        "svdquant_w4a4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--rank",
                        "8",
                        "--scale-dtype",
                        "float32",
                        "--device",
                        "cpu",
                        "--json",
                        "--no-progress",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            checkpoint = output_dir / "diffusion_pytorch_model.svdquant_w4a4.safetensors"
            report_path = output_dir / "quantization_report.json"
            self.assertEqual(result["status"], "model_written")
            self.assertEqual(result["pipeline_kind"], "direct_quantize_to_kitchen_tilepack")
            self.assertEqual(result["quantized_layer_count"], 1)
            self.assertEqual(result["runtime_reference_state"], "repo_runtime_like_activation_w4_branch_oracle_runtime_unverified")
            self.assertEqual(result["lowrank_branch_input_basis"], "raw")
            self.assertTrue(result["proj_down_smooth_folded"])
            self.assertTrue(checkpoint.exists())
            self.assertTrue(report_path.exists())

            exported = self.load_file(str(checkpoint))
            self.assertEqual(tuple(exported[f"{prefix}.weight"].shape), (1, 2, 32, 128))
            self.assertEqual(tuple(unpack_weight_tile(exported[f"{prefix}.weight"]).shape), (128, 64))
            self.assertEqual(tuple(unpack_weight_scale(exported[f"{prefix}.weight_scale"]).shape), (2, 128))
            self.assertEqual(exported[f"{prefix}.weight_scale"].dtype, torch.bfloat16)
            self.assertEqual(tuple(exported[f"{prefix}.proj_up"].shape), (1, 8, 128))
            self.assertTrue(torch.equal(exported["transformer_blocks.0.norm.weight"], source_tensors["transformer_blocks.0.norm.weight"]))
            quant_config = decode_quant_config_tensor(exported[f"{prefix}.comfy_quant"])
            self.assertEqual(quant_config["format"], "svdquant_w4a4")
            self.assertEqual(quant_config["layout"], "kitchen_tile_packed_w4a4")
            self.assertEqual(quant_config["lowrank_branch_input_basis"], "raw")
            self.assertIs(quant_config["proj_down_smooth_folded"], True)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["pipeline_kind"], "direct_quantize_to_kitchen_tilepack")
            self.assertEqual(report["storage_layout"], "kitchen_tile_packed_w4a4")
            self.assertEqual(report["algorithm_state"], "weight_only_initialization_no_calibration_no_gptq")
            self.assertFalse(report["publishable_svdquant_gptq"])
            self.assertEqual(report["gptq_state"], "not_implemented")
            self.assertEqual(report["runtime_contract_state"], "static_artifact_contract_only")
            self.assertEqual(report["runtime_reference_state"], "repo_runtime_like_activation_w4_branch_oracle_runtime_unverified")
            self.assertEqual(report["lowrank_branch_input_basis"], "raw")
            self.assertTrue(report["proj_down_smooth_folded"])
            self.assertEqual(report["mixed_quantization_state"], "svdquant_only_awq_modulation_not_implemented")
            self.assertEqual(report["rank"], 8)
            self.assertEqual(report["selected_layers"][0]["source_prefix"], prefix)

    def test_cli_quantize_int4_writes_mixed_svdquant_and_awq_modulation(self):
        torch = self.torch
        from comfy_quants.formats.int4_common import decode_quant_config_tensor
        from comfy_quants.formats.kitchen_tilepack import unpack_weight_tile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output_dir = root / "out"
            svd_prefix = "transformer_blocks.0.attn.to_q"
            awq_prefix = "transformer_blocks.0.img_mod.1"
            source_tensors = {
                f"{svd_prefix}.weight": torch.randn((128, 128), generator=torch.Generator().manual_seed(1), dtype=torch.float32).to(torch.float16),
                f"{awq_prefix}.weight": torch.linspace(-1.5, 2.0, 12 * 128, dtype=torch.float32).view(12, 128).to(torch.float16),
                f"{awq_prefix}.bias": torch.arange(12, dtype=torch.float16),
                "transformer_blocks.0.norm.weight": torch.ones((128,), dtype=torch.float16),
            }
            self.save_file(source_tensors, str(source))

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--rank",
                        "8",
                        "--device",
                        "cpu",
                        "--json",
                        "--no-progress",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["quantized_layer_count"], 1)
            self.assertEqual(result["awq_modulation_layer_count"], 1)
            self.assertEqual(result["mixed_quantization_state"], "experimental_svdquant_w4a4_awq_w4a16_runtime_unverified")

            checkpoint = output_dir / "diffusion_pytorch_model.svdquant_w4a4.safetensors"
            exported = self.load_file(str(checkpoint))
            self.assertEqual(tuple(exported[f"{svd_prefix}.weight"].shape), (1, 2, 32, 128))
            self.assertEqual(tuple(unpack_weight_tile(exported[f"{svd_prefix}.weight"]).shape), (128, 64))
            self.assertIn(exported[f"{svd_prefix}.weight_scale"].dtype, {torch.float16, torch.bfloat16})
            self.assertNotEqual(exported[f"{svd_prefix}.weight_scale"].dtype, torch.float32)
            svd_config = decode_quant_config_tensor(exported[f"{svd_prefix}.comfy_quant"])
            self.assertEqual(svd_config["format"], "svdquant_w4a4")
            self.assertEqual(svd_config["lowrank_branch_input_basis"], "raw")
            self.assertIs(svd_config["proj_down_smooth_folded"], True)
            self.assertEqual(tuple(exported[f"{awq_prefix}.weight"].shape), (12, 64))
            self.assertEqual(tuple(exported[f"{awq_prefix}.weight_scale"].shape), (2, 12))
            self.assertEqual(tuple(exported[f"{awq_prefix}.weight_zero"].shape), (2, 12))
            self.assertEqual(exported[f"{awq_prefix}.weight"].dtype, torch.int8)
            self.assertEqual(tuple(exported[f"{awq_prefix}.bias"].shape), (12,))
            awq_config = decode_quant_config_tensor(exported[f"{awq_prefix}.comfy_quant"])
            self.assertEqual(awq_config["format"], "awq_w4a16")
            self.assertEqual(awq_config["group_size"], 64)

            report = json.loads((output_dir / "quantization_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["awq_modulation_layer_count"], 1)
            self.assertEqual(report["awq_modulation_layers"][0]["source_prefix"], awq_prefix)
            self.assertIn(f"{awq_prefix}.weight", report["skipped_tensors"])

    def test_cli_quantize_int4_dry_run_writes_plan(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output_dir = root / "plan"
            self.save_file(
                {"transformer_blocks.0.attn.to_q.weight": torch.zeros((128, 128), dtype=torch.float16)},
                str(source),
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--dry-run",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "dry_run_planned")
            self.assertEqual(result["selected_layer_count"], 1)
            plan = json.loads((output_dir / "quantization_report.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["status"], "dry_run_planned")
            self.assertEqual(plan["algorithm_state"], "weight_only_initialization_no_calibration_no_gptq")
            self.assertFalse(plan["publishable_svdquant_gptq"])
            self.assertEqual(plan["lowrank_branch_input_basis"], "raw")
            self.assertTrue(plan["proj_down_smooth_folded"])
            self.assertEqual(plan["selected_layers"][0]["output_prefix"], "transformer_blocks.0.attn.to_q")

    def test_cli_quantize_int4_calibrated_dry_run_validates_activation_stats(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            output_dir = root / "plan"
            prefix = "transformer_blocks.0.attn.to_q"
            self.save_file({f"{prefix}.weight": torch.zeros((128, 128), dtype=torch.float16)}, str(source))
            stats.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_activation_stats.v1",
                        "layers": {prefix: {"input_amax": [1.0] * 128, "sample_count": 2, "element_count": 32}},
                    }
                ),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--quantization-mode",
                        "calibrated_svdquant",
                        "--activation-stats",
                        str(stats),
                        "--dry-run",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "dry_run_planned")
            self.assertEqual(result["algorithm_state"], "experimental_smooth_rtn_svd_no_gptq")
            self.assertFalse(result["publishable_svdquant_gptq"])
            self.assertEqual(result["gptq_state"], "not_implemented")
            self.assertEqual(result["activation_stats_coverage_state"], "valid")
            plan = json.loads((output_dir / "quantization_report.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["algorithm_state"], "experimental_smooth_rtn_svd_no_gptq")
            self.assertFalse(plan["publishable_svdquant_gptq"])
            coverage = plan["activation_stats_coverage"]
            self.assertEqual(coverage["state"], "valid")
            self.assertEqual(coverage["matched_layer_count"], 1)
            self.assertEqual(coverage["shape_checked_layer_count"], 1)
            self.assertEqual(plan["selected_layers"][0]["shape"], [128, 128])

    def test_cli_quantize_int4_calibrated_dry_run_reports_missing_stats(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            output_dir = root / "plan"
            prefix = "transformer_blocks.0.attn.to_q"
            self.save_file({f"{prefix}.weight": torch.zeros((128, 128), dtype=torch.float16)}, str(source))
            stats.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_activation_stats.v1",
                        "layers": {"transformer_blocks.0.attn.to_k": {"input_amax": [1.0] * 128}},
                    }
                ),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--quantization-mode",
                        "calibrated_svdquant",
                        "--activation-stats",
                        str(stats),
                        "--dry-run",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 2)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "dry_run_validation_failed")
            self.assertEqual(result["activation_stats_coverage_state"], "invalid")
            self.assertEqual(result["activation_stats_missing_layer_count"], 1)
            plan = json.loads((output_dir / "quantization_report.json").read_text(encoding="utf-8"))
            missing = plan["activation_stats_coverage"]["missing_layers"]
            self.assertEqual(missing[0]["output_prefix"], prefix)
            self.assertIn(prefix, missing[0]["candidates"])

    def test_cli_quantize_int4_calibrated_dry_run_reports_stats_shape_mismatch(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            output_dir = root / "plan"
            prefix = "transformer_blocks.0.attn.to_q"
            self.save_file({f"{prefix}.weight": torch.zeros((128, 128), dtype=torch.float16)}, str(source))
            stats.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_activation_stats.v1",
                        "layers": {prefix: {"input_amax": [1.0] * 64}},
                    }
                ),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--quantization-mode",
                        "calibrated_svdquant",
                        "--activation-stats",
                        str(stats),
                        "--dry-run",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 2)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "dry_run_validation_failed")
            self.assertEqual(result["activation_stats_shape_mismatch_count"], 1)
            plan = json.loads((output_dir / "quantization_report.json").read_text(encoding="utf-8"))
            mismatch = plan["activation_stats_coverage"]["shape_mismatches"][0]
            self.assertEqual(mismatch["expected_input_channels"], 128)
            self.assertEqual(mismatch["actual_input_channels"], 64)

    def test_cli_quantize_int4_calibrated_mode_uses_activation_stats(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            output_dir = root / "out"
            prefix = "transformer_blocks.0.attn.to_q"
            generator = torch.Generator().manual_seed(17)
            self.save_file(
                {f"{prefix}.weight": torch.randn((128, 128), generator=generator, dtype=torch.float32).to(torch.float16)},
                str(source),
            )
            stats.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_activation_stats.v1",
                        "layers": {prefix: {"input_amax": torch.linspace(0.5, 3.0, 128).tolist(), "sample_count": 4}},
                    }
                ),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--quantization-mode",
                        "calibrated_svdquant",
                        "--activation-stats",
                        str(stats),
                        "--rank",
                        "4",
                        "--device",
                        "cpu",
                        "--json",
                        "--no-progress",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            checkpoint = output_dir / "diffusion_pytorch_model.svdquant_w4a4.safetensors"
            report_path = output_dir / "quantization_report.json"
            self.assertEqual(result["quantization_mode"], "calibrated_svdquant")
            self.assertEqual(result["algorithm_state"], "experimental_smooth_rtn_svd_no_gptq")
            self.assertFalse(result["publishable_svdquant_gptq"])
            self.assertEqual(result["gptq_state"], "not_implemented")
            self.assertEqual(result["runtime_contract_state"], "static_artifact_contract_only")
            self.assertEqual(result["mixed_quantization_state"], "svdquant_only_awq_modulation_not_implemented")
            self.assertEqual(result["activation_stats_state"], "loaded")
            exported = self.load_file(str(checkpoint))
            self.assertIn(exported[f"{prefix}.weight_scale"].dtype, {torch.float16, torch.bfloat16})
            self.assertNotEqual(exported[f"{prefix}.weight_scale"].dtype, torch.float32)
            self.assertFalse(torch.allclose(exported[f"{prefix}.smooth_factor"], torch.ones_like(exported[f"{prefix}.smooth_factor"])))
            self.assertGreater(float(exported[f"{prefix}.proj_down"].abs().amax()), 0.0)
            self.assertGreater(float(exported[f"{prefix}.proj_up"].abs().amax()), 0.0)
            quant_config = decode_quant_config_tensor(exported[f"{prefix}.comfy_quant"])
            self.assertEqual(quant_config["lowrank_branch_input_basis"], "raw")
            self.assertIs(quant_config["proj_down_smooth_folded"], True)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["algorithm_state"], "experimental_smooth_rtn_svd_no_gptq")
            self.assertFalse(report["publishable_svdquant_gptq"])
            self.assertEqual(report["gptq_state"], "not_implemented")
            self.assertEqual(report["lowrank_branch_input_basis"], "raw")
            self.assertTrue(report["proj_down_smooth_folded"])
            self.assertEqual(report["activation_stats_layer_count"], 1)
            self.assertEqual(report["selected_layers"][0]["activation_stats_key"], prefix)

    def test_cli_quantize_int4_raw_lowrank_branch_basis_folds_proj_down(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.branch_basis import fold_proj_down_for_raw_branch
        from comfy_quants.algorithms.int4_svdquant.weight_quant import quantize_linear_weight_to_calibrated_natural_svdquant

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            output_dir = root / "out"
            prefix = "transformer_blocks.0.attn.to_q"
            generator = torch.Generator().manual_seed(271)
            dense_weight = torch.randn((128, 128), generator=generator, dtype=torch.float32).to(torch.float16)
            activation_amax = torch.linspace(0.4, 2.8, 128)
            self.save_file({f"{prefix}.weight": dense_weight}, str(source))
            stats.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_activation_stats.v1",
                        "layers": {prefix: {"input_amax": activation_amax.tolist(), "sample_count": 4}},
                    }
                ),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--quantization-mode",
                        "calibrated_svdquant",
                        "--activation-stats",
                        str(stats),
                        "--rank",
                        "4",
                        "--lowrank-branch-input-basis",
                        "raw",
                        "--device",
                        "cpu",
                        "--json",
                        "--no-progress",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["lowrank_branch_input_basis"], "raw")
            self.assertTrue(result["proj_down_smooth_folded"])

            exported = self.load_file(str(output_dir / "diffusion_pytorch_model.svdquant_w4a4.safetensors"))
            quant_config = decode_quant_config_tensor(exported[f"{prefix}.comfy_quant"])
            self.assertEqual(quant_config["lowrank_branch_input_basis"], "raw")
            self.assertIs(quant_config["proj_down_smooth_folded"], True)

            natural_post = quantize_linear_weight_to_calibrated_natural_svdquant(
                dense_weight,
                activation_stats=activation_amax,
                rank=4,
                scale_dtype="source",
            )
            expected_raw = fold_proj_down_for_raw_branch(natural_post.proj_down, natural_post.smooth_factor)
            self.assertTrue(torch.allclose(exported[f"{prefix}.smooth_factor"], natural_post.smooth_factor.cpu()))
            self.assertTrue(torch.allclose(exported[f"{prefix}.proj_down"], expected_raw.cpu(), atol=1e-3, rtol=1e-3))

            report = json.loads((output_dir / "quantization_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["lowrank_branch_input_basis"], "raw")
            self.assertTrue(report["proj_down_smooth_folded"])

    def test_cli_quantize_int4_gptq_dry_run_validates_hessian_manifest(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            hessian_dir = root / "hessian"
            tensor_dir = hessian_dir / "gptq_hessians"
            tensor_dir.mkdir(parents=True)
            hessian_tensor = tensor_dir / "to_q.safetensors"
            manifest = hessian_dir / "int4_gptq_hessian_stats.json"
            output_dir = root / "plan"
            prefix = "transformer_blocks.0.attn.to_q"
            self.save_file({f"{prefix}.weight": torch.zeros((128, 128), dtype=torch.float16)}, str(source))
            stats.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_activation_stats.v1",
                        "layers": {prefix: {"input_amax": [1.0] * 128, "sample_count": 2, "element_count": 256}},
                    }
                ),
                encoding="utf-8",
            )
            self.save_file({"hessian": torch.eye(128, dtype=torch.float32)}, str(hessian_tensor))
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_gptq_hessian_stats.v1",
                        "normalization": "two_over_row_count",
                        "hessian_tensor_dir": "gptq_hessians",
                        "layer_count": 1,
                        "sample_ref_count": 1,
                        "row_count": 256,
                        "layers": {
                            prefix: {
                                "layer_name": prefix,
                                "file_path": "gptq_hessians/to_q.safetensors",
                                "tensor_name": "hessian",
                                "channel_count": 128,
                                "sample_count": 1,
                                "row_count": 256,
                                "normalization_count": 256,
                                "channel_dim": -1,
                                "dtype": "float32",
                                "shape": [128, 128],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--quantization-mode",
                        "svdquant_gptq_experimental",
                        "--activation-stats",
                        str(stats),
                        "--gptq-hessian-stats",
                        str(manifest),
                        "--dry-run",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "dry_run_planned")
            self.assertEqual(result["algorithm_state"], "experimental_svdquant_gptq_no_awq_runtime_unverified")
            self.assertFalse(result["publishable_svdquant_gptq"])
            self.assertEqual(result["gptq_state"], "layer_core_integrated")
            self.assertEqual(result["lowrank_branch_input_basis"], "raw")
            self.assertTrue(result["proj_down_smooth_folded"])
            self.assertEqual(result["activation_stats_coverage_state"], "valid")
            self.assertEqual(result["gptq_hessian_coverage_state"], "valid")
            plan = json.loads((output_dir / "quantization_report.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["gptq_hessian_stats_state"], "valid")
            self.assertEqual(plan["gptq_hessian_layer_count"], 1)
            self.assertEqual(plan["gptq_hessian_coverage"]["matched_layer_count"], 1)
            self.assertEqual(plan["selected_layers"][0]["shape"], [128, 128])

    def test_cli_quantize_int4_gptq_dry_run_reports_missing_hessian(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            hessian_dir = root / "hessian"
            tensor_dir = hessian_dir / "gptq_hessians"
            tensor_dir.mkdir(parents=True)
            hessian_tensor = tensor_dir / "to_k.safetensors"
            manifest = hessian_dir / "int4_gptq_hessian_stats.json"
            output_dir = root / "plan"
            prefix = "transformer_blocks.0.attn.to_q"
            other_prefix = "transformer_blocks.0.attn.to_k"
            self.save_file({f"{prefix}.weight": torch.zeros((128, 128), dtype=torch.float16)}, str(source))
            stats.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_activation_stats.v1",
                        "layers": {prefix: {"input_amax": [1.0] * 128, "sample_count": 2, "element_count": 256}},
                    }
                ),
                encoding="utf-8",
            )
            self.save_file({"hessian": torch.eye(128, dtype=torch.float32)}, str(hessian_tensor))
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_gptq_hessian_stats.v1",
                        "normalization": "two_over_row_count",
                        "hessian_tensor_dir": "gptq_hessians",
                        "layer_count": 1,
                        "sample_ref_count": 1,
                        "row_count": 256,
                        "layers": {
                            other_prefix: {
                                "layer_name": other_prefix,
                                "file_path": "gptq_hessians/to_k.safetensors",
                                "tensor_name": "hessian",
                                "channel_count": 128,
                                "sample_count": 1,
                                "row_count": 256,
                                "normalization_count": 256,
                                "channel_dim": -1,
                                "dtype": "float32",
                                "shape": [128, 128],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--quantization-mode",
                        "svdquant_gptq_experimental",
                        "--activation-stats",
                        str(stats),
                        "--gptq-hessian-stats",
                        str(manifest),
                        "--dry-run",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 2)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "dry_run_validation_failed")
            self.assertEqual(result["gptq_hessian_coverage_state"], "invalid")
            self.assertEqual(result["gptq_hessian_missing_layer_count"], 1)
            plan = json.loads((output_dir / "quantization_report.json").read_text(encoding="utf-8"))
            missing = plan["gptq_hessian_coverage"]["missing_layers"][0]
            self.assertEqual(missing["output_prefix"], prefix)
            self.assertIn(prefix, missing["candidates"])

    def test_cli_quantize_int4_gptq_mode_consumes_hessian_manifest(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            hessian_dir = root / "hessian"
            tensor_dir = hessian_dir / "gptq_hessians"
            tensor_dir.mkdir(parents=True)
            hessian_tensor = tensor_dir / "to_q.safetensors"
            manifest = hessian_dir / "int4_gptq_hessian_stats.json"
            output_dir = root / "out"
            prefix = "transformer_blocks.0.attn.to_q"
            generator = torch.Generator().manual_seed(19)
            self.save_file(
                {f"{prefix}.weight": torch.randn((128, 128), generator=generator, dtype=torch.float32).to(torch.float16)},
                str(source),
            )
            stats.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_activation_stats.v1",
                        "layers": {prefix: {"input_amax": torch.linspace(0.5, 2.5, 128).tolist(), "sample_count": 4}},
                    }
                ),
                encoding="utf-8",
            )
            self.save_file({"hessian": torch.eye(128, dtype=torch.float32)}, str(hessian_tensor))
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_gptq_hessian_stats.v1",
                        "normalization": "two_over_row_count",
                        "hessian_tensor_dir": "gptq_hessians",
                        "layer_count": 1,
                        "sample_ref_count": 1,
                        "row_count": 512,
                        "layers": {
                            prefix: {
                                "layer_name": prefix,
                                "file_path": "gptq_hessians/to_q.safetensors",
                                "tensor_name": "hessian",
                                "channel_count": 128,
                                "sample_count": 1,
                                "row_count": 512,
                                "normalization_count": 512,
                                "channel_dim": -1,
                                "dtype": "float32",
                                "shape": [128, 128],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--quantization-mode",
                        "svdquant_gptq_experimental",
                        "--activation-stats",
                        str(stats),
                        "--gptq-hessian-stats",
                        str(manifest),
                        "--gptq-block-size",
                        "32",
                        "--gptq-num-inv-tries",
                        "10",
                        "--rank",
                        "4",
                        "--device",
                        "cpu",
                        "--json",
                        "--no-progress",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            checkpoint = output_dir / "diffusion_pytorch_model.svdquant_w4a4.safetensors"
            report_path = output_dir / "quantization_report.json"
            self.assertEqual(result["quantization_mode"], "svdquant_gptq_experimental")
            self.assertEqual(result["algorithm_state"], "experimental_svdquant_gptq_no_awq_runtime_unverified")
            self.assertFalse(result["publishable_svdquant_gptq"])
            self.assertEqual(result["gptq_state"], "layer_core_integrated")
            self.assertEqual(result["runtime_contract_state"], "static_artifact_contract_only")
            self.assertEqual(result["mixed_quantization_state"], "svdquant_only_awq_modulation_not_implemented")
            self.assertEqual(result["activation_stats_state"], "loaded")
            self.assertEqual(result["gptq_hessian_stats_state"], "loaded")
            self.assertEqual(result["gptq_hessian_layer_count"], 1)
            self.assertTrue(checkpoint.exists())
            exported = self.load_file(str(checkpoint))
            self.assertFalse(torch.allclose(exported[f"{prefix}.smooth_factor"], torch.ones_like(exported[f"{prefix}.smooth_factor"])))
            self.assertEqual(tuple(exported[f"{prefix}.weight"].shape), (1, 2, 32, 128))
            self.assertIn(exported[f"{prefix}.weight_scale"].dtype, {torch.float16, torch.bfloat16})
            self.assertNotEqual(exported[f"{prefix}.weight_scale"].dtype, torch.float32)
            quant_config = decode_quant_config_tensor(exported[f"{prefix}.comfy_quant"])
            self.assertEqual(quant_config["lowrank_branch_input_basis"], "raw")
            self.assertIs(quant_config["proj_down_smooth_folded"], True)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["gptq_hessian_coverage"]["state"], "valid")
            self.assertEqual(report["gptq_config"]["block_size"], 32)
            self.assertEqual(report["runtime_reference_state"], "repo_runtime_like_activation_w4_branch_oracle_runtime_unverified")
            self.assertEqual(report["lowrank_branch_input_basis"], "raw")
            self.assertTrue(report["proj_down_smooth_folded"])
            self.assertEqual(report["selected_layers"][0]["activation_stats_key"], prefix)
            self.assertEqual(report["selected_layers"][0]["gptq_hessian_key"], prefix)

    def test_cli_quantize_int4_gptq_output_error_dry_run_validates_activation_samples(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            hessian_dir = root / "hessian"
            tensor_dir = hessian_dir / "gptq_hessians"
            tensor_dir.mkdir(parents=True)
            hessian_tensor = tensor_dir / "to_q.safetensors"
            manifest = hessian_dir / "int4_gptq_hessian_stats.json"
            sample_tensor = root / "activation_sample.safetensors"
            samples = root / "activation_samples.json"
            output_dir = root / "plan"
            prefix = "transformer_blocks.0.attn.to_q"
            self.save_file({f"{prefix}.weight": torch.zeros((128, 128), dtype=torch.float16)}, str(source))
            stats.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_activation_stats.v1",
                        "layers": {prefix: {"input_amax": [1.0] * 128, "sample_count": 2, "element_count": 256}},
                    }
                ),
                encoding="utf-8",
            )
            self.save_file({"hessian": torch.eye(128, dtype=torch.float32)}, str(hessian_tensor))
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_gptq_hessian_stats.v1",
                        "normalization": "two_over_row_count",
                        "hessian_tensor_dir": "gptq_hessians",
                        "layer_count": 1,
                        "sample_ref_count": 1,
                        "row_count": 256,
                        "layers": {
                            prefix: {
                                "layer_name": prefix,
                                "file_path": "gptq_hessians/to_q.safetensors",
                                "tensor_name": "hessian",
                                "channel_count": 128,
                                "sample_count": 1,
                                "row_count": 256,
                                "normalization_count": 256,
                                "channel_dim": -1,
                                "dtype": "float32",
                                "shape": [128, 128],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.save_file({"activation": torch.randn((3, 128), generator=torch.Generator().manual_seed(31), dtype=torch.float32)}, str(sample_tensor))
            samples.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "layer_name": prefix,
                                "file_path": sample_tensor.name,
                                "tensor_name": "activation",
                                "channel_dim": -1,
                                "sample_id": "sample-0",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--quantization-mode",
                        "svdquant_gptq_experimental",
                        "--activation-stats",
                        str(stats),
                        "--gptq-hessian-stats",
                        str(manifest),
                        "--activation-samples",
                        str(samples),
                        "--lowrank-calibration",
                        "output_error",
                        "--dry-run",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "dry_run_planned")
            self.assertEqual(result["lowrank_calibration"], "output_error")
            self.assertEqual(result["activation_samples_coverage_state"], "valid")
            self.assertEqual(result["activation_sample_ref_count"], 1)
            self.assertFalse(result["publishable_svdquant_gptq"])
            plan = json.loads((output_dir / "quantization_report.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["activation_samples_state"], "valid")
            self.assertEqual(plan["activation_samples_coverage"]["matched_layer_count"], 1)
            self.assertEqual(plan["activation_samples_coverage"]["matched_layers"][0]["sample_ref_count"], 1)
            self.assertEqual(plan["lowrank_calibration"], "output_error")

    def test_cli_quantize_int4_gptq_output_error_mode_consumes_activation_samples(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            hessian_dir = root / "hessian"
            tensor_dir = hessian_dir / "gptq_hessians"
            tensor_dir.mkdir(parents=True)
            hessian_tensor = tensor_dir / "to_q.safetensors"
            manifest = hessian_dir / "int4_gptq_hessian_stats.json"
            sample_tensor = root / "activation_sample.safetensors"
            samples = root / "activation_samples.json"
            output_dir = root / "out"
            prefix = "transformer_blocks.0.attn.to_q"
            generator = torch.Generator().manual_seed(37)
            self.save_file(
                {f"{prefix}.weight": torch.randn((128, 128), generator=generator, dtype=torch.float32).to(torch.float16)},
                str(source),
            )
            stats.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_activation_stats.v1",
                        "layers": {prefix: {"input_amax": torch.linspace(0.5, 2.5, 128).tolist(), "sample_count": 4}},
                    }
                ),
                encoding="utf-8",
            )
            self.save_file({"hessian": torch.eye(128, dtype=torch.float32)}, str(hessian_tensor))
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_gptq_hessian_stats.v1",
                        "normalization": "two_over_row_count",
                        "hessian_tensor_dir": "gptq_hessians",
                        "layer_count": 1,
                        "sample_ref_count": 1,
                        "row_count": 512,
                        "layers": {
                            prefix: {
                                "layer_name": prefix,
                                "file_path": "gptq_hessians/to_q.safetensors",
                                "tensor_name": "hessian",
                                "channel_count": 128,
                                "sample_count": 1,
                                "row_count": 512,
                                "normalization_count": 512,
                                "channel_dim": -1,
                                "dtype": "float32",
                                "shape": [128, 128],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.save_file({"activation": torch.randn((5, 128), generator=torch.Generator().manual_seed(41), dtype=torch.float32)}, str(sample_tensor))
            samples.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "layer_name": prefix,
                                "file_path": sample_tensor.name,
                                "tensor_name": "activation",
                                "channel_dim": -1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--quantization-mode",
                        "svdquant_gptq_experimental",
                        "--activation-stats",
                        str(stats),
                        "--gptq-hessian-stats",
                        str(manifest),
                        "--activation-samples",
                        str(samples),
                        "--lowrank-calibration",
                        "output_error",
                        "--gptq-block-size",
                        "32",
                        "--gptq-num-inv-tries",
                        "10",
                        "--rank",
                        "4",
                        "--device",
                        "cpu",
                        "--json",
                        "--no-progress",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["lowrank_calibration"], "output_error")
            self.assertEqual(result["activation_samples_state"], "loaded")
            self.assertEqual(result["activation_sample_ref_count"], 1)
            self.assertFalse(result["publishable_svdquant_gptq"])
            checkpoint = output_dir / "diffusion_pytorch_model.svdquant_w4a4.safetensors"
            exported = self.load_file(str(checkpoint))
            self.assertEqual(tuple(exported[f"{prefix}.weight"].shape), (1, 2, 32, 128))
            self.assertIn(exported[f"{prefix}.weight_scale"].dtype, {torch.float16, torch.bfloat16})
            self.assertNotEqual(exported[f"{prefix}.weight_scale"].dtype, torch.float32)
            self.assertGreater(float(exported[f"{prefix}.proj_down"].abs().amax()), 0.0)
            self.assertGreater(float(exported[f"{prefix}.proj_up"].abs().amax()), 0.0)
            report = json.loads((output_dir / "quantization_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["lowrank_calibration"], "output_error")
            self.assertEqual(report["activation_samples_coverage"]["state"], "valid")
            self.assertEqual(report["selected_layers"][0]["activation_samples_key"], prefix)
            self.assertEqual(report["selected_layers"][0]["activation_sample_count"], 1)
            self.assertEqual(report["selected_layers"][0]["activation_sample_channel_dim"], -1)

    def test_cli_quantize_int4_output_error_requires_activation_samples(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            hessian_dir = root / "hessian"
            tensor_dir = hessian_dir / "gptq_hessians"
            tensor_dir.mkdir(parents=True)
            hessian_tensor = tensor_dir / "to_q.safetensors"
            manifest = hessian_dir / "int4_gptq_hessian_stats.json"
            output_dir = root / "plan"
            prefix = "transformer_blocks.0.attn.to_q"
            self.save_file({f"{prefix}.weight": torch.zeros((128, 128), dtype=torch.float16)}, str(source))
            stats.write_text(
                json.dumps({"schema_version": "int4_activation_stats.v1", "layers": {prefix: {"input_amax": [1.0] * 128}}}),
                encoding="utf-8",
            )
            self.save_file({"hessian": torch.eye(128, dtype=torch.float32)}, str(hessian_tensor))
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_gptq_hessian_stats.v1",
                        "hessian_tensor_dir": "gptq_hessians",
                        "layers": {
                            prefix: {
                                "layer_name": prefix,
                                "file_path": "gptq_hessians/to_q.safetensors",
                                "tensor_name": "hessian",
                                "channel_count": 128,
                                "shape": [128, 128],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            rc = main(
                [
                    "quantize-int4",
                    "--source",
                    str(source),
                    "--out",
                    str(output_dir),
                    "--quantization-mode",
                    "svdquant_gptq_experimental",
                    "--activation-stats",
                    str(stats),
                    "--gptq-hessian-stats",
                    str(manifest),
                    "--lowrank-calibration",
                    "output_error",
                    "--dry-run",
                    "--json",
                ]
            )

            self.assertEqual(rc, 2)

    def test_cli_quantize_int4_gptq_output_error_dry_run_reports_sample_shape_mismatch(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            stats = root / "activation_stats.json"
            hessian_dir = root / "hessian"
            tensor_dir = hessian_dir / "gptq_hessians"
            tensor_dir.mkdir(parents=True)
            hessian_tensor = tensor_dir / "to_q.safetensors"
            manifest = hessian_dir / "int4_gptq_hessian_stats.json"
            sample_tensor = root / "activation_sample.safetensors"
            samples = root / "activation_samples.json"
            output_dir = root / "plan"
            prefix = "transformer_blocks.0.attn.to_q"
            self.save_file({f"{prefix}.weight": torch.zeros((128, 128), dtype=torch.float16)}, str(source))
            stats.write_text(
                json.dumps({"schema_version": "int4_activation_stats.v1", "layers": {prefix: {"input_amax": [1.0] * 128}}}),
                encoding="utf-8",
            )
            self.save_file({"hessian": torch.eye(128, dtype=torch.float32)}, str(hessian_tensor))
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "int4_gptq_hessian_stats.v1",
                        "hessian_tensor_dir": "gptq_hessians",
                        "layers": {
                            prefix: {
                                "layer_name": prefix,
                                "file_path": "gptq_hessians/to_q.safetensors",
                                "tensor_name": "hessian",
                                "channel_count": 128,
                                "sample_count": 1,
                                "row_count": 512,
                                "normalization_count": 512,
                                "shape": [128, 128],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.save_file({"activation": torch.randn((3, 64), generator=torch.Generator().manual_seed(43), dtype=torch.float32)}, str(sample_tensor))
            samples.write_text(
                json.dumps({"samples": [{"layer_name": prefix, "file_path": sample_tensor.name, "tensor_name": "activation", "channel_dim": -1}]}),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "quantize-int4",
                        "--source",
                        str(source),
                        "--out",
                        str(output_dir),
                        "--quantization-mode",
                        "svdquant_gptq_experimental",
                        "--activation-stats",
                        str(stats),
                        "--gptq-hessian-stats",
                        str(manifest),
                        "--activation-samples",
                        str(samples),
                        "--lowrank-calibration",
                        "output_error",
                        "--dry-run",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 2)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "dry_run_validation_failed")
            self.assertEqual(result["activation_samples_coverage_state"], "invalid")
            self.assertEqual(result["activation_samples_shape_mismatch_count"], 1)
            plan = json.loads((output_dir / "quantization_report.json").read_text(encoding="utf-8"))
            mismatch = plan["activation_samples_coverage"]["shape_mismatches"][0]
            self.assertEqual(mismatch["expected_input_channels"], 128)
            self.assertEqual(mismatch["actual_input_channels"], 64)


if __name__ == "__main__":
    unittest.main()
