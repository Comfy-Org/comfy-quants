import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.cli.main import main
from comfy_quants.core.errors import PayloadWriteError


def _torch_safetensors_deps():
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError:
        return None
    return torch, load_file


def _dense_int4(torch, *, n: int, k: int, offset: int = 0):
    return (torch.arange(n * k, dtype=torch.int16).add(offset).remainder(15) - 7).view(n, k).to(torch.int8)


def _add_ptq_layer(torch, *, model, scales, smooth, branch, prefix: str, n: int = 128, k: int = 128, rank: int = 8, dtype=None, offset: int = 0):
    dtype = dtype or torch.float16
    groups = k // 64
    dense_q = _dense_int4(torch, n=n, k=k, offset=offset)
    scale = torch.ones((n, 1, groups, 1), dtype=dtype)
    weight = dense_q.to(torch.float32).view(n, groups, 64).mul(scale.view(n, groups, 1).to(torch.float32)).view(n, k).to(dtype)
    model[f"{prefix}.weight"] = weight
    model[f"{prefix}.bias"] = torch.arange(n, dtype=torch.float32).to(dtype)
    scales[f"{prefix}.weight.scale.0"] = scale
    smooth[prefix] = torch.linspace(1.0, 2.0, k, dtype=torch.float32).to(dtype)
    branch[prefix] = {
        "a.weight": torch.arange(rank * k, dtype=torch.float32).view(rank, k).to(dtype),
        "b.weight": torch.arange(n * rank, dtype=torch.float32).view(n, rank).to(dtype),
    }
    return dense_q, scale


class TestDeepCompressorInt4Import(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.load_file = deps

    def test_builds_natural_svdquant_from_deepcompressor_artifacts(self):
        torch = self.torch
        from comfy_quants.backends.deepcompressor_import import (
            build_qwen_image_edit_svdquant_natural_state_dict,
            load_deepcompressor_ptq_artifacts,
        )
        from comfy_quants.formats.int4_common import decode_quant_config_tensor, pack_signed_int4_pairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = root / "ptq"
            quant.mkdir()
            model, scales, smooth, branch = {}, {}, {}, {}
            prefix = "transformer_blocks.0.attn.to_q"
            expected_dense, _scale = _add_ptq_layer(
                torch,
                model=model,
                scales=scales,
                smooth=smooth,
                branch=branch,
                prefix=prefix,
                dtype=torch.bfloat16,
            )
            model["transformer_blocks.0.attn.norm_q.weight"] = torch.ones((128,), dtype=torch.bfloat16)
            torch.save(model, quant / "model.pt")
            torch.save(scales, quant / "scale.pt")
            torch.save(smooth, quant / "smooth.pt")
            torch.save(branch, quant / "branch.pt")

            artifacts = load_deepcompressor_ptq_artifacts(quant)
            natural, report = build_qwen_image_edit_svdquant_natural_state_dict(artifacts, device="cpu")

            self.assertEqual(report.imported_layer_count, 1)
            self.assertEqual(report.imported_prefixes, [prefix])
            self.assertIn(f"{prefix}.weight", natural)
            self.assertTrue(torch.equal(natural[f"{prefix}.weight"], pack_signed_int4_pairs(expected_dense)))
            self.assertEqual(tuple(natural[f"{prefix}.weight_scale"].shape), (2, 128))
            self.assertEqual(tuple(natural[f"{prefix}.smooth_factor"].shape), (128,))
            self.assertEqual(tuple(natural[f"{prefix}.proj_down"].shape), (128, 8))
            self.assertEqual(tuple(natural[f"{prefix}.proj_up"].shape), (128, 8))
            self.assertTrue(torch.equal(natural["transformer_blocks.0.attn.norm_q.weight"], model["transformer_blocks.0.attn.norm_q.weight"]))
            quant_config = decode_quant_config_tensor(natural[f"{prefix}.comfy_quant"])
            self.assertEqual(quant_config["format"], "svdquant_w4a4")
            self.assertEqual(quant_config["layout"], "kitchen_tile_packed_w4a4")
            self.assertEqual(quant_config["lowrank_branch_input_basis"], "raw")
            self.assertIs(quant_config["proj_down_smooth_folded"], True)
            expected_raw_proj_down = branch[prefix]["a.weight"].transpose(0, 1).to(torch.float32).div(
                smooth[prefix].to(torch.float32).reshape(-1, 1)
            )
            expected_raw_proj_down = expected_raw_proj_down.to(natural[f"{prefix}.proj_down"].dtype).to(torch.float32)
            self.assertTrue(torch.allclose(natural[f"{prefix}.proj_down"].to(torch.float32), expected_raw_proj_down))

    def test_deepcompressor_import_splits_grouped_qkv_lowrank_branch(self):
        torch = self.torch
        from comfy_quants.backends.deepcompressor_import import (
            build_qwen_image_edit_svdquant_natural_state_dict,
            load_deepcompressor_ptq_artifacts,
        )
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = root / "ptq"
            quant.mkdir()
            model, scales, smooth, branch = {}, {}, {}, {}
            prefixes = [
                "transformer_blocks.0.attn.to_q",
                "transformer_blocks.0.attn.to_k",
                "transformer_blocks.0.attn.to_v",
            ]
            n, k, rank = 128, 128, 4
            expected_dense = []
            for index, prefix in enumerate(prefixes):
                dense, _scale = _add_ptq_layer(
                    torch,
                    model=model,
                    scales=scales,
                    smooth=smooth,
                    branch=branch,
                    prefix=prefix,
                    n=n,
                    k=k,
                    rank=rank,
                    dtype=torch.float16,
                    offset=index * 3,
                )
                expected_dense.append(dense)

            anchor = prefixes[0]
            smooth[anchor] = torch.linspace(1.0, 2.0, k, dtype=torch.float16)
            smooth.pop(prefixes[1])
            smooth.pop(prefixes[2])
            branch[anchor] = {
                "a.weight": (torch.arange(rank * k, dtype=torch.float32).view(rank, k) / 1000.0).to(torch.float16),
                "b.weight": (torch.arange(3 * n * rank, dtype=torch.float32).view(3 * n, rank) / 1000.0).to(torch.float16),
            }
            branch.pop(prefixes[1])
            branch.pop(prefixes[2])
            torch.save(model, quant / "model.pt")
            torch.save(scales, quant / "scale.pt")
            torch.save(smooth, quant / "smooth.pt")
            torch.save(branch, quant / "branch.pt")

            artifacts = load_deepcompressor_ptq_artifacts(quant)
            natural, report = build_qwen_image_edit_svdquant_natural_state_dict(artifacts, device="cpu")

            self.assertEqual(report.imported_layer_count, 3)
            self.assertEqual(report.grouped_qkv_branch_count, 1)
            self.assertEqual(report.grouped_qkv_branch_anchors, [anchor])
            self.assertEqual(report.imported_prefixes, prefixes)
            expected_raw_proj_down = branch[anchor]["a.weight"].transpose(0, 1).to(torch.float32).div(
                smooth[anchor].to(torch.float32).reshape(-1, 1)
            )
            expected_raw_proj_down = expected_raw_proj_down.to(natural[f"{anchor}.proj_down"].dtype).to(torch.float32)
            expected_proj_up_chunks = branch[anchor]["b.weight"].split((n, n, n), dim=0)
            for index, prefix in enumerate(prefixes):
                self.assertTrue(torch.equal(natural[f"{prefix}.weight"], pack_signed_int4_pairs(expected_dense[index])))
                self.assertTrue(torch.equal(natural[f"{prefix}.smooth_factor"], smooth[anchor]))
                self.assertTrue(torch.allclose(natural[f"{prefix}.proj_down"].to(torch.float32), expected_raw_proj_down))
                self.assertTrue(
                    torch.allclose(
                        natural[f"{prefix}.proj_up"].to(torch.float32),
                        expected_proj_up_chunks[index].to(natural[f"{prefix}.proj_up"].dtype).to(torch.float32),
                    )
                )

    def test_deepcompressor_export_writes_kitchen_tilepacked_checkpoint(self):
        torch = self.torch
        from comfy_quants.backends.deepcompressor_import import write_qwen_image_edit_deepcompressor_svdquant_kitchen_checkpoint
        from comfy_quants.formats.int4_common import decode_quant_config_tensor, pack_signed_int4_pairs
        from comfy_quants.formats.kitchen_tilepack import unpack_weight_tile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = root / "ptq"
            quant.mkdir()
            model, scales, smooth, branch = {}, {}, {}, {}
            q_prefix = "transformer_blocks.0.attn.to_q"
            down_source_prefix = "transformer_blocks.0.img_mlp.net.2.linear"
            down_output_prefix = "transformer_blocks.0.img_mlp.net.2"
            q_dense, _ = _add_ptq_layer(
                torch, model=model, scales=scales, smooth=smooth, branch=branch, prefix=q_prefix, dtype=torch.float16, offset=0
            )
            down_dense, _ = _add_ptq_layer(
                torch,
                model=model,
                scales=scales,
                smooth=smooth,
                branch=branch,
                prefix=down_source_prefix,
                dtype=torch.float16,
                offset=3,
            )
            torch.save(model, quant / "model.pt")
            torch.save(scales, quant / "scale.pt")
            torch.save(smooth, quant / "smooth.pt")
            torch.save(branch, quant / "branch.pt")

            output = root / "out" / "model.safetensors"
            report = write_qwen_image_edit_deepcompressor_svdquant_kitchen_checkpoint(
                quant_path=quant,
                output_checkpoint=output,
                device="cpu",
                hash_output=True,
            )

            self.assertEqual(report.status, "model_written")
            self.assertEqual(report.source_format, "deepcompressor_ptq_artifacts")
            self.assertEqual(report.repacked_layer_count, 2)
            self.assertEqual(report.source_import["imported_layer_count"], 2)
            self.assertEqual(report.output_hash_state, "written")

            exported = self.load_file(str(output))
            self.assertTrue(torch.equal(unpack_weight_tile(exported[f"{q_prefix}.weight"]), pack_signed_int4_pairs(q_dense)))
            self.assertTrue(torch.equal(unpack_weight_tile(exported[f"{down_output_prefix}.weight"]), pack_signed_int4_pairs(down_dense)))
            self.assertNotIn(f"{down_source_prefix}.weight", exported)
            quant_config = decode_quant_config_tensor(exported[f"{down_output_prefix}.comfy_quant"])
            self.assertEqual(quant_config["format"], "svdquant_w4a4")
            self.assertEqual(quant_config["layout"], "kitchen_tile_packed_w4a4")
            self.assertIs(quant_config["act_unsigned"], True)
            self.assertEqual(quant_config["lowrank_branch_input_basis"], "raw")
            self.assertIs(quant_config["proj_down_smooth_folded"], True)

    def test_deepcompressor_import_applies_qwen_smooth_alias(self):
        torch = self.torch
        from comfy_quants.backends.deepcompressor_import import (
            build_qwen_image_edit_svdquant_natural_state_dict,
            load_deepcompressor_ptq_artifacts,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = root / "ptq"
            quant.mkdir()
            model, scales, smooth, branch = {}, {}, {}, {}
            source_prefix = "transformer_blocks.0.attn.to_add_out"
            _add_ptq_layer(
                torch,
                model=model,
                scales=scales,
                smooth=smooth,
                branch=branch,
                prefix=source_prefix,
                dtype=torch.float16,
            )
            alias_smooth = torch.linspace(3.0, 4.0, 128, dtype=torch.float16)
            smooth.pop(source_prefix)
            smooth["transformer_blocks.0.attn.to_out.0"] = alias_smooth
            torch.save(model, quant / "model.pt")
            torch.save(scales, quant / "scale.pt")
            torch.save(smooth, quant / "smooth.pt")
            torch.save(branch, quant / "branch.pt")

            artifacts = load_deepcompressor_ptq_artifacts(quant)
            natural, report = build_qwen_image_edit_svdquant_natural_state_dict(artifacts, device="cpu")

            self.assertEqual(report.imported_prefixes, [source_prefix])
            self.assertTrue(torch.equal(natural[f"{source_prefix}.smooth_factor"], alias_smooth))
            expected_raw_proj_down = branch[source_prefix]["a.weight"].transpose(0, 1).to(torch.float32).div(
                alias_smooth.to(torch.float32).reshape(-1, 1)
            )
            expected_raw_proj_down = expected_raw_proj_down.to(natural[f"{source_prefix}.proj_down"].dtype).to(torch.float32)
            self.assertTrue(torch.allclose(natural[f"{source_prefix}.proj_down"].to(torch.float32), expected_raw_proj_down))

    def test_deepcompressor_import_combines_representable_subscale(self):
        torch = self.torch
        from comfy_quants.backends.deepcompressor_import import (
            build_qwen_image_edit_svdquant_natural_state_dict,
            load_deepcompressor_ptq_artifacts,
        )
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = root / "ptq"
            quant.mkdir()
            model, scales, smooth, branch = {}, {}, {}, {}
            prefix = "transformer_blocks.0.attn.to_q"
            n, k, rank = 128, 128, 8
            groups = k // 64
            dense_q = _dense_int4(torch, n=n, k=k, offset=5)
            scale0 = torch.full((n, 1, 1, 1), 2.0, dtype=torch.float16)
            subscale = torch.empty((n, 1, groups, 1), dtype=torch.float16)
            subscale[:, :, 0, :] = 0.5
            subscale[:, :, 1, :] = 0.25
            effective = scale0.view(n, 1, 1) * subscale.view(n, groups, 1)
            model[f"{prefix}.weight"] = dense_q.to(torch.float32).view(n, groups, 64).mul(effective.to(torch.float32)).view(n, k).to(torch.float16)
            scales[f"{prefix}.weight.scale.0"] = scale0
            scales[f"{prefix}.weight.scale.1"] = subscale
            smooth[prefix] = torch.ones((k,), dtype=torch.float16)
            branch[prefix] = {
                "a.weight": torch.arange(rank * k, dtype=torch.float32).view(rank, k).to(torch.float16),
                "b.weight": torch.arange(n * rank, dtype=torch.float32).view(n, rank).to(torch.float16),
            }
            torch.save(model, quant / "model.pt")
            torch.save(scales, quant / "scale.pt")
            torch.save(smooth, quant / "smooth.pt")
            torch.save(branch, quant / "branch.pt")

            artifacts = load_deepcompressor_ptq_artifacts(quant)
            natural, report = build_qwen_image_edit_svdquant_natural_state_dict(artifacts, device="cpu")

            self.assertEqual(report.imported_layer_count, 1)
            self.assertTrue(torch.equal(natural[f"{prefix}.weight"], pack_signed_int4_pairs(dense_q)))
            expected_scale = torch.stack(
                [
                    torch.full((n,), 1.0, dtype=torch.float16),
                    torch.full((n,), 0.5, dtype=torch.float16),
                ],
                dim=0,
            )
            self.assertTrue(torch.equal(natural[f"{prefix}.weight_scale"], expected_scale))

    def test_deepcompressor_import_folds_smooth_into_raw_branch_basis(self):
        torch = self.torch
        from comfy_quants.backends.deepcompressor_import import (
            build_qwen_image_edit_svdquant_natural_state_dict,
            load_deepcompressor_ptq_artifacts,
        )
        from comfy_quants.formats.int4_common import decode_quant_config_tensor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = root / "ptq"
            quant.mkdir()
            model, scales, smooth, branch = {}, {}, {}, {}
            prefix = "transformer_blocks.0.attn.to_q"
            _add_ptq_layer(
                torch,
                model=model,
                scales=scales,
                smooth=smooth,
                branch=branch,
                prefix=prefix,
                dtype=torch.float16,
            )
            smooth[prefix] = torch.linspace(1.0, 4.0, 128, dtype=torch.float16)
            torch.save(model, quant / "model.pt")
            torch.save(scales, quant / "scale.pt")
            torch.save(smooth, quant / "smooth.pt")
            torch.save(branch, quant / "branch.pt")

            artifacts = load_deepcompressor_ptq_artifacts(quant)
            natural, report = build_qwen_image_edit_svdquant_natural_state_dict(artifacts, device="cpu")

            self.assertEqual(report.lowrank_branch_input_basis, "raw")
            self.assertTrue(report.proj_down_smooth_folded)
            expected_raw = branch[prefix]["a.weight"].transpose(0, 1).to(torch.float32).div(
                smooth[prefix].to(torch.float32).reshape(-1, 1)
            )
            expected_raw = expected_raw.to(natural[f"{prefix}.proj_down"].dtype).to(torch.float32)
            self.assertTrue(torch.allclose(natural[f"{prefix}.proj_down"].to(torch.float32), expected_raw, atol=1e-4, rtol=1e-4))
            quant_config = decode_quant_config_tensor(natural[f"{prefix}.comfy_quant"])
            self.assertEqual(quant_config["lowrank_branch_input_basis"], "raw")
            self.assertIs(quant_config["proj_down_smooth_folded"], True)

    def test_deepcompressor_import_applies_shift_bias_correction_to_raw_branch(self):
        torch = self.torch
        from comfy_quants.backends.deepcompressor_import import (
            build_qwen_image_edit_svdquant_natural_state_dict,
            load_deepcompressor_ptq_artifacts,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = root / "ptq"
            quant.mkdir()
            model, scales, smooth, branch = {}, {}, {}, {}
            source_prefix = "transformer_blocks.0.img_mlp.net.2.linear"
            output_prefix = "transformer_blocks.0.img_mlp.net.2"
            n, k, rank = 128, 128, 8
            _add_ptq_layer(
                torch,
                model=model,
                scales=scales,
                smooth=smooth,
                branch=branch,
                prefix=source_prefix,
                n=n,
                k=k,
                rank=rank,
                dtype=torch.float16,
            )
            branch[source_prefix] = {
                "a.weight": (torch.arange(rank * k, dtype=torch.float32).view(rank, k) / 1000.0).to(torch.float16),
                "b.weight": (torch.arange(n * rank, dtype=torch.float32).view(n, rank) / 1000.0).to(torch.float16),
            }
            original_bias = torch.linspace(-0.5, 0.5, n, dtype=torch.float16)
            shift = torch.linspace(-0.2, 0.3, k, dtype=torch.float16)
            smooth[source_prefix] = torch.linspace(1.0, 3.0, k, dtype=torch.float16)
            model[f"{source_prefix}.bias"] = original_bias
            model[f"{output_prefix}.shift"] = shift
            torch.save(model, quant / "model.pt")
            torch.save(scales, quant / "scale.pt")
            torch.save(smooth, quant / "smooth.pt")
            torch.save(branch, quant / "branch.pt")

            artifacts = load_deepcompressor_ptq_artifacts(quant)
            natural, report = build_qwen_image_edit_svdquant_natural_state_dict(artifacts, device="cpu")

            proj_down_raw = branch[source_prefix]["a.weight"].transpose(0, 1).to(torch.float32).div(
                smooth[source_prefix].to(torch.float32).reshape(-1, 1)
            )
            proj_up = branch[source_prefix]["b.weight"].to(torch.float32)
            expected_bias = original_bias.to(torch.float32) + (
                proj_up @ (proj_down_raw.transpose(0, 1) @ shift.to(torch.float32).reshape(k, 1))
            ).reshape(n)
            expected_bias = expected_bias.to(natural[f"{output_prefix}.bias"].dtype).to(torch.float32)
            self.assertEqual(report.shift_bias_correction_count, 1)
            self.assertEqual(report.shift_bias_corrected_prefixes, [output_prefix])
            self.assertTrue(torch.allclose(natural[f"{output_prefix}.bias"].to(torch.float32), expected_bias, atol=5e-3, rtol=5e-3))

    def test_deepcompressor_import_unsigned_layer_matches_shifted_runtime_contract(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import (
            GELU_UNSIGNED_SHIFT,
            quantize_activation_w4_unsigned,
            reference_svdquant_w4a4_linear_runtime,
        )
        from comfy_quants.algorithms.int4_svdquant.weight_quant import dequantize_natural_svdquant_weight
        from comfy_quants.backends.deepcompressor_import import (
            build_qwen_image_edit_svdquant_natural_state_dict,
            load_deepcompressor_ptq_artifacts,
        )
        from comfy_quants.formats.int4_common import decode_quant_config_tensor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = root / "ptq"
            quant.mkdir()
            model, scales, smooth, branch = {}, {}, {}, {}
            source_prefix = "transformer_blocks.0.img_mlp.net.2.linear"
            output_prefix = "transformer_blocks.0.img_mlp.net.2"
            n, k, rank = 128, 128, 4
            _add_ptq_layer(
                torch,
                model=model,
                scales=scales,
                smooth=smooth,
                branch=branch,
                prefix=source_prefix,
                n=n,
                k=k,
                rank=rank,
                dtype=torch.float16,
                offset=4,
            )
            smooth[source_prefix] = torch.linspace(0.9, 1.9, k, dtype=torch.float16)
            branch[source_prefix] = {
                "a.weight": (torch.randn((rank, k), generator=torch.Generator().manual_seed(6101)) * 0.02).to(torch.float16),
                "b.weight": (torch.randn((n, rank), generator=torch.Generator().manual_seed(6102)) * 0.03).to(torch.float16),
            }
            original_bias = torch.linspace(-0.1, 0.1, n, dtype=torch.float16)
            dc_shift = torch.linspace(-0.05, 0.07, k, dtype=torch.float16)
            model[f"{source_prefix}.bias"] = original_bias
            model[f"{output_prefix}.shift"] = dc_shift
            torch.save(model, quant / "model.pt")
            torch.save(scales, quant / "scale.pt")
            torch.save(smooth, quant / "smooth.pt")
            torch.save(branch, quant / "branch.pt")

            artifacts = load_deepcompressor_ptq_artifacts(quant)
            natural, report = build_qwen_image_edit_svdquant_natural_state_dict(artifacts, device="cpu")

            quant_config = decode_quant_config_tensor(natural[f"{output_prefix}.comfy_quant"])
            self.assertIs(quant_config["act_unsigned"], True)
            self.assertEqual(quant_config["lowrank_branch_input_basis"], "raw")
            self.assertIs(quant_config["proj_down_smooth_folded"], True)
            self.assertEqual(report.shift_bias_correction_count, 1)
            self.assertEqual(report.shift_bias_corrected_prefixes, [output_prefix])

            inputs = torch.linspace(-1.2, 0.9, 3 * k, dtype=torch.float32).reshape(3, k)
            main_inputs = inputs + GELU_UNSIGNED_SHIFT
            activation = quantize_activation_w4_unsigned(main_inputs / natural[f"{output_prefix}.smooth_factor"].float().reshape(1, k))
            dense_weight = dequantize_natural_svdquant_weight(
                natural[f"{output_prefix}.weight"],
                natural[f"{output_prefix}.weight_scale"],
            )
            expected = (
                activation.dequantized @ dense_weight.t()
                + inputs @ natural[f"{output_prefix}.proj_down"].float() @ natural[f"{output_prefix}.proj_up"].float().t()
                + natural[f"{output_prefix}.bias"].float().reshape(1, n)
            )
            actual = reference_svdquant_w4a4_linear_runtime(
                inputs,
                natural[f"{output_prefix}.weight"],
                natural[f"{output_prefix}.weight_scale"],
                natural[f"{output_prefix}.smooth_factor"],
                natural[f"{output_prefix}.proj_down"],
                natural[f"{output_prefix}.proj_up"],
                bias=natural[f"{output_prefix}.bias"],
                activation_signedness="unsigned",
                branch_input_basis="raw",
            )
            self.assertTrue(torch.allclose(actual, expected, atol=1e-4, rtol=1e-4))

    def test_cli_export_int4_accepts_deepcompressor_source_format(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = root / "ptq"
            quant.mkdir()
            model, scales, smooth, branch = {}, {}, {}, {}
            _add_ptq_layer(
                torch,
                model=model,
                scales=scales,
                smooth=smooth,
                branch=branch,
                prefix="transformer_blocks.0.attn.to_q",
                dtype=torch.float16,
            )
            torch.save(model, quant / "model.pt")
            torch.save(scales, quant / "scale.pt")
            torch.save(smooth, quant / "smooth.pt")
            torch.save(branch, quant / "branch.pt")

            output_dir = root / "export"
            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "export-int4",
                        "--format",
                        "svdquant_w4a4",
                        "--source-format",
                        "deepcompressor-qwen-image-edit",
                        "--source",
                        str(quant),
                        "--out",
                        str(output_dir),
                        "--device",
                        "cpu",
                        "--json",
                        "--no-progress",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["source_format"], "deepcompressor-qwen-image-edit")
            self.assertEqual(result["source_import"]["imported_layer_count"], 1)
            self.assertTrue((output_dir / "diffusion_pytorch_model.svdquant_w4a4.safetensors").exists())
            report = json.loads((output_dir / "export_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["source_import"]["source_format"], "deepcompressor_ptq_artifacts")

    def test_deepcompressor_import_requires_branch_for_quantized_layer(self):
        torch = self.torch
        from comfy_quants.backends.deepcompressor_import import (
            build_qwen_image_edit_svdquant_natural_state_dict,
            load_deepcompressor_ptq_artifacts,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant = root / "ptq"
            quant.mkdir()
            model, scales, smooth, branch = {}, {}, {}, {}
            _add_ptq_layer(
                torch,
                model=model,
                scales=scales,
                smooth=smooth,
                branch=branch,
                prefix="transformer_blocks.0.attn.to_q",
                dtype=torch.float16,
            )
            torch.save(model, quant / "model.pt")
            torch.save(scales, quant / "scale.pt")
            torch.save(smooth, quant / "smooth.pt")
            torch.save({}, quant / "branch.pt")

            artifacts = load_deepcompressor_ptq_artifacts(quant)
            with self.assertRaisesRegex(PayloadWriteError, "low-rank branch is missing"):
                build_qwen_image_edit_svdquant_natural_state_dict(artifacts, device="cpu")


if __name__ == "__main__":
    unittest.main()
