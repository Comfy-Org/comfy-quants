import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.backends.int4_kitchen_export import write_svdquant_w4a4_kitchen_checkpoint_from_safetensors
from comfy_quants.cli.main import main
from comfy_quants.core.errors import PayloadWriteError


def _torch_safetensors_deps():
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
    except ImportError:
        return None
    return torch, safe_open, load_file, save_file


def _natural_svdquant_params(torch, *, n: int = 128, k: int = 128, rank: int = 8):
    from comfy_quants.formats.int4_common import encode_quant_config_tensor, pack_signed_int4_pairs

    dense = (torch.arange(n * k, dtype=torch.int16).remainder(16) - 8).view(n, k).to(torch.int8)
    return {
        "weight": pack_signed_int4_pairs(dense),
        "weight_scale": torch.arange((k // 64) * n, dtype=torch.float32).view(k // 64, n).to(torch.float16),
        "smooth_factor": torch.arange(k, dtype=torch.float32).to(torch.float16),
        "proj_down": torch.arange(k * rank, dtype=torch.float32).view(k, rank).to(torch.float16),
        "proj_up": torch.arange(n * rank, dtype=torch.float32).view(n, rank).to(torch.float16),
        "bias": torch.arange(n, dtype=torch.float32).to(torch.float16),
        "comfy_quant": encode_quant_config_tensor({"format": "svdquant_w4a4", "act_unsigned": True}),
    }


class TestInt4KitchenExport(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.safe_open, self.load_file, self.save_file = deps

    def test_writer_repackages_natural_svdquant_checkpoint(self):
        torch = self.torch
        from comfy_quants.formats.int4_common import decode_quant_config_tensor
        from comfy_quants.formats.kitchen_tilepack import unpack_n_axis, unpack_weight_scale, unpack_weight_tile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "out" / "model.svdquant_w4a4.safetensors"
            params = _natural_svdquant_params(torch)
            params["weight_scale"] = params["weight_scale"].to(torch.float32)
            prefix = "transformer_blocks.0.attn.to_q"
            source_tensors = {f"{prefix}.{key}": value for key, value in params.items()}
            source_tensors["transformer_blocks.0.norm.weight"] = torch.ones((128,), dtype=torch.float16)
            self.save_file(source_tensors, str(source))

            report = write_svdquant_w4a4_kitchen_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                device="cpu",
                hash_output=True,
            )

            self.assertEqual(report.status, "model_written")
            self.assertEqual(report.target_dtype, "svdquant_w4a4")
            self.assertEqual(report.storage_layout, "kitchen_tile_packed_w4a4")
            self.assertEqual(report.repacked_layer_count, 1)
            self.assertEqual(report.repacked_tensor_count, 7)
            self.assertEqual(report.copied_tensor_count, 1)
            self.assertEqual(report.output_tensor_count, 8)
            self.assertEqual(report.output_hash_state, "written")
            self.assertEqual(report.repacked_prefixes, [prefix])
            self.assertIn("source.safetensors", report.selected_source_files)
            self.assertTrue(output.exists())

            exported = self.load_file(str(output))
            self.assertEqual(tuple(exported[f"{prefix}.weight"].shape), (1, 2, 32, 128))
            self.assertEqual(tuple(exported[f"{prefix}.weight_scale"].shape), (1, 2, 128))
            self.assertEqual(tuple(exported[f"{prefix}.proj_up"].shape), (1, 8, 128))
            self.assertEqual(exported[f"{prefix}.weight"].dtype, torch.int8)
            self.assertEqual(exported[f"{prefix}.weight_scale"].dtype, torch.bfloat16)
            self.assertEqual(exported["transformer_blocks.0.norm.weight"].dtype, torch.float16)
            self.assertTrue(torch.equal(unpack_weight_tile(exported[f"{prefix}.weight"]), params["weight"]))
            self.assertTrue(torch.allclose(unpack_weight_scale(exported[f"{prefix}.weight_scale"]).float(), params["weight_scale"], atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.equal(unpack_n_axis(exported[f"{prefix}.proj_up"]), params["proj_up"]))

            quant_config = decode_quant_config_tensor(exported[f"{prefix}.comfy_quant"])
            self.assertEqual(quant_config["format"], "svdquant_w4a4")
            self.assertEqual(quant_config["layout"], "kitchen_tile_packed_w4a4")
            self.assertIs(quant_config["act_unsigned"], True)

            with self.safe_open(str(output), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            self.assertEqual(metadata["target_dtype"], "svdquant_w4a4")
            self.assertEqual(metadata["storage_layout"], "kitchen_tile_packed_w4a4")
            self.assertEqual(metadata["artifact_contract"], "svdquant_w4a4_kitchen_tilepack.v1")

    def test_writer_rejects_missing_svdquant_layers_by_default(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            self.save_file({"not_quant.weight": torch.zeros((2, 2), dtype=torch.float16)}, str(source))

            with self.assertRaisesRegex(PayloadWriteError, "no SVDQuant W4A4 layers"):
                write_svdquant_w4a4_kitchen_checkpoint_from_safetensors(
                    source_checkpoint=source,
                    output_checkpoint=root / "out.safetensors",
                    device="cpu",
                )

    def test_writer_rejects_overwriting_source_file(self):
        torch = self.torch
        params = _natural_svdquant_params(torch)
        source_tensors = {f"layer.{key}": value for key, value in params.items()}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.safetensors"
            self.save_file(source_tensors, str(source))

            with self.assertRaisesRegex(PayloadWriteError, "must not overwrite"):
                write_svdquant_w4a4_kitchen_checkpoint_from_safetensors(
                    source_checkpoint=source,
                    output_checkpoint=source,
                    device="cpu",
                )

    def test_cli_export_int4_writes_checkpoint_and_report(self):
        torch = self.torch
        params = _natural_svdquant_params(torch)
        source_tensors = {f"layer.{key}": value for key, value in params.items()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output_dir = root / "export"
            self.save_file(source_tensors, str(source))

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "export-int4",
                        "--format",
                        "svdquant_w4a4",
                        "--source",
                        str(source),
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
            checkpoint = output_dir / "diffusion_pytorch_model.svdquant_w4a4.safetensors"
            report_path = output_dir / "export_report.json"
            self.assertEqual(result["status"], "model_written")
            self.assertEqual(result["format"], "svdquant_w4a4")
            self.assertEqual(result["repacked_layer_count"], 1)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(report_path.exists())
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["storage_layout"], "kitchen_tile_packed_w4a4")


if __name__ == "__main__":
    unittest.main()
