import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.cli.main import main


def _torch_safetensors_deps():
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        return None
    return torch, save_file


def _packed_svdquant_layer(torch, *, n: int = 128, k: int = 128, rank: int = 4, smooth=None, proj_down=None, offset: int = 0):
    from comfy_quants.formats.int4_common import encode_quant_config_tensor, pack_signed_int4_pairs
    from comfy_quants.formats.kitchen_tilepack import to_kitchen_tile_packed_params

    dense = (torch.arange(n * k, dtype=torch.int16).add(offset).remainder(16) - 8).view(n, k).to(torch.int8)
    smooth = smooth if smooth is not None else torch.linspace(1.0, 2.0, k, dtype=torch.float16)
    proj_down = proj_down if proj_down is not None else torch.arange(k * rank, dtype=torch.float32).view(k, rank).to(torch.float16)
    natural = {
        "weight": pack_signed_int4_pairs(dense),
        "weight_scale": torch.ones((k // 64, n), dtype=torch.float16),
        "smooth_factor": smooth,
        "proj_down": proj_down,
        "proj_up": torch.arange(n * rank, dtype=torch.float32).add(offset).view(n, rank).to(torch.float16),
        "bias": torch.arange(n, dtype=torch.float32).to(torch.float16),
        "comfy_quant": encode_quant_config_tensor(
            {
                "format": "svdquant_w4a4",
                "layout": "kitchen_tile_packed_w4a4",
                "lowrank_branch_input_basis": "raw",
                "proj_down_smooth_folded": True,
            }
        ),
    }
    return to_kitchen_tile_packed_params(natural)


def _write_split_qkv_fixture(torch, save_file, path: Path):
    tensors = {}
    rank = 4
    k = 128
    image_smooth = torch.linspace(1.0, 2.0, k, dtype=torch.float16)
    image_down = torch.arange(k * rank, dtype=torch.float32).view(k, rank).to(torch.float16)
    text_smooth = torch.linspace(2.0, 3.0, k, dtype=torch.float16)
    text_down = torch.arange(k * rank, dtype=torch.float32).add(1000).view(k, rank).to(torch.float16)

    for index, suffix in enumerate(("attn.to_q", "attn.to_k", "attn.to_v")):
        prefix = f"transformer_blocks.0.{suffix}"
        params = _packed_svdquant_layer(torch, rank=rank, k=k, smooth=image_smooth, proj_down=image_down, offset=index * 17)
        tensors.update({f"{prefix}.{key}": value.clone() for key, value in params.items()})

    for index, suffix in enumerate(("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")):
        prefix = f"transformer_blocks.0.{suffix}"
        params = _packed_svdquant_layer(torch, rank=rank, k=k, smooth=text_smooth, proj_down=text_down, offset=100 + index * 17)
        tensors.update({f"{prefix}.{key}": value.clone() for key, value in params.items()})

    tensors["transformer_blocks.0.norm.weight"] = torch.ones((128,), dtype=torch.float16)
    save_file(tensors, str(path), metadata={"target_dtype": "svdquant_w4a4", "storage_layout": "kitchen_tile_packed_w4a4"})


class TestInt4ArtifactInspectCli(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.save_file = deps

    def test_backend_inspects_split_qkv_tilepack_fixture(self):
        from comfy_quants.backends.int4_artifact_inspect import inspect_svdquant_w4a4_artifact

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.safetensors"
            _write_split_qkv_fixture(self.torch, self.save_file, artifact)

            report = inspect_svdquant_w4a4_artifact(
                artifact,
                family="qwen_image_edit",
                require_all_lowrank=True,
                check_qkv_splits=True,
            )

            self.assertEqual(report.status, "ok")
            self.assertTrue(report.ok_expected_counts)
            self.assertEqual(report.svdquant_w4a4_count, 6)
            self.assertEqual(report.svdquant_lowrank_count, 6)
            self.assertEqual(report.missing_required_tensor_count, 0)
            self.assertEqual(report.bad_layout_count, 0)
            self.assertEqual(report.bad_shape_count, 0)
            self.assertEqual(report.qkv_group_count, 2)
            self.assertEqual(report.qkv_split_prefix_count, 6)
            self.assertEqual(report.qkv_full_proj_up_shape_count, 0)
            self.assertEqual(report.qkv_rank0_count, 0)
            self.assertEqual(report.qkv_bad_shared_count, 0)

    def test_cli_inspect_int4_writes_json_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.safetensors"
            report_path = root / "inspection.json"
            _write_split_qkv_fixture(self.torch, self.save_file, artifact)

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "inspect-int4",
                        "--artifact",
                        str(artifact),
                        "--family",
                        "qwen_image_edit",
                        "--format",
                        "svdquant_w4a4",
                        "--require-all-lowrank",
                        "--check-qkv-splits",
                        "--out",
                        str(report_path),
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["svdquant_w4a4_count"], 6)
            self.assertEqual(result["qkv_group_count"], 2)
            self.assertTrue(report_path.exists())
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["qkv_split_prefix_count"], 6)

    def test_cli_strict_qwen_image_edit_2511_fails_on_small_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.safetensors"
            _write_split_qkv_fixture(self.torch, self.save_file, artifact)

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "inspect-int4",
                        "--artifact",
                        str(artifact),
                        "--strict-qwen-image-edit-2511",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 2)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "failed")
            checks = {item["check"] for item in result["errors"]}
            self.assertIn("expected_svdquant_layers", checks)
            self.assertIn("expected_qkv_group_count", checks)
            self.assertIn("expected_qkv_split_prefix_count", checks)

    def test_inspector_flags_unsplit_grouped_qkv_proj_up(self):
        from comfy_quants.backends.int4_artifact_inspect import inspect_svdquant_w4a4_artifact

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "bad.safetensors"
            _write_split_qkv_fixture(self.torch, self.save_file, artifact)

            from safetensors.torch import load_file, save_file

            tensors = load_file(str(artifact))
            tensors["transformer_blocks.0.attn.to_q.proj_up"] = self.torch.zeros((3, 4, 128), dtype=self.torch.float16)
            save_file(tensors, str(artifact))

            report = inspect_svdquant_w4a4_artifact(
                artifact,
                family="qwen_image_edit",
                require_all_lowrank=True,
                check_qkv_splits=True,
            )

            self.assertEqual(report.status, "failed")
            self.assertEqual(report.qkv_full_proj_up_shape_count, 1)
            self.assertGreaterEqual(report.bad_shape_count, 1)


if __name__ == "__main__":
    unittest.main()
