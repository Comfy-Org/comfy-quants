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
        from safetensors.torch import load_file, save_file
    except ImportError:
        return None
    return torch, load_file, save_file


class TestInt4GptqHessianArtifact(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.load_file, self.save_file = deps

    def test_reduce_gptq_hessians_writes_manifest_and_tensors(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.calibration import load_activation_sample_refs
        from comfy_quants.algorithms.int4_svdquant.hessian import (
            GPTQ_HESSIAN_STATS_SCHEMA_VERSION,
            load_gptq_hessian_manifest,
            reduce_gptq_hessians_from_safetensors,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.safetensors"
            second = root / "second.safetensors"
            samples = root / "samples.jsonl"
            out = root / "hessians"
            layer_a = "transformer_blocks.0.attn.to_q"
            layer_b = "transformer_blocks.0.ff.net.0"
            a0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            a1 = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
            b0 = torch.tensor([[[1.0, 0.0, 2.0]]], dtype=torch.float32)
            self.save_file({"a": a0, "b": b0}, str(first))
            self.save_file({"a": a1}, str(second))
            samples.write_text(
                "\n".join(
                    [
                        json.dumps({"layer": layer_a, "file": first.name, "tensor": "a"}),
                        json.dumps({"layer": layer_a, "file": second.name, "tensor": "a"}),
                        json.dumps({"layer": layer_b, "file": first.name, "tensor": "b"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            refs = load_activation_sample_refs(samples)
            report = reduce_gptq_hessians_from_safetensors(refs, output_dir=out, samples_path=samples, device="cpu")

            self.assertEqual(report.status, "ok")
            self.assertEqual(report.layer_count, 2)
            self.assertEqual(report.sample_ref_count, 3)
            self.assertEqual(report.row_count, 4)
            manifest_path = out / "int4_gptq_hessian_stats.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], GPTQ_HESSIAN_STATS_SCHEMA_VERSION)
            self.assertEqual(manifest["normalization"], "two_over_row_count")
            self.assertEqual(manifest["layer_count"], 2)
            self.assertEqual(set(manifest["layers"]), {layer_a, layer_b})

            records = load_gptq_hessian_manifest(manifest_path)
            record_a = records[layer_a]
            self.assertEqual(record_a.channel_count, 2)
            self.assertEqual(record_a.sample_count, 2)
            self.assertEqual(record_a.row_count, 3)
            self.assertEqual(record_a.normalization_count, 3)
            self.assertEqual(record_a.shape, [2, 2])
            self.assertFalse(Path(record_a.file_path).is_absolute())

            hessian_a = self.load_file(str(out / record_a.file_path))[record_a.tensor_name]
            stacked_a = torch.cat([a0, a1], dim=0)
            expected_a = stacked_a.t().matmul(stacked_a) * (2.0 / float(stacked_a.shape[0]))
            self.assertTrue(torch.allclose(hessian_a, expected_a))

            record_b = records[layer_b]
            hessian_b = self.load_file(str(out / record_b.file_path))[record_b.tensor_name]
            flat_b = b0.reshape(-1, 3)
            expected_b = flat_b.t().matmul(flat_b) * 2.0
            self.assertTrue(torch.allclose(hessian_b, expected_b))

    def test_cli_reduce_int4_gptq_hessians(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            activation_file = root / "case-1.safetensors"
            samples = root / "samples.jsonl"
            out = root / "hessian-out"
            layer = "transformer_blocks.0.attn.to_k"
            activation = torch.tensor([[1.0, -1.0], [2.0, 0.0]], dtype=torch.float32)
            self.save_file({"hidden": activation}, str(activation_file))
            samples.write_text(json.dumps({"layer": layer, "file": activation_file.name, "tensor": "hidden"}) + "\n", encoding="utf-8")

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "calib",
                        "reduce-int4-gptq-hessians",
                        "--samples",
                        str(samples),
                        "--out",
                        str(out),
                        "--device",
                        "cpu",
                        "--no-progress",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["layer_count"], 1)
            self.assertEqual(result["sample_ref_count"], 1)
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["output"], str(out / "int4_gptq_hessian_stats.json"))
            self.assertTrue((out / "int4_gptq_hessian_stats.json").exists())

    def test_reduce_gptq_hessians_rejects_channel_mismatch(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.calibration import load_activation_sample_refs
        from comfy_quants.algorithms.int4_svdquant.hessian import reduce_gptq_hessians_from_safetensors

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.safetensors"
            second = root / "second.safetensors"
            samples = root / "samples.jsonl"
            layer = "transformer_blocks.0.attn.to_v"
            self.save_file({"hidden": torch.zeros((1, 2), dtype=torch.float32)}, str(first))
            self.save_file({"hidden": torch.zeros((1, 3), dtype=torch.float32)}, str(second))
            samples.write_text(
                json.dumps({"layer": layer, "file": first.name, "tensor": "hidden"})
                + "\n"
                + json.dumps({"layer": layer, "file": second.name, "tensor": "hidden"})
                + "\n",
                encoding="utf-8",
            )

            refs = load_activation_sample_refs(samples)
            with self.assertRaisesRegex(ValueError, "channel count changed"):
                reduce_gptq_hessians_from_safetensors(refs, output_dir=root / "out", device="cpu")


if __name__ == "__main__":
    unittest.main()
