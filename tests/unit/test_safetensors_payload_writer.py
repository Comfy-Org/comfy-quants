import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.backends.safetensors_payload import write_fp8_e4m3_payload_from_safetensors, write_fp8_e5m2_payload_from_safetensors
from comfy_quants.backends.safetensors_source import SafetensorsTensorSource, build_safetensors_source_coverage
from comfy_quants.cli.main import main
from comfy_quants.core.artifact_layout import DEFAULT_ARTIFACT_PAYLOAD_LAYOUT
from comfy_quants.core.errors import PayloadWriteError


def _torch_safetensors_deps():
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError:
        return None
    if not hasattr(torch, "float8_e4m3fn"):
        return None
    return torch, load_file, save_file


def _single_tensor_index(target_dtype: str = "fp8_e4m3") -> dict:
    tensor_name = "transformer_blocks.0.attn.to_q.weight"
    return {
        "schema_version": "quant_tensor_index.v1",
        "artifact_state": "metadata_only",
        "tensor_payload_state": "pending_export",
        "payload_layout": DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.to_dict(),
        "format": {
            "name": target_dtype,
            "storage_dtype": "uint8",
            "scale_granularity": "per_channel",
            "scale_axis": "out_features",
            "scale_method": "amax",
            "rounding": "nearest_even",
        },
        "selection": {
            "algorithm": "fp8_static",
            "algorithm_version": "0.1.0",
            "target_dtype": target_dtype,
            "quantized_tensor_count": 1,
        },
        "tensors": [
            {
                "name": tensor_name,
                "source_name": tensor_name,
                "shape": [2, 4],
                "source_dtype": "bf16",
                "quant_dtype": target_dtype,
                "storage_dtype": "uint8",
                "algorithm": "fp8_static",
                "scale": {
                    "dtype": "fp32",
                    "shape": [2],
                    "granularity": "per_channel",
                    "axis": "out_features",
                    "file": "scales/fp8_static_scales.safetensors",
                    "tensor_name": f"{tensor_name}.scale",
                },
                "payload": {
                    "file": "tensors/fp8_weights.safetensors",
                    "tensor_name": tensor_name,
                    "storage_dtype": "uint8",
                },
                "rounding": "nearest_even",
                "fallback": False,
                "compatibility_level": "L2",
                "metadata": {"module_name": "transformer_blocks.0.attn.to_q"},
            }
        ],
    }


def _two_tensor_index() -> dict:
    index = _single_tensor_index()
    first = index["tensors"][0]
    second_name = "transformer_blocks.0.attn.to_k.weight"
    second = json.loads(json.dumps(first))
    second["name"] = second_name
    second["source_name"] = second_name
    second["shape"] = [3, 4]
    second["scale"]["shape"] = [3]
    second["scale"]["tensor_name"] = f"{second_name}.scale"
    second["payload"]["tensor_name"] = second_name
    second["metadata"]["module_name"] = "transformer_blocks.0.attn.to_k"
    index["tensors"] = [first, second]
    index["selection"]["quantized_tensor_count"] = 2
    return index


class TestSafetensorsPayloadWriter(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch FP8 and safetensors are required")
        self.torch, self.load_file, self.save_file = deps

    def test_writer_creates_fp8_payload_and_scale_files(self):
        torch = self.torch
        from comfy_quants.backends.torch_ref import dequantize_fp8_e4m3_payload

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            artifact = root / "artifact"
            tensor_name = "transformer_blocks.0.attn.to_q.weight"
            source_tensor = torch.tensor(
                [
                    [0.0, 1.0, -1.0, 2.0],
                    [0.5, -0.25, 0.125, 0.0],
                ],
                dtype=torch.float32,
            )
            self.save_file(
                {
                    tensor_name: source_tensor,
                    "transformer_blocks.0.attn.to_q.bias": torch.ones((2,), dtype=torch.float32),
                },
                str(source),
            )

            report = write_fp8_e4m3_payload_from_safetensors(
                source_checkpoint=source,
                artifact_dir=artifact,
                tensor_index=_single_tensor_index(),
            )

            self.assertEqual(report.status, "payload_written")
            self.assertEqual(report.quantized_tensor_count, 1)
            self.assertEqual(report.missing_tensor_count, 0)
            self.assertTrue((artifact / "tensors" / "fp8_weights.safetensors").exists())
            self.assertTrue((artifact / "scales" / "fp8_static_scales.safetensors").exists())
            self.assertIn("tensors/fp8_weights.safetensors", report.hashes)
            self.assertIn("scales/fp8_static_scales.safetensors", report.hashes)

            payload = self.load_file(str(artifact / "tensors" / "fp8_weights.safetensors"))
            scales = self.load_file(str(artifact / "scales" / "fp8_static_scales.safetensors"))
            self.assertEqual(payload[tensor_name].dtype, torch.uint8)
            self.assertEqual(scales[f"{tensor_name}.scale"].dtype, torch.float32)
            self.assertNotIn("transformer_blocks.0.attn.to_q.bias", payload)

            restored = dequantize_fp8_e4m3_payload(payload[tensor_name], scales[f"{tensor_name}.scale"], axis="out_features")
            self.assertTrue(torch.allclose(restored, source_tensor, rtol=0.06, atol=0.01))

    def test_writer_creates_fp8_e5m2_payload_and_scale_files(self):
        torch = self.torch
        if not hasattr(torch, "float8_e5m2"):
            self.skipTest("torch.float8_e5m2 is unavailable")
        from comfy_quants.backends.torch_ref import dequantize_fp8_e5m2_payload

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            artifact = root / "artifact"
            tensor_name = "transformer_blocks.0.attn.to_q.weight"
            source_tensor = torch.tensor(
                [
                    [0.0, 1.0, -1.0, 2.0],
                    [0.5, -0.25, 0.125, 0.0],
                ],
                dtype=torch.float32,
            )
            self.save_file({tensor_name: source_tensor}, str(source))

            report = write_fp8_e5m2_payload_from_safetensors(
                source_checkpoint=source,
                artifact_dir=artifact,
                tensor_index=_single_tensor_index("fp8_e5m2"),
            )

            self.assertEqual(report.status, "payload_written")
            self.assertEqual(report.target_dtype, "fp8_e5m2")
            payload = self.load_file(str(artifact / "tensors" / "fp8_weights.safetensors"))
            scales = self.load_file(str(artifact / "scales" / "fp8_static_scales.safetensors"))
            self.assertEqual(payload[tensor_name].dtype, torch.uint8)
            restored = dequantize_fp8_e5m2_payload(payload[tensor_name], scales[f"{tensor_name}.scale"], axis="out_features")
            self.assertTrue(torch.allclose(restored, source_tensor, rtol=0.18, atol=0.05))

    def test_writer_fails_on_missing_selected_tensor(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            self.save_file({"other.weight": torch.zeros((2, 4), dtype=torch.float32)}, str(source))

            with self.assertRaises(PayloadWriteError) as ctx:
                write_fp8_e4m3_payload_from_safetensors(
                    source_checkpoint=source,
                    artifact_dir=root / "artifact",
                    tensor_index=_single_tensor_index(),
                )
            self.assertIn("missing selected tensors", str(ctx.exception))

    def test_writer_reads_indexed_safetensors_shards(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "transformer"
            source_dir.mkdir()
            first_name = "transformer_blocks.0.attn.to_q.weight"
            second_name = "transformer_blocks.0.attn.to_k.weight"
            first_file = "diffusion_pytorch_model-00001-of-00002.safetensors"
            second_file = "diffusion_pytorch_model-00002-of-00002.safetensors"
            self.save_file({first_name: torch.zeros((2, 4), dtype=torch.float32)}, str(source_dir / first_file))
            self.save_file({second_name: torch.ones((3, 4), dtype=torch.float32)}, str(source_dir / second_file))
            (source_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 80},
                        "weight_map": {
                            first_name: first_file,
                            second_name: second_file,
                        },
                    }
                ),
                encoding="utf-8",
            )

            source = SafetensorsTensorSource.from_path(source_dir)
            self.assertEqual(source.layout, "indexed_shards")
            self.assertEqual(source.selected_file_counts([first_name, second_name]), {first_file: 1, second_file: 1})
            coverage = build_safetensors_source_coverage(
                source_checkpoint=source_dir,
                tensor_index=_two_tensor_index(),
                check_shapes=True,
            )
            self.assertEqual(coverage.matched_tensor_count, 2)
            self.assertEqual(coverage.shape_checked_tensor_count, 2)
            self.assertEqual(coverage.shape_mismatch_count, 0)

            report = write_fp8_e4m3_payload_from_safetensors(
                source_checkpoint=source_dir,
                artifact_dir=root / "artifact",
                tensor_index=_two_tensor_index(),
            )
            payload = self.load_file(str(root / "artifact" / "tensors" / "fp8_weights.safetensors"))
            scales = self.load_file(str(root / "artifact" / "scales" / "fp8_static_scales.safetensors"))

            self.assertEqual(report.source_layout, "indexed_shards")
            self.assertEqual(report.source_file_count, 2)
            self.assertEqual(report.quantized_tensor_count, 2)
            self.assertIn(first_name, payload)
            self.assertIn(second_name, payload)
            self.assertEqual(list(scales[f"{first_name}.scale"].shape), [2])
            self.assertEqual(list(scales[f"{second_name}.scale"].shape), [3])

    def test_cli_non_dry_run_writes_selected_payload(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "qwen-one-tensor.safetensors"
            run_dir = root / "run"
            run_dir.mkdir()
            wrong_cwd_source = run_dir / "qwen-one-tensor.safetensors"
            tensor_name = "transformer_blocks.0.attn.to_q.weight"
            self.save_file({tensor_name: torch.zeros((3072, 3072), dtype=torch.bfloat16)}, str(source))
            self.save_file({tensor_name: torch.zeros((1, 1), dtype=torch.bfloat16)}, str(wrong_cwd_source))
            config = root / "config.yaml"
            config.write_text(
                f"""
project:
  name: qwen-one-tensor-payload
model:
  family: qwen_image
  model_id: {source.name}
  source: local
  dtype: bf16
quant:
  algorithm: fp8_static
  target_dtype: fp8_e4m3
  scale:
    granularity: per_channel
    axis: out_features
    method: amax
  rounding: nearest_even
  modules:
    include:
      - transformer_blocks.0.attn.to_q
    exclude: []
artifact:
  compatibility_target: L2
""",
                encoding="utf-8",
            )
            work_dir = root / "job"

            captured = StringIO()
            old_cwd = os.getcwd()
            try:
                os.chdir(run_dir)
                with redirect_stdout(captured):
                    rc = main(["quantize", "--config", str(config), "--work-dir", str(work_dir), "--json"])
            finally:
                os.chdir(old_cwd)

            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(captured.getvalue())["status"], "payload_written")
            manifest = json.loads((work_dir / "artifact" / "manifest.json").read_text())
            index = json.loads((work_dir / "artifact" / "quant_tensor_index.json").read_text())
            report = json.loads((work_dir / "artifact" / "payload_report.json").read_text())
            payload = self.load_file(str(work_dir / "artifact" / "tensors" / "fp8_weights.safetensors"))
            scales = self.load_file(str(work_dir / "artifact" / "scales" / "fp8_static_scales.safetensors"))

            self.assertEqual(manifest["compatibility"]["artifact_state"], "payload_written")
            self.assertEqual(manifest["compatibility"]["tensor_payload_state"], "written")
            self.assertEqual(index["tensor_payload_state"], "written")
            self.assertEqual(index["selection"]["quantized_tensor_count"], 1)
            self.assertEqual(report["quantized_tensor_count"], 1)
            self.assertIn(tensor_name, payload)
            self.assertIn(f"{tensor_name}.scale", scales)
            self.assertEqual(payload[tensor_name].dtype, torch.uint8)
            self.assertEqual(list(scales[f"{tensor_name}.scale"].shape), [3072])

            validate_dir = root / "validate"
            captured = StringIO()
            with redirect_stdout(captured):
                rc = main([
                    "validate",
                    "--artifact", str(work_dir / "artifact"),
                    "--out", str(validate_dir),
                    "--json",
                ])
            self.assertEqual(rc, 0)
            validation = json.loads(captured.getvalue())
            self.assertEqual(validation["status"], "valid")
            self.assertEqual(validation["tensor_count"], 1)
            self.assertEqual(validation["payload_tensor_count"], 1)
            self.assertEqual(validation["scale_tensor_count"], 1)
            self.assertTrue((validate_dir / "validation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
