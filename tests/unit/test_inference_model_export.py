import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.backends.inference_model_export import (
    write_fp8_e4m3_inference_checkpoint_from_safetensors,
    write_fp8_e5m2_inference_checkpoint_from_safetensors,
)
from comfy_quants.cli.main import main
from comfy_quants.core.artifact_layout import DEFAULT_ARTIFACT_PAYLOAD_LAYOUT


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
            "scale_granularity": "per_tensor",
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
                    "shape": [1],
                    "granularity": "per_tensor",
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


class TestInferenceModelExport(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch FP8 and safetensors are required")
        self.torch, self.load_file, self.save_file = deps

    def test_writer_creates_full_checkpoint_with_quant_metadata(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        layer_name = "transformer_blocks.0.attn.to_q"
        bias_name = "transformer_blocks.0.attn.to_q.bias"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.fp8.safetensors"
            self.save_file(
                {
                    tensor_name: torch.tensor([[0.0, 1.0, -1.0, 2.0], [0.5, -0.25, 0.125, 0.0]], dtype=torch.float32),
                    bias_name: torch.ones((2,), dtype=torch.float32),
                },
                str(source),
            )

            report = write_fp8_e4m3_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_single_tensor_index(),
            )

            self.assertEqual(report.status, "model_written")
            self.assertEqual(report.quantized_tensor_count, 1)
            self.assertEqual(report.copied_tensor_count, 1)
            self.assertEqual(report.output_tensor_count, 5)
            self.assertTrue(output.exists())

            exported = self.load_file(str(output))
            self.assertEqual(exported[tensor_name].dtype, torch.float8_e4m3fn)
            self.assertEqual(exported[f"{layer_name}.weight_scale"].dtype, torch.float32)
            self.assertEqual(exported[f"{layer_name}.input_scale"].dtype, torch.float32)
            self.assertEqual(exported[f"{layer_name}.input_scale"].shape, torch.Size([]))
            self.assertEqual(float(exported[f"{layer_name}.input_scale"].item()), 1.0)
            self.assertEqual(exported[f"{layer_name}.comfy_quant"].dtype, torch.uint8)
            self.assertEqual(exported[bias_name].dtype, torch.float32)
            quant_conf = json.loads(bytes(exported[f"{layer_name}.comfy_quant"].tolist()).decode("utf-8"))
            self.assertEqual(quant_conf, {"format": "float8_e4m3fn", "full_precision_matrix_mult": True})

    def test_writer_creates_fp8_e5m2_checkpoint_with_quant_metadata(self):
        torch = self.torch
        if not hasattr(torch, "float8_e5m2"):
            self.skipTest("torch.float8_e5m2 is unavailable")
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        layer_name = "transformer_blocks.0.attn.to_q"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.fp8_e5m2.safetensors"
            self.save_file(
                {tensor_name: torch.tensor([[0.0, 1.0, -1.0, 2.0], [0.5, -0.25, 0.125, 0.0]], dtype=torch.float32)},
                str(source),
            )

            report = write_fp8_e5m2_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_single_tensor_index("fp8_e5m2"),
            )

            self.assertEqual(report.status, "model_written")
            self.assertEqual(report.target_dtype, "fp8_e5m2")
            self.assertEqual(report.quant_storage_dtype, "float8_e5m2")
            exported = self.load_file(str(output))
            self.assertEqual(exported[tensor_name].dtype, torch.float8_e5m2)
            self.assertEqual(exported[f"{layer_name}.weight_scale"].dtype, torch.float32)
            self.assertEqual(exported[f"{layer_name}.input_scale"].dtype, torch.float32)
            self.assertEqual(float(exported[f"{layer_name}.input_scale"].item()), 1.0)
            quant_conf = json.loads(bytes(exported[f"{layer_name}.comfy_quant"].tolist()).decode("utf-8"))
            self.assertEqual(quant_conf, {"format": "float8_e5m2", "full_precision_matrix_mult": True})

    def test_writer_copies_adjacent_model_config_for_single_file_source(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            source = model_dir / "source.safetensors"
            (model_dir / "config.json").write_text('{"model_type":"qwen_image"}\n', encoding="utf-8")
            output = root / "export" / "model.fp8.safetensors"
            self.save_file({tensor_name: torch.zeros((2, 4), dtype=torch.float32)}, str(source))

            report = write_fp8_e4m3_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_single_tensor_index(),
                config_source=source,
            )

            copied_config = output.parent / "config.json"
            self.assertEqual(report.config_path, str(copied_config))
            self.assertEqual(json.loads(copied_config.read_text(encoding="utf-8")), {"model_type": "qwen_image"})

    def test_writer_emits_qwen_edit_2511_reference_marker(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.fp8.safetensors"
            self.save_file({tensor_name: torch.zeros((2, 4), dtype=torch.float32)}, str(source))
            tensor_index = _single_tensor_index()
            tensor_index["reference_image_mode"] = "index_timestep_zero"

            write_fp8_e4m3_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=tensor_index,
            )

            exported = self.load_file(str(output))
            self.assertIn("__index_timestep_zero__", exported)
            self.assertEqual(exported["__index_timestep_zero__"].dtype, torch.float32)
            self.assertEqual(exported["__index_timestep_zero__"].shape, torch.Size([0]))

    def test_writer_rejects_overwriting_source_tensor_file(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.safetensors"
            self.save_file({tensor_name: torch.zeros((2, 4), dtype=torch.float32)}, str(source))

            with self.assertRaisesRegex(Exception, "must not overwrite"):
                write_fp8_e4m3_inference_checkpoint_from_safetensors(
                    source_checkpoint=source,
                    output_checkpoint=source,
                    tensor_index=_single_tensor_index(),
                )

    def test_cli_export_model_writes_checkpoint_and_report(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "qwen-one-tensor.safetensors"
            self.save_file({tensor_name: torch.zeros((3072, 3072), dtype=torch.bfloat16)}, str(source))
            config = root / "config.yaml"
            config.write_text(
                f"""
project:
  name: qwen-one-tensor-model
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
            output = root / "export" / "model.safetensors"

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main([
                    "export-model",
                    "--config",
                    str(config),
                    "--source",
                    str(source),
                    "--out",
                    str(output),
                    "--json",
                ])

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "model_written")
            self.assertTrue(output.exists())
            self.assertTrue((root / "export" / "model.export_report.json").exists())
            self.assertFalse((root / "export" / "config.json").exists())
            exported = self.load_file(str(output))
            self.assertIn(tensor_name, exported)
            self.assertIn("transformer_blocks.0.attn.to_q.comfy_quant", exported)
            self.assertEqual(exported[tensor_name].dtype, torch.float8_e4m3fn)

    def test_cli_local_model_id_is_resolved_from_config_directory(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            run_dir = root / "run"
            config_dir.mkdir()
            run_dir.mkdir()

            correct_source = config_dir / "source.safetensors"
            wrong_cwd_source = run_dir / "source.safetensors"
            self.save_file({tensor_name: torch.zeros((3072, 3072), dtype=torch.bfloat16)}, str(correct_source))
            self.save_file({tensor_name: torch.zeros((1, 1), dtype=torch.bfloat16)}, str(wrong_cwd_source))

            config = config_dir / "config.yaml"
            config.write_text(
                """
project:
  name: qwen-relative-source
model:
  family: qwen_image
  model_id: source.safetensors
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
            output = root / "export"

            old_cwd = os.getcwd()
            try:
                os.chdir(run_dir)
                captured = StringIO()
                with redirect_stdout(captured):
                    rc = main([
                        "export-model",
                        "--config",
                        str(config),
                        "--out",
                        str(output),
                        "--json",
                        "--no-progress",
                    ])
            finally:
                os.chdir(old_cwd)

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "model_written")
            self.assertTrue((output / "diffusion_pytorch_model.fp8_e4m3.safetensors").exists())

    def test_cli_export_model_e5m2_output_directory_uses_target_suffix(self):
        torch = self.torch
        if not hasattr(torch, "float8_e5m2"):
            self.skipTest("torch.float8_e5m2 is unavailable")
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "qwen-one-tensor.safetensors"
            self.save_file({tensor_name: torch.zeros((3072, 3072), dtype=torch.bfloat16)}, str(source))
            config = root / "config.yaml"
            config.write_text(
                f"""
project:
  name: qwen-one-tensor-model-e5m2
model:
  family: qwen_image
  model_id: {source.name}
  source: local
  dtype: bf16
quant:
  algorithm: fp8_static
  target_dtype: fp8_e5m2
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
            output = root / "export"

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main([
                    "export-model",
                    "--config",
                    str(config),
                    "--source",
                    str(source),
                    "--out",
                    str(output),
                    "--json",
                    "--no-progress",
                ])

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            checkpoint = output / "diffusion_pytorch_model.fp8_e5m2.safetensors"
            self.assertEqual(result["status"], "model_written")
            self.assertTrue(checkpoint.exists())
            exported = self.load_file(str(checkpoint))
            self.assertEqual(exported[tensor_name].dtype, torch.float8_e5m2)


if __name__ == "__main__":
    unittest.main()
