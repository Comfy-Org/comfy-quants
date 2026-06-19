import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.backends.int8_w8a8_model_export import (
    write_int8_w8a8_inference_checkpoint_from_safetensors,
)
from comfy_quants.cli.main import main
from comfy_quants.core.artifact_layout import DEFAULT_ARTIFACT_PAYLOAD_LAYOUT
from comfy_quants.formats.registry import get_format


def _torch_safetensors_deps():
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError:
        return None
    return torch, load_file, save_file


def _single_tensor_index(in_features: int = 256):
    tensor_name = "transformer_blocks.0.attn.to_q.weight"
    return {
        "schema_version": "quant_tensor_index.v1",
        "artifact_state": "model_export",
        "tensor_payload_state": "written_in_checkpoint",
        "payload_layout": DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.to_dict(),
        "format": {
            "name": "int8_w8a8",
            "storage_dtype": "int8",
            "scale_granularity": "per_channel",
            "scale_axis": "out_features",
            "scale_method": "amax",
            "rounding": "nearest_even",
        },
        "selection": {"algorithm": "int8_w8a8", "algorithm_version": "0.1.0", "target_dtype": "int8_w8a8", "quantized_tensor_count": 1},
        "tensors": [
            {
                "name": tensor_name,
                "source_name": tensor_name,
                "shape": [4, in_features],
                "source_dtype": "bf16",
                "quant_dtype": "int8_w8a8",
                "storage_dtype": "int8",
                "algorithm": "int8_w8a8",
                "scale": {"dtype": "fp32", "shape": [4], "granularity": "per_channel", "axis": "out_features", "tensor_name": f"{tensor_name}.scale"},
                "rounding": "nearest_even",
                "fallback": False,
                "compatibility_level": "L2",
                "metadata": {"module_name": "transformer_blocks.0.attn.to_q"},
            }
        ],
    }


class TestInt8W8A8Export(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.load_file, self.save_file = deps

    def test_format_registered(self):
        fmt = get_format("int8_w8a8")
        self.assertEqual(fmt.storage_dtype, "int8")
        self.assertEqual(fmt.bits, 8)
        self.assertEqual(fmt.scale_required, True)
        self.assertEqual(fmt.default_scale_granularity, "per_channel")

    def test_writer_emits_int8_perrow_scale_marker_no_input_scale(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        layer = "transformer_blocks.0.attn.to_q"
        bias_name = "transformer_blocks.0.attn.to_q.bias"
        other = "transformer_blocks.0.norm_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.int8_w8a8.safetensors"
            self.save_file(
                {
                    tensor_name: torch.randn(4, 256, dtype=torch.float32),
                    bias_name: torch.ones((4,), dtype=torch.float32),
                    other: torch.ones((256,), dtype=torch.float32),
                },
                str(source),
            )
            report = write_int8_w8a8_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_single_tensor_index(256),
                convrot=True,
            )
            self.assertEqual(report.status, "model_written")
            self.assertEqual(report.quantized_tensor_count, 1)
            self.assertEqual(report.rotated_tensor_count, 1)  # 256 % 256 == 0
            self.assertEqual(report.copied_tensor_count, 2)   # bias + norm
            self.assertEqual(report.scale_tensor_count, 1)

            exported = self.load_file(str(output))
            self.assertEqual(exported[tensor_name].dtype, torch.int8)
            self.assertEqual(list(exported[tensor_name].shape), [4, 256])
            ws = exported[f"{layer}.weight_scale"]
            self.assertEqual(ws.dtype, torch.float32)
            self.assertEqual(list(ws.shape), [4, 1])  # 2D per-row -> downstream _is_per_row
            self.assertNotIn(f"{layer}.input_scale", exported)
            marker = json.loads(bytes(exported[f"{layer}.comfy_quant"].tolist()).decode("utf-8"))
            self.assertEqual(marker, {"convrot": True, "convrot_groupsize": 256, "per_row": True})
            self.assertEqual(exported[bias_name].dtype, torch.float32)  # copied through
            self.assertEqual(exported[other].dtype, torch.float32)

    def test_no_convrot_marker(self):
        torch = self.torch
        layer = "transformer_blocks.0.attn.to_q"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.safetensors"
            self.save_file({f"{layer}.weight": torch.randn(4, 256, dtype=torch.float32)}, str(source))
            report = write_int8_w8a8_inference_checkpoint_from_safetensors(
                source_checkpoint=source, output_checkpoint=output, tensor_index=_single_tensor_index(256), convrot=False,
            )
            self.assertEqual(report.rotated_tensor_count, 0)
            exported = self.load_file(str(output))
            marker = json.loads(bytes(exported[f"{layer}.comfy_quant"].tolist()).decode("utf-8"))
            self.assertEqual(marker, {"convrot": False, "per_row": True})

    def test_writer_rejects_overwriting_source(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.safetensors"
            self.save_file({"transformer_blocks.0.attn.to_q.weight": torch.randn(4, 256, dtype=torch.float32)}, str(source))
            with self.assertRaisesRegex(Exception, "must not overwrite"):
                write_int8_w8a8_inference_checkpoint_from_safetensors(
                    source_checkpoint=source, output_checkpoint=source, tensor_index=_single_tensor_index(256),
                )

    def test_cli_export_model_w8a8(self):
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
  name: qwen-one-tensor-w8a8
model:
  family: qwen_image
  model_id: {source.name}
  source: local
  dtype: bf16
quant:
  algorithm: int8_w8a8
  target_dtype: int8_w8a8
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
                rc = main(["export-model-w8a8", "--config", str(config), "--source", str(source), "--out", str(output), "--json", "--no-progress"])
            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "model_written")
            self.assertTrue(result["convrot"])
            exported = self.load_file(str(output))
            self.assertEqual(exported[tensor_name].dtype, torch.int8)
            self.assertEqual(list(exported["transformer_blocks.0.attn.to_q.weight_scale"].shape), [3072, 1])
            self.assertNotIn("transformer_blocks.0.attn.to_q.input_scale", exported)

    def test_cli_rejects_non_w8a8_target(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src.safetensors"
            self.save_file({"transformer_blocks.0.attn.to_q.weight": torch.zeros((8, 256), dtype=torch.bfloat16)}, str(source))
            config = root / "config.yaml"
            config.write_text(
                f"""
project:
  name: wrong-target
model:
  family: qwen_image
  model_id: {source.name}
  source: local
quant:
  algorithm: fp8_static
  target_dtype: fp8_e4m3
""",
                encoding="utf-8",
            )
            rc = main(["export-model-w8a8", "--config", str(config), "--source", str(source), "--out", str(root / "out"), "--json", "--no-progress"])
            self.assertEqual(rc, 2)  # ConfigurationError -> handle_cli_error -> exit 2


if __name__ == "__main__":
    unittest.main()
