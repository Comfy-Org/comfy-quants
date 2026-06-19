import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.backends.nvfp4_model_export import (
    write_nvfp4_inference_checkpoint_from_safetensors,
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


def _single_tensor_index(in_features: int = 64, out_features: int = 4):
    tensor_name = "transformer_blocks.0.attn.to_q.weight"
    blocks = in_features // 16
    return {
        "schema_version": "quant_tensor_index.v1",
        "artifact_state": "model_export",
        "tensor_payload_state": "written_in_checkpoint",
        "payload_layout": DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.to_dict(),
        "format": {
            "name": "nvfp4",
            "storage_dtype": "uint8",
            "scale_granularity": "block",
            "scale_axis": "in_features",
            "scale_method": "amax",
            "rounding": "nearest_even",
        },
        "selection": {"algorithm": "nvfp4", "algorithm_version": "0.1.0", "target_dtype": "nvfp4", "quantized_tensor_count": 1},
        "tensors": [
            {
                "name": tensor_name,
                "source_name": tensor_name,
                "shape": [out_features, in_features],
                "source_dtype": "bf16",
                "quant_dtype": "nvfp4",
                "storage_dtype": "uint8",
                "algorithm": "nvfp4",
                "scale": {
                    "dtype": "float8_e4m3fn",
                    "shape": [out_features, blocks],
                    "granularity": "block",
                    "axis": "in_features",
                    "block_size": 16,
                    "tensor_name": f"{tensor_name}.scale",
                },
                "rounding": "nearest_even",
                "fallback": False,
                "compatibility_level": "L2",
                "metadata": {"module_name": "transformer_blocks.0.attn.to_q"},
            }
        ],
    }


class TestNvFp4Export(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.load_file, self.save_file = deps

    def test_format_registered(self):
        fmt = get_format("nvfp4")
        self.assertEqual(fmt.storage_dtype, "uint8")
        self.assertEqual(fmt.bits, 4)
        self.assertEqual(fmt.scale_required, True)
        self.assertEqual(fmt.default_scale_granularity, "block")
        self.assertEqual(fmt.metadata["block_size"], 16)

    def test_writer_emits_packed_fp4_two_scales_no_input_scale(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        layer = "transformer_blocks.0.attn.to_q"
        bias_name = "transformer_blocks.0.attn.to_q.bias"
        other = "transformer_blocks.0.norm_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.nvfp4.safetensors"
            self.save_file(
                {
                    tensor_name: torch.randn(4, 64, dtype=torch.float32),
                    bias_name: torch.ones((4,), dtype=torch.float32),
                    other: torch.ones((64,), dtype=torch.float32),
                },
                str(source),
            )
            report = write_nvfp4_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_single_tensor_index(64, 4),
            )
            self.assertEqual(report.status, "model_written")
            self.assertEqual(report.quantized_tensor_count, 1)
            self.assertEqual(report.copied_tensor_count, 2)  # bias + norm
            self.assertEqual(report.tensor_scale_count, 1)
            self.assertEqual(report.quant_storage_dtype, "uint8")
            self.assertEqual(report.scale_dtype, "float8_e4m3fn")
            self.assertEqual(report.per_tensor_scale_dtype, "float32")

            exported = self.load_file(str(output))
            self.assertEqual(exported[tensor_name].dtype, torch.uint8)
            self.assertEqual(list(exported[tensor_name].shape), [4, 32])  # packed [out, in/2]
            ws = exported[f"{layer}.weight_scale"]
            self.assertEqual(ws.dtype, torch.float8_e4m3fn)
            # block grid [4, 64/16=4] -> to_blocked (128*ceil(4/128)=128, 4*ceil(4/4)=4)
            self.assertEqual(list(ws.shape), [128, 4])
            ws2 = exported[f"{layer}.weight_scale_2"]
            self.assertEqual(ws2.dtype, torch.float32)
            self.assertEqual(ws2.dim(), 0)  # per-tensor scalar
            self.assertNotIn(f"{layer}.input_scale", exported)
            marker = json.loads(bytes(exported[f"{layer}.comfy_quant"].tolist()).decode("utf-8"))
            self.assertEqual(marker, {"format": "nvfp4"})
            self.assertEqual(exported[bias_name].dtype, torch.float32)  # copied through
            self.assertEqual(exported[other].dtype, torch.float32)

    def test_writer_rejects_overwriting_source(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.safetensors"
            self.save_file({"transformer_blocks.0.attn.to_q.weight": torch.randn(4, 64, dtype=torch.float32)}, str(source))
            with self.assertRaisesRegex(Exception, "must not overwrite"):
                write_nvfp4_inference_checkpoint_from_safetensors(
                    source_checkpoint=source, output_checkpoint=source, tensor_index=_single_tensor_index(64, 4),
                )

    def test_cli_export_model_nvfp4(self):
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
  name: qwen-one-tensor-nvfp4
model:
  family: qwen_image
  model_id: {source.name}
  source: local
  dtype: bf16
quant:
  algorithm: nvfp4
  target_dtype: nvfp4
  scale:
    granularity: block
    axis: in_features
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
                rc = main(["export-model-nvfp4", "--config", str(config), "--source", str(source), "--out", str(output), "--json", "--no-progress"])
            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "model_written")
            self.assertEqual(result["block_size"], 16)
            exported = self.load_file(str(output))
            self.assertEqual(exported[tensor_name].dtype, torch.uint8)
            self.assertEqual(list(exported[tensor_name].shape), [3072, 1536])  # packed
            # block grid [3072, 3072/16=192] -> (128*24, 4*48) = (3072, 192)
            self.assertEqual(list(exported["transformer_blocks.0.attn.to_q.weight_scale"].shape), [3072, 192])
            self.assertEqual(exported["transformer_blocks.0.attn.to_q.weight_scale"].dtype, torch.float8_e4m3fn)
            self.assertEqual(exported["transformer_blocks.0.attn.to_q.weight_scale_2"].dtype, torch.float32)
            self.assertNotIn("transformer_blocks.0.attn.to_q.input_scale", exported)

    def test_cli_rejects_non_nvfp4_target(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src.safetensors"
            self.save_file({"transformer_blocks.0.attn.to_q.weight": torch.zeros((8, 64), dtype=torch.bfloat16)}, str(source))
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
            rc = main(["export-model-nvfp4", "--config", str(config), "--source", str(source), "--out", str(root / "out"), "--json", "--no-progress"])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
