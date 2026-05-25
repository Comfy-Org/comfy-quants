import json
import tempfile
import unittest
from pathlib import Path

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.cli.main import main
from comfy_quants.core.config import load_quant_config
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter


def _build_index(config_path: str):
    cfg = load_quant_config(config_path)
    adapter = get_adapter(cfg.model.family)
    _, graph = adapter.inspect(ModelSource(family=cfg.model.family, model_id=cfg.model.model_id, revision=cfg.model.revision))
    policy = adapter.default_policy(cfg.quant.target_dtype)
    policy.algorithm = cfg.quant.algorithm
    policy.include = cfg.quant.modules.get("include", policy.include)
    policy.exclude = cfg.quant.modules.get("exclude", policy.exclude)
    return build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm=cfg.quant.algorithm,
            algorithm_version="0.1.0",
            target_dtype=cfg.quant.target_dtype,
            scale_granularity=cfg.quant.scale.granularity,
            scale_axis=cfg.quant.scale.axis,
            scale_method=cfg.quant.scale.method,
            rounding=cfg.quant.rounding,
            compatibility_level=cfg.artifact.compatibility_target,
        ),
    )


class TestFP8TensorIndex(unittest.TestCase):
    def test_qwen_image_fp8_index_contains_weight_tensors_and_scales(self):
        index = _build_index("configs/qwen_image_2512_fp8_static.yaml")
        self.assertEqual(index["schema_version"], "quant_tensor_index.v1")
        self.assertEqual(index["artifact_target"], "comfyui")
        self.assertEqual(index["contract_source"], "comfy_quants")
        self.assertEqual(index["format"]["name"], "fp8_e4m3")
        self.assertEqual(index["format"]["scale_granularity"], "per_tensor")
        self.assertIsNone(index["format"]["scale_axis"])
        self.assertEqual(index["selection"]["quantized_module_count"], 839)
        self.assertEqual(index["selection"]["quantized_tensor_count"], 839)

        by_name = {tensor["name"]: tensor for tensor in index["tensors"]}
        tensor = by_name["transformer_blocks.0.attn.to_q.weight"]
        self.assertEqual(tensor["shape"], [3072, 3072])
        self.assertEqual(tensor["source_dtype"], "bf16")
        self.assertEqual(tensor["quant_dtype"], "fp8_e4m3")
        self.assertEqual(tensor["storage_dtype"], "uint8")
        self.assertEqual(tensor["scale"]["dtype"], "fp32")
        self.assertEqual(tensor["scale"]["shape"], [1])
        self.assertEqual(tensor["scale"]["granularity"], "per_tensor")
        self.assertIsNone(tensor["scale"]["axis"])
        self.assertEqual(tensor["payload"]["file"], "tensors/fp8_weights.safetensors")
        self.assertEqual(tensor["payload"]["tensor_name"], "transformer_blocks.0.attn.to_q.weight")
        self.assertEqual(tensor["payload"]["storage_dtype"], "uint8")
        self.assertEqual(tensor["metadata"]["module_name"], "transformer_blocks.0.attn.to_q")
        self.assertEqual(tensor["metadata"]["source_role"], "weight")
        self.assertNotIn("transformer_blocks.0.attn.to_q.bias", by_name)

    def test_qwen_image_edit_index_keeps_edit_path_high_precision(self):
        index = _build_index("configs/qwen_image_edit_2511_fp8_static.yaml")
        self.assertEqual(index["contract_schema"], "qwen_image_edit_static_contract.v1")
        self.assertEqual(index["reference_image_mode"], "index_timestep_zero")
        self.assertEqual(index["selection"]["quantized_tensor_count"], 839)
        names = {tensor["name"] for tensor in index["tensors"]}
        self.assertIn("transformer_blocks.0.img_mlp.net.0.proj.weight", names)
        self.assertIn("transformer_blocks.0.txt_mod.1.weight", names)
        self.assertIn("transformer_blocks.1.img_mod.1.weight", names)
        self.assertNotIn("transformer_blocks.0.img_mod.1.weight", names)
        self.assertNotIn("visual_semantic_path", names)
        self.assertNotIn("text_encoders.qwen25_7b", names)

    def test_qwen_image_e5m2_index_reuses_same_layer_policy(self):
        index = _build_index("configs/qwen_image_2512_fp8_e5m2_static.yaml")
        self.assertEqual(index["format"]["name"], "fp8_e5m2")
        self.assertEqual(index["selection"]["target_dtype"], "fp8_e5m2")
        self.assertEqual(index["selection"]["quantized_tensor_count"], 839)
        by_name = {tensor["name"]: tensor for tensor in index["tensors"]}
        tensor = by_name["transformer_blocks.0.attn.to_q.weight"]
        self.assertEqual(tensor["quant_dtype"], "fp8_e5m2")
        self.assertEqual(tensor["storage_dtype"], "uint8")
        self.assertEqual(tensor["scale"]["shape"], [1])

    def test_quantize_dry_run_writes_populated_tensor_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "job"
            rc = main([
                "quantize",
                "--config",
                "configs/qwen_image_2512_fp8_static.yaml",
                "--work-dir",
                str(work_dir),
                "--dry-run",
            ])
            self.assertEqual(rc, 0)
            index = json.loads((work_dir / "artifact" / "quant_tensor_index.json").read_text())
            manifest = json.loads((work_dir / "artifact" / "manifest.json").read_text())
            self.assertEqual(index["selection"]["quantized_tensor_count"], 839)
            self.assertEqual(index["tensor_payload_state"], "pending_export")
            self.assertEqual(index["payload_layout"]["weight_payload_path"], "tensors/fp8_weights.safetensors")
            self.assertEqual(index["payload_layout"]["scale_payload_path"], "scales/fp8_static_scales.safetensors")
            self.assertEqual(manifest["compatibility"]["target_level"], "L2")
            self.assertEqual(manifest["quantization"]["payload_layout"]["schema_version"], "artifact_payload_layout.v1")
            self.assertEqual(manifest["files"][0]["path"], "quant_tensor_index.json")


if __name__ == "__main__":
    unittest.main()
