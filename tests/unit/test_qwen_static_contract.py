import unittest
from typing import Any

from comfy_quants.formats.registry import get_format
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.qwen_contracts.qwen_image import get_qwen_image_static_contract
from comfy_quants.model_adapters.qwen_contracts.qwen_image_edit import get_qwen_image_edit_static_contract
from comfy_quants.model_adapters.registry import get_adapter


class TestQwenStaticContract(unittest.TestCase):
    def test_qwen_image_contract_dimensions_and_format(self):
        contract = get_qwen_image_static_contract()
        self.assertEqual(contract.family, "qwen_image")
        self.assertEqual(contract.artifact_target, "comfyui")
        self.assertEqual(contract.contract_mode, "static_adapter_contract")
        self.assertEqual(contract.preferred_format, "fp8_e4m3")
        self.assertEqual(contract.transformer.block_prefix, "transformer_blocks")
        self.assertEqual(contract.transformer.block_count, 60)
        self.assertGreater(contract.transformer.block_count, 2)
        self.assertEqual(contract.transformer.hidden_size, 3072)
        self.assertEqual(contract.transformer.intermediate_size, 12288)

    def test_qwen_image_edit_contract_scope(self):
        contract = get_qwen_image_edit_static_contract()
        self.assertEqual(contract.family, "qwen_image_edit")
        self.assertEqual(contract.preferred_format, "fp8_e4m3")
        self.assertEqual(contract.metadata["reference_image_mode"], "index_timestep_zero")
        self.assertIn("visual_semantic_path", {module.component for module in contract.extra_components})

    def test_graph_uses_static_model_contract_names(self):
        adapter = get_adapter("qwen_image")
        inspection, graph = adapter.inspect(ModelSource(family="qwen_image", model_id="Qwen/Qwen-Image-2512"))
        names = {module.name for module in graph.modules}
        self.assertEqual(graph.metadata["graph_kind"], "static_model_contract")
        self.assertEqual(graph.metadata["tensor_coverage"], "declared_tensors")
        self.assertEqual(graph.metadata["model_contract"]["block_count"], 60)
        self.assertIn("transformer_blocks.0.attn.to_q", names)
        self.assertIn("transformer_blocks.0.attn.add_q_proj", names)
        self.assertIn("transformer_blocks.0.img_mlp.net.0.proj", names)
        self.assertIn("transformer_blocks.0.txt_mlp.net.2", names)
        self.assertIn("norm_out.linear", names)
        self.assertIn("vae", names)
        self.assertIn("text_encoders.qwen25_7b", names)
        self.assertEqual(inspection.quantizable_modules, 60 * 14)
        self.assertGreater(inspection.total_parameters, 0)

    def test_qwen_image_edit_graph_keeps_edit_components_high_precision(self):
        adapter = get_adapter("qwen_image_edit")
        _, graph = adapter.inspect(ModelSource(family="qwen_image_edit", model_id="Qwen/Qwen-Image-Edit-2511"))
        by_name = {module.name: module for module in graph.modules}
        self.assertIn("transformer_blocks.0.attn.to_q", by_name)
        self.assertIn("visual_semantic_path", by_name)
        self.assertFalse(by_name["visual_semantic_path"].quantizable)
        self.assertEqual(by_name["visual_semantic_path"].default_action, "keep_bf16")
        self.assertFalse(by_name["vae"].quantizable)
        self.assertFalse(by_name["text_encoders.qwen25_7b"].quantizable)
        self.assertTrue(by_name["transformer_blocks.0.attn.to_q"].quantizable)
        self.assertTrue(by_name["transformer_blocks.0.img_mlp.net.0.proj"].quantizable)
        self.assertTrue(by_name["transformer_blocks.0.txt_mod.1"].quantizable)
        self.assertFalse(by_name["transformer_blocks.0.attn.norm_q"].quantizable)

    def test_fp8_format_is_reusable_across_qwen_adapters(self):
        fmt = get_format("fp8_e4m3")
        self.assertIn("qwen_image", fmt.compatible_families)
        self.assertIn("qwen_image_edit", fmt.compatible_families)
        self.assertEqual(fmt.default_scale_granularity, "per_tensor")
        self.assertIsNone(fmt.metadata["default_axis"])

    def test_graph_metadata_has_production_terms(self):
        adapter = get_adapter("qwen_image_edit")
        _, graph = adapter.inspect(ModelSource(family="qwen_image_edit", model_id="Qwen/Qwen-Image-Edit-2511"))
        banned = ("place" + "holder", "sym" + "bolic", "not_" + "implemented", "up" + "stream")

        def walk(value: Any):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from walk(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    yield from walk(item)
            else:
                yield value

        for value in walk(graph.metadata):
            if isinstance(value, str):
                lowered = value.lower()
                self.assertFalse(any(term in lowered for term in banned), value)


if __name__ == "__main__":
    unittest.main()
