import unittest

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.comfy.artifact_contracts import get_artifact_contract_index
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter, list_adapters
from comfy_quants.registry.global_registry import registry


class TestComfyAlignment(unittest.TestCase):
    def test_registry_has_builtin_components(self):
        self.assertIn("qwen_image", list_adapters())
        self.assertIn("qwen_image_edit", list_adapters())
        self.assertIn("qwen_image_layered", list_adapters())
        self.assertIn("fp8_static", registry.list_algorithms())
        self.assertIn("int8_w8a8", registry.list_algorithms())
        self.assertIn("mxfp8", registry.list_algorithms())
        self.assertIn("nvfp4", registry.list_algorithms())
        self.assertIn("fp8_e4m3", registry.list_formats())
        self.assertIn("fp8_e5m2", registry.list_formats())
        self.assertIn("int8_w8a8", registry.list_formats())
        self.assertIn("mxfp8", registry.list_formats())
        self.assertIn("nvfp4", registry.list_formats())
        self.assertIn("torch_ref", registry.list_backends())

    def test_qwen_adapter_reports_static_artifact_contract(self):
        adapter = get_adapter("qwen_image_edit")
        inspection, graph = adapter.inspect(ModelSource(family="qwen_image_edit", model_id="Qwen/Qwen-Image-Edit-2511"))
        self.assertEqual(graph.metadata["artifact_target"], "comfyui")
        self.assertEqual(graph.metadata["contract_source"], "comfy_quants")
        self.assertEqual(graph.metadata["contract_mode"], "static_adapter_contract")
        self.assertIn("artifact_contract", inspection.metadata)
        self.assertEqual(inspection.metadata["artifact_contract"]["artifact_target"], "comfyui")
        self.assertEqual(
            inspection.metadata["artifact_contract"]["schema_version"],
            "qwen_image_edit_contract.v1",
        )

    def test_comfyui_contract_index_is_static_artifact_metadata(self):
        contracts = get_artifact_contract_index().to_dict()
        self.assertEqual(contracts["schema_version"], "artifact_contract_index.v1")
        self.assertEqual(contracts["artifact_target"], "comfyui")
        self.assertEqual(contracts["contract_source"], "comfy_quants")
        self.assertEqual(contracts["contract_mode"], "static_adapter_contract")
        self.assertIn("qwen_image", contracts["contracts"])
        self.assertIn("qwen_image_edit", contracts["contracts"])
        self.assertIn("qwen_image_layered", contracts["contracts"])
        self.assertEqual(contracts["contracts"]["qwen_image"]["artifact_target"], "comfyui")
        self.assertEqual(contracts["contracts"]["qwen_image_edit"]["artifact_target"], "comfyui")
        self.assertEqual(contracts["contracts"]["qwen_image_layered"]["artifact_target"], "comfyui")
        self.assertEqual(
            set(contracts),
            {
                "schema_version",
                "artifact_target",
                "contract_source",
                "contract_mode",
                "contracts",
            },
        )

    def test_qwen_default_fp8_policy_matches_mixed_checkpoint_layer_set(self):
        for family, model_id in [
            ("qwen_image", "Qwen/Qwen-Image"),
            ("qwen_image_edit", "Qwen/Qwen-Image-Edit-2511"),
            ("qwen_image_layered", "Qwen/Qwen-Image-Layered"),
        ]:
            for target_dtype in ["fp8_e4m3", "fp8_e5m2"]:
                with self.subTest(family=family, target_dtype=target_dtype):
                    adapter = get_adapter(family)
                    _inspection, graph = adapter.inspect(ModelSource(family=family, model_id=model_id))
                    index = build_quant_tensor_index(
                        graph,
                        adapter.default_policy(target_dtype),
                        TensorIndexOptions(
                            algorithm="fp8_static",
                            algorithm_version="0.1.0",
                            target_dtype=target_dtype,
                            scale_granularity="per_tensor",
                            scale_axis=None,
                            scale_method="amax",
                            rounding="nearest_even",
                            compatibility_level="L2",
                        ),
                    )
                    selected = {row["name"] for row in index["tensors"]}
                    self.assertEqual(index["format"]["name"], target_dtype)
                    self.assertEqual(len(selected), 839)
                    self.assertIn("transformer_blocks.0.txt_mod.1.weight", selected)
                    self.assertIn("transformer_blocks.1.img_mod.1.weight", selected)
                    self.assertNotIn("transformer_blocks.0.img_mod.1.weight", selected)
                    self.assertIn("transformer_blocks.0.attn.to_q.weight", selected)
                    self.assertIn("transformer_blocks.0.img_mlp.net.0.proj.weight", selected)


if __name__ == "__main__":
    unittest.main()
