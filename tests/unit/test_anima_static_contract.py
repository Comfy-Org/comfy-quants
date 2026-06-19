import unittest

from comfy_quants.model_adapters.anima_contracts.anima import build_anima_static_contract, get_anima_static_contract
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter


def _tensors_by_name(graph):
    return {t.name: t for m in graph.modules for t in m.tensors}


class TestAnimaStaticContract(unittest.TestCase):
    def test_contract_2b(self):
        c = get_anima_static_contract(2048)
        self.assertEqual(c.family, "anima")
        self.assertEqual(c.schema_version, "anima_static_contract.v1")
        self.assertEqual(c.artifact_target, "comfyui")
        self.assertEqual(c.contract_mode, "static_adapter_contract")
        self.assertEqual(c.preferred_format, "fp8_e4m3")
        self.assertEqual(c.transformer.block_count, 28)
        self.assertEqual(c.transformer.num_heads, 16)
        self.assertEqual(c.transformer.dimensions()["X"], 2048)
        self.assertEqual(c.transformer.dimensions()["C"], 1024)

    def test_contract_14b(self):
        c = get_anima_static_contract(5120)
        self.assertEqual(c.family, "anima_14b")
        self.assertEqual(c.transformer.block_count, 36)
        self.assertEqual(c.transformer.num_heads, 40)
        self.assertEqual(c.transformer.dimensions()["X"], 5120)

    def test_invalid_model_channels(self):
        with self.assertRaises(ValueError):
            build_anima_static_contract(model_channels=4096)

    def test_adapters_registered(self):
        self.assertEqual(get_adapter("anima").family, "anima")
        self.assertEqual(get_adapter("anima_14b").family, "anima_14b")

    def test_graph_tensor_names_and_shapes_2b(self):
        adapter = get_adapter("anima")
        _insp, graph = adapter.inspect(ModelSource(family="anima", model_id="Comfy-Org/anima"))
        by_name = _tensors_by_name(graph)
        self.assertEqual(by_name["net.blocks.0.self_attn.q_proj.weight"].shape, [2048, 2048])
        self.assertEqual(by_name["net.blocks.0.self_attn.output_proj.weight"].shape, [2048, 2048])
        self.assertEqual(by_name["net.blocks.0.cross_attn.k_proj.weight"].shape, [2048, 1024])
        self.assertEqual(by_name["net.blocks.0.cross_attn.v_proj.weight"].shape, [2048, 1024])
        self.assertEqual(by_name["net.blocks.0.mlp.layer1.weight"].shape, [8192, 2048])
        self.assertEqual(by_name["net.blocks.0.mlp.layer2.weight"].shape, [2048, 8192])
        self.assertEqual(by_name["net.blocks.0.adaln_modulation_self_attn.1.weight"].shape, [256, 2048])
        self.assertEqual(by_name["net.blocks.0.adaln_modulation_self_attn.2.weight"].shape, [6144, 256])
        self.assertEqual(by_name["net.blocks.0.self_attn.q_proj.weight"].scale_axis, "out_features")
        # all in_features are multiples of 16 and 32 (nvfp4/mxfp8 block-aligned)
        for name, t in by_name.items():
            if name.startswith("net.blocks.") and t.role == "weight" and len(t.shape) == 2:
                self.assertEqual(t.shape[1] % 32, 0, name)

    def test_graph_shapes_14b(self):
        adapter = get_adapter("anima_14b")
        _insp, graph = adapter.inspect(ModelSource(family="anima_14b", model_id="x"))
        by_name = _tensors_by_name(graph)
        self.assertEqual(by_name["net.blocks.0.self_attn.q_proj.weight"].shape, [5120, 5120])
        self.assertEqual(by_name["net.blocks.0.cross_attn.k_proj.weight"].shape, [5120, 1024])
        self.assertEqual(by_name["net.blocks.35.mlp.layer1.weight"].shape, [20480, 5120])


if __name__ == "__main__":
    unittest.main()
