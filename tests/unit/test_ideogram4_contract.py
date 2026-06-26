import unittest

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.ideogram4 import build_ideogram4_static_contract
from comfy_quants.model_adapters.registry import get_adapter


def _tensors_by_name(graph):
    return {t.name: t for m in graph.modules for t in m.tensors}


def _index(family, target_dtype, granularity, axis, block_size=None):
    adapter = get_adapter(family)
    _insp, graph = adapter.inspect(ModelSource(family=family, model_id="x"))
    policy = adapter.default_policy(target_dtype)
    return build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm=policy.algorithm,
            algorithm_version="0.1.0",
            target_dtype=target_dtype,
            scale_granularity=granularity,
            scale_axis=axis,
            scale_method="amax",
            rounding="nearest_even",
            compatibility_level="L2",
            scale_block_size=block_size,
            scale_dtype="float8_e4m3fn" if granularity == "block" else "fp32",
        ),
    )


class TestIdeogram4Contract(unittest.TestCase):
    def test_contract(self):
        c = build_ideogram4_static_contract()
        self.assertEqual(c.family, "ideogram4")
        self.assertEqual(c.schema_version, "ideogram4_static_contract.v1")
        self.assertEqual(c.preferred_format, "fp8_e4m3")
        self.assertEqual([(g.prefix, g.count) for g in c.block_groups], [("layers", 34)])
        self.assertEqual(c.dimensions()["E"], 4608)
        self.assertEqual(c.dimensions()["INTER"], 12288)
        self.assertEqual(c.dimensions()["ADALN"], 512)

    def test_adapter_registered(self):
        self.assertEqual(get_adapter("ideogram4").family, "ideogram4")

    def test_graph_tensor_names_and_shapes(self):
        adapter = get_adapter("ideogram4")
        _insp, graph = adapter.inspect(ModelSource(family="ideogram4", model_id="Comfy-Org/Ideogram-4"))
        by_name = _tensors_by_name(graph)
        self.assertEqual(by_name["layers.0.attention.qkv.weight"].shape, [13824, 4608])
        self.assertEqual(by_name["layers.0.attention.o.weight"].shape, [4608, 4608])
        self.assertEqual(by_name["layers.0.feed_forward.w1.weight"].shape, [12288, 4608])
        self.assertEqual(by_name["layers.0.feed_forward.w2.weight"].shape, [4608, 12288])
        self.assertEqual(by_name["layers.0.feed_forward.w3.weight"].shape, [12288, 4608])
        self.assertEqual(by_name["layers.0.adaln_modulation.weight"].shape, [18432, 512])
        self.assertEqual(by_name["layers.33.attention.qkv.weight"].shape, [13824, 4608])
        self.assertEqual(by_name["layers.0.attention.qkv.weight"].scale_axis, "out_features")
        for name, t in by_name.items():
            if name.startswith("layers.") and t.role == "weight":
                self.assertEqual(t.shape[1] % 32, 0, name)
                self.assertEqual(t.shape[1] % 16, 0, name)

    def test_fp8_selection(self):
        idx = _index("ideogram4", "fp8_e4m3", "per_tensor", None)
        sel = {row["name"] for row in idx["tensors"]}
        # 34 layers * 6 linears = 204
        self.assertEqual(len(sel), 204)
        self.assertIn("layers.0.attention.qkv.weight", sel)
        self.assertIn("layers.0.feed_forward.w1.weight", sel)
        self.assertIn("layers.0.adaln_modulation.weight", sel)
        for name in sel:
            self.assertNotIn("input_proj", name)
            self.assertNotIn("llm_cond", name)
            self.assertNotIn("t_embedding", name)
            self.assertNotIn("adaln_proj", name)
            self.assertNotIn("embed_image_indicator", name)
            self.assertNotIn("final_layer", name)

    def test_block_formats_build(self):
        for target_dtype, block_size in [("mxfp8", 32), ("nvfp4", 16)]:
            with self.subTest(target_dtype=target_dtype):
                idx = _index("ideogram4", target_dtype, "block", "in_features", block_size)
                self.assertEqual(idx["format"]["name"], target_dtype)
                self.assertEqual(len({row["name"] for row in idx["tensors"]}), 204)
                for row in idx["tensors"]:
                    self.assertEqual(row["scale"]["block_size"], block_size)


if __name__ == "__main__":
    unittest.main()
