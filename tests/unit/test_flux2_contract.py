import unittest

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.flux2 import build_flux2_static_contract
from comfy_quants.model_adapters.registry import get_adapter


def _tensors_by_name(graph):
    return {t.name: t for m in graph.modules for t in m.tensors}


def _index_with_policy(family, policy):
    adapter = get_adapter(family)
    _insp, graph = adapter.inspect(ModelSource(family=family, model_id="x"))
    return build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm=policy.algorithm,
            algorithm_version="0.1.0",
            target_dtype=policy.target_dtype,
            scale_granularity="per_tensor",
            scale_axis=None,
            scale_method="amax",
            rounding="nearest_even",
            compatibility_level="L2",
        ),
    )


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


class TestFlux2Contract(unittest.TestCase):
    def test_contract(self):
        c = build_flux2_static_contract()
        self.assertEqual(c.family, "flux2")
        self.assertEqual(c.schema_version, "flux2_static_contract.v1")
        self.assertEqual([(g.prefix, g.count) for g in c.block_groups], [("double_blocks", 8), ("single_blocks", 48)])
        self.assertEqual(c.dimensions()["H"], 6144)

    def test_adapter_registered(self):
        self.assertEqual(get_adapter("flux2").family, "flux2")

    def test_graph_tensor_names_and_shapes(self):
        adapter = get_adapter("flux2")
        _insp, graph = adapter.inspect(ModelSource(family="flux2", model_id="black-forest-labs/FLUX.2-dev"))
        by_name = _tensors_by_name(graph)
        # gated MLP + fused single linears (validated against the real flux2-dev header)
        self.assertEqual(by_name["double_blocks.0.img_attn.qkv.weight"].shape, [18432, 6144])
        self.assertEqual(by_name["double_blocks.0.img_attn.proj.weight"].shape, [6144, 6144])
        self.assertEqual(by_name["double_blocks.0.img_mlp.0.weight"].shape, [36864, 6144])
        self.assertEqual(by_name["double_blocks.0.img_mlp.2.weight"].shape, [6144, 18432])
        self.assertEqual(by_name["single_blocks.0.linear1.weight"].shape, [55296, 6144])
        self.assertEqual(by_name["single_blocks.0.linear2.weight"].shape, [6144, 24576])
        self.assertEqual(by_name["single_blocks.47.linear1.weight"].shape, [55296, 6144])
        for name, t in by_name.items():
            if (name.startswith("double_blocks.") or name.startswith("single_blocks.")) and t.role == "weight":
                self.assertEqual(t.shape[1] % 32, 0, name)

    def test_fp8_selection(self):
        idx = _index("flux2", "fp8_e4m3", "per_tensor", None)
        sel = {row["name"] for row in idx["tensors"]}
        # 8*8 + 48*2 = 160  (global modulation Linears are NOT per-block -> kept)
        self.assertEqual(len(sel), 160)
        self.assertIn("double_blocks.0.img_attn.qkv.weight", sel)
        self.assertIn("single_blocks.0.linear1.weight", sel)
        for name in sel:
            self.assertNotIn("_mod.lin", name)  # flux2 has no per-block modulation Linear
            self.assertNotIn("modulation", name)
            self.assertNotIn("img_in", name)
            self.assertNotIn("final_layer", name)

    def test_mixed_precision_policy_keeps_attention(self):
        # mixed=True reproduces official flux2_dev_fp8mixed: double-stream attention kept bf16,
        # MLP + fused single-stream linears quantized -> 8*4 + 48*2 = 128.
        idx = _index_with_policy("flux2", get_adapter("flux2").default_policy("fp8_e4m3", mixed=True))
        sel = {row["name"] for row in idx["tensors"]}
        self.assertEqual(len(sel), 128)
        for name in sel:
            self.assertNotIn("_attn.qkv", name)
            self.assertNotIn("_attn.proj", name)
        self.assertIn("double_blocks.0.img_mlp.0.weight", sel)
        self.assertIn("double_blocks.0.txt_mlp.2.weight", sel)
        self.assertIn("single_blocks.0.linear1.weight", sel)  # fused qkv+mlp still quantized
        # default (full) still quantizes attention
        full = {r["name"] for r in _index_with_policy("flux2", get_adapter("flux2").default_policy("fp8_e4m3"))["tensors"]}
        self.assertEqual(len(full), 160)
        self.assertIn("double_blocks.0.img_attn.qkv.weight", full)

    def test_block_formats_build(self):
        for target_dtype, block_size in [("mxfp8", 32), ("nvfp4", 16)]:
            with self.subTest(target_dtype=target_dtype):
                idx = _index("flux2", target_dtype, "block", "in_features", block_size)
                self.assertEqual(idx["format"]["name"], target_dtype)
                self.assertEqual(len({row["name"] for row in idx["tensors"]}), 160)
                for row in idx["tensors"]:
                    self.assertEqual(row["scale"]["block_size"], block_size)


if __name__ == "__main__":
    unittest.main()
