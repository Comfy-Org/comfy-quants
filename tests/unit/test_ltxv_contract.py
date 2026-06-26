import unittest

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.ltxv import build_ltxv_static_contract
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


class TestLtxvContract(unittest.TestCase):
    def test_contract(self):
        c = build_ltxv_static_contract()
        self.assertEqual(c.family, "ltxv")
        self.assertEqual(c.schema_version, "ltxv_static_contract.v1")
        self.assertEqual(c.preferred_format, "fp8_e4m3")
        self.assertEqual([(g.prefix, g.count) for g in c.block_groups], [("model.diffusion_model.transformer_blocks", 28)])
        self.assertEqual(c.dimensions()["D"], 2048)
        self.assertEqual(c.dimensions()["FF"], 8192)

    def test_adapter_registered(self):
        self.assertEqual(get_adapter("ltxv").family, "ltxv")

    def test_graph_tensor_names_and_shapes(self):
        adapter = get_adapter("ltxv")
        _insp, graph = adapter.inspect(ModelSource(family="ltxv", model_id="Lightricks/LTX-Video"))
        by_name = _tensors_by_name(graph)
        self.assertEqual(by_name["model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"].shape, [2048, 2048])
        self.assertEqual(by_name["model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight"].shape, [2048, 2048])
        self.assertEqual(by_name["model.diffusion_model.transformer_blocks.0.attn2.to_k.weight"].shape, [2048, 2048])
        self.assertEqual(by_name["model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight"].shape, [8192, 2048])
        self.assertEqual(by_name["model.diffusion_model.transformer_blocks.0.ff.net.2.weight"].shape, [2048, 8192])
        self.assertEqual(by_name["model.diffusion_model.transformer_blocks.27.ff.net.2.weight"].shape, [2048, 8192])
        self.assertEqual(by_name["model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"].scale_axis, "out_features")
        for name, t in by_name.items():
            if name.startswith("model.diffusion_model.transformer_blocks.") and t.role == "weight":
                self.assertEqual(t.shape[1] % 32, 0, name)

    def test_fp8_selection(self):
        idx = _index("ltxv", "fp8_e4m3", "per_tensor", None)
        sel = {row["name"] for row in idx["tensors"]}
        # 28 blocks * 10 linears = 280
        self.assertEqual(len(sel), 280)
        self.assertIn("model.diffusion_model.transformer_blocks.0.attn1.to_q.weight", sel)
        self.assertIn("model.diffusion_model.transformer_blocks.0.attn2.to_v.weight", sel)
        self.assertIn("model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight", sel)
        for name in sel:
            self.assertNotIn("patchify_proj", name)
            self.assertNotIn("adaln_single", name)
            self.assertNotIn("caption_projection", name)
            self.assertNotIn("proj_out", name)
            self.assertNotIn("scale_shift_table", name)

    def test_block_formats_build(self):
        for target_dtype, block_size in [("mxfp8", 32), ("nvfp4", 16)]:
            with self.subTest(target_dtype=target_dtype):
                idx = _index("ltxv", target_dtype, "block", "in_features", block_size)
                self.assertEqual(idx["format"]["name"], target_dtype)
                self.assertEqual(len({row["name"] for row in idx["tensors"]}), 280)
                for row in idx["tensors"]:
                    self.assertEqual(row["scale"]["block_size"], block_size)


if __name__ == "__main__":
    unittest.main()
