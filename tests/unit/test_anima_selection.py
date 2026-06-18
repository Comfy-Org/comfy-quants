import unittest

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter


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


class TestAnimaSelection(unittest.TestCase):
    def test_fp8_selection_2b(self):
        idx = _index("anima", "fp8_e4m3", "per_tensor", None)
        sel = {row["name"] for row in idx["tensors"]}
        # (num_blocks-2)*16 + 10  =  26*16 + 10  =  426
        self.assertEqual(len(sel), 426)
        # blocks 2+ fully quantized
        self.assertIn("net.blocks.2.self_attn.q_proj.weight", sel)
        self.assertIn("net.blocks.2.cross_attn.k_proj.weight", sel)
        self.assertIn("net.blocks.2.mlp.layer1.weight", sel)
        self.assertIn("net.blocks.2.adaln_modulation_self_attn.1.weight", sel)
        # block 1: attn/mlp quantized, adaln kept (convert_to_quant preset)
        self.assertIn("net.blocks.1.self_attn.q_proj.weight", sel)
        self.assertNotIn("net.blocks.1.adaln_modulation_self_attn.1.weight", sel)
        # block 0 fully kept
        self.assertNotIn("net.blocks.0.self_attn.q_proj.weight", sel)
        # embedders / final / llm_adapter never selected
        for name in sel:
            self.assertFalse(name.startswith("net.llm_adapter"), name)
            self.assertNotIn("final_layer", name)
            self.assertNotIn("t_embedder", name)
            self.assertNotIn("x_embedder", name)
            self.assertNotIn("q_norm", name)
            self.assertNotIn("k_norm", name)

    def test_fp8_selection_14b(self):
        idx = _index("anima_14b", "fp8_e4m3", "per_tensor", None)
        # 34*16 + 10 = 554
        self.assertEqual(len({row["name"] for row in idx["tensors"]}), 554)

    def test_block_formats_build_2b(self):
        for target_dtype, block_size in [("mxfp8", 32), ("nvfp4", 16)]:
            with self.subTest(target_dtype=target_dtype):
                idx = _index("anima", target_dtype, "block", "in_features", block_size)
                self.assertEqual(idx["format"]["name"], target_dtype)
                self.assertEqual(len({row["name"] for row in idx["tensors"]}), 426)
                for row in idx["tensors"]:
                    self.assertEqual(row["quant_dtype"], target_dtype)
                    self.assertEqual(row["storage_dtype"], "uint8")
                    self.assertEqual(row["scale"]["granularity"], "block")
                    self.assertEqual(row["scale"]["block_size"], block_size)


if __name__ == "__main__":
    unittest.main()
