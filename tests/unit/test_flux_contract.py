import unittest

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.flux import build_flux_static_contract
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


class TestFluxContract(unittest.TestCase):
    def test_contract(self):
        c = build_flux_static_contract()
        self.assertEqual(c.family, "flux")
        self.assertEqual(c.schema_version, "flux_static_contract.v1")
        self.assertEqual(c.artifact_target, "comfyui")
        self.assertEqual(c.contract_mode, "static_adapter_contract")
        self.assertEqual(c.preferred_format, "fp8_e4m3")
        # two block groups: 19 double + 38 single
        self.assertEqual([(g.prefix, g.count) for g in c.block_groups], [("double_blocks", 19), ("single_blocks", 38)])
        self.assertEqual(c.block_count, 57)
        self.assertEqual(c.dimensions()["H"], 3072)

    def test_adapter_registered(self):
        self.assertEqual(get_adapter("flux").family, "flux")

    def test_graph_tensor_names_and_shapes(self):
        adapter = get_adapter("flux")
        _insp, graph = adapter.inspect(ModelSource(family="flux", model_id="black-forest-labs/FLUX.1-dev"))
        by_name = _tensors_by_name(graph)
        self.assertEqual(by_name["double_blocks.0.img_attn.qkv.weight"].shape, [9216, 3072])
        self.assertEqual(by_name["double_blocks.0.img_attn.proj.weight"].shape, [3072, 3072])
        self.assertEqual(by_name["double_blocks.0.img_mlp.0.weight"].shape, [12288, 3072])
        self.assertEqual(by_name["double_blocks.0.img_mlp.2.weight"].shape, [3072, 12288])
        self.assertEqual(by_name["double_blocks.0.txt_mod.lin.weight"].shape, [18432, 3072])
        self.assertEqual(by_name["single_blocks.0.linear1.weight"].shape, [21504, 3072])
        self.assertEqual(by_name["single_blocks.0.linear2.weight"].shape, [3072, 15360])
        self.assertEqual(by_name["single_blocks.0.modulation.lin.weight"].shape, [9216, 3072])
        self.assertEqual(by_name["single_blocks.37.linear1.weight"].shape, [21504, 3072])
        self.assertEqual(by_name["double_blocks.0.img_attn.qkv.weight"].scale_axis, "out_features")
        # block-aligned: every selected weight's in_features is a multiple of 32 (mxfp8) and 16 (nvfp4)
        for name, t in by_name.items():
            if (name.startswith("double_blocks.") or name.startswith("single_blocks.")) and t.role == "weight":
                self.assertEqual(t.shape[1] % 32, 0, name)

    def test_fp8_selection(self):
        idx = _index("flux", "fp8_e4m3", "per_tensor", None)
        sel = {row["name"] for row in idx["tensors"]}
        # 19*10 + 38*3 = 304
        self.assertEqual(len(sel), 304)
        self.assertIn("double_blocks.0.img_attn.qkv.weight", sel)
        self.assertIn("single_blocks.0.linear1.weight", sel)
        # embedders / final layer never selected
        for name in sel:
            self.assertNotIn("img_in", name)
            self.assertNotIn("txt_in", name)
            self.assertNotIn("time_in", name)
            self.assertNotIn("final_layer", name)

    def test_block_formats_build(self):
        for target_dtype, block_size in [("mxfp8", 32), ("nvfp4", 16)]:
            with self.subTest(target_dtype=target_dtype):
                idx = _index("flux", target_dtype, "block", "in_features", block_size)
                self.assertEqual(idx["format"]["name"], target_dtype)
                self.assertEqual(len({row["name"] for row in idx["tensors"]}), 304)
                for row in idx["tensors"]:
                    self.assertEqual(row["quant_dtype"], target_dtype)
                    self.assertEqual(row["scale"]["granularity"], "block")
                    self.assertEqual(row["scale"]["block_size"], block_size)


if __name__ == "__main__":
    unittest.main()
