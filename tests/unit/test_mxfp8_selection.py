import unittest

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter

# MXFP8 is the native-ComfyUI sibling of FP8, so it reuses the FP8 default policy
# (block-0 img_mod.1 excluded). Same 839-tensor set as the FP8 alignment test.
_FAMILIES = [
    ("qwen_image", "Qwen/Qwen-Image"),
    ("qwen_image_edit", "Qwen/Qwen-Image-Edit-2511"),
    ("qwen_image_layered", "Qwen/Qwen-Image-Layered"),
]


def _build_index(family, model_id):
    adapter = get_adapter(family)
    _inspection, graph = adapter.inspect(ModelSource(family=family, model_id=model_id))
    return build_quant_tensor_index(
        graph,
        adapter.default_policy("mxfp8"),
        TensorIndexOptions(
            algorithm="mxfp8",
            algorithm_version="0.1.0",
            target_dtype="mxfp8",
            scale_granularity="block",
            scale_axis="in_features",
            scale_method="amax",
            rounding="nearest_even",
            compatibility_level="L2",
            scale_block_size=32,
            scale_dtype="float8_e8m0fnu",
        ),
    )


class TestMxFp8Selection(unittest.TestCase):
    def test_index_builds_with_mxfp8_dtype(self):
        # Guards the dtype-registration coupling: target_dtype flows into
        # QuantTensorMetadata.__post_init__ -> get_dtype_spec, which KeyErrors
        # unless 'mxfp8' is in KNOWN_DTYPES.
        index = _build_index("qwen_image", "Qwen/Qwen-Image")
        self.assertEqual(index["format"]["name"], "mxfp8")
        self.assertEqual(index["format"]["storage_dtype"], "uint8")
        self.assertEqual(index["format"]["scale_granularity"], "block")
        for row in index["tensors"]:
            self.assertEqual(row["quant_dtype"], "mxfp8")
            self.assertEqual(row["storage_dtype"], "uint8")
            self.assertEqual(row["scale"]["granularity"], "block")
            self.assertEqual(row["scale"]["block_size"], 32)
            self.assertEqual(row["scale"]["dtype"], "float8_e8m0fnu")

    def test_block_scale_shape_is_logical_grid(self):
        index = _build_index("qwen_image", "Qwen/Qwen-Image")
        row = next(r for r in index["tensors"] if r["name"] == "transformer_blocks.0.attn.to_q.weight")
        out_f, in_f = row["shape"]
        self.assertEqual(in_f % 32, 0)
        # logical pre-swizzle E8M0 grid: [out, ceil(in/32)]
        self.assertEqual(row["scale"]["shape"], [out_f, in_f // 32])

    def test_selection_matches_fp8_native_set(self):
        for family, model_id in _FAMILIES:
            with self.subTest(family=family):
                index = _build_index(family, model_id)
                selected = {row["name"] for row in index["tensors"]}
                self.assertEqual(len(selected), 839)
                # FP8/native divergence from INT8-Fast: block-0 img_mod.1 is EXCLUDED.
                self.assertNotIn("transformer_blocks.0.img_mod.1.weight", selected)
                self.assertIn("transformer_blocks.1.img_mod.1.weight", selected)
                self.assertIn("transformer_blocks.0.txt_mod.1.weight", selected)
                self.assertIn("transformer_blocks.0.attn.to_q.weight", selected)
                self.assertIn("transformer_blocks.0.img_mlp.net.0.proj.weight", selected)
                for name in selected:
                    self.assertNotIn("norm_out", name)
                    self.assertNotIn("proj_out", name)
                    self.assertNotIn(".norm_q", name)


if __name__ == "__main__":
    unittest.main()
