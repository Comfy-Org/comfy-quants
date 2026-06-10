import unittest

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter

# Exclude list mirrors ComfyUI-INT8-Fast's "qwen" model_type (int8_unet_loader.py).
_EXCLUDE = ["*time_text_embed*", "*img_in*", "*norm_out*", "*proj_out*", "*txt_in*"]
_FAMILIES = [
    ("qwen_image", "Qwen/Qwen-Image"),
    ("qwen_image_edit", "Qwen/Qwen-Image-Edit-2511"),
    ("qwen_image_layered", "Qwen/Qwen-Image-Layered"),
]


def _build_index(family, model_id, exclude):
    adapter = get_adapter(family)
    _inspection, graph = adapter.inspect(ModelSource(family=family, model_id=model_id))
    policy = adapter.default_policy("int8_w8a8")
    policy.algorithm = "int8_w8a8"
    policy.include = []  # empty include => all quantizable modules pass the include gate
    policy.exclude = list(exclude)
    return build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm="int8_w8a8",
            algorithm_version="0.1.0",
            target_dtype="int8_w8a8",
            scale_granularity="per_channel",
            scale_axis="out_features",
            scale_method="amax",
            rounding="nearest_even",
            compatibility_level="L2",
        ),
    )


class TestInt8W8A8Selection(unittest.TestCase):
    def test_index_builds_with_int8_w8a8_dtype(self):
        # Guards the dtype-registration coupling: target_dtype flows into
        # QuantTensorMetadata.__post_init__ -> get_dtype_spec, which KeyErrors
        # unless 'int8_w8a8' is in KNOWN_DTYPES.
        index = _build_index("qwen_image", "Qwen/Qwen-Image", _EXCLUDE)
        self.assertEqual(index["format"]["name"], "int8_w8a8")
        self.assertEqual(index["format"]["storage_dtype"], "int8")
        for row in index["tensors"]:
            self.assertEqual(row["quant_dtype"], "int8_w8a8")
            self.assertEqual(row["storage_dtype"], "int8")

    def test_selection_matches_int8_fast_for_qwen_families(self):
        for family, model_id in _FAMILIES:
            with self.subTest(family=family):
                exclude = list(_EXCLUDE)
                if family == "qwen_image_edit":
                    exclude.append("*visual_semantic_path*")
                index = _build_index(family, model_id, exclude)
                selected = {row["name"] for row in index["tensors"]}

                # All 60 blocks x 14 quantizable Linear are selected (no block-0 special case).
                self.assertEqual(len(selected), 60 * 14)
                # The FP8/INT8 divergence: block-0 img_mod.1 IS selected here.
                self.assertIn("transformer_blocks.0.img_mod.1.weight", selected)
                self.assertIn("transformer_blocks.0.txt_mod.1.weight", selected)
                self.assertIn("transformer_blocks.0.attn.to_q.weight", selected)
                self.assertIn("transformer_blocks.0.attn.add_q_proj.weight", selected)
                self.assertIn("transformer_blocks.0.attn.to_out.0.weight", selected)
                self.assertIn("transformer_blocks.0.attn.to_add_out.weight", selected)
                self.assertIn("transformer_blocks.0.img_mlp.net.0.proj.weight", selected)
                self.assertIn("transformer_blocks.0.txt_mlp.net.2.weight", selected)

                # Excluded prefixes (and non-Linear norms) are never selected.
                for name in selected:
                    self.assertNotIn("time_text_embed", name)
                    self.assertNotIn("img_in", name)
                    self.assertNotIn("txt_in", name)
                    self.assertNotIn("norm_out", name)
                    self.assertNotIn("proj_out", name)
                    self.assertNotIn(".norm_q", name)
                    self.assertNotIn(".norm_k", name)


if __name__ == "__main__":
    unittest.main()
