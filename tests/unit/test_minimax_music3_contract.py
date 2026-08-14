import unittest

from comfy_quants.algorithms.tensor_index import (
    TensorIndexOptions,
    build_quant_tensor_index,
)
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.minimax_music3 import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    build_minimax_music3_dit_static_contract,
    build_minimax_music3_text_encoder_static_contract,
)
from comfy_quants.model_adapters.registry import get_adapter


def _tensors_by_name(graph):
    return {
        tensor.name: tensor for module in graph.modules for tensor in module.tensors
    }


def _nvfp4_index(family):
    adapter = get_adapter(family)
    _inspection, graph = adapter.inspect(
        ModelSource(family=family, model_id=MODEL_REPOSITORY, revision=MODEL_REVISION)
    )
    policy = adapter.default_policy("nvfp4")
    return build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm="nvfp4",
            algorithm_version="0.1.0",
            target_dtype="nvfp4",
            scale_granularity="block",
            scale_axis="in_features",
            scale_method="amax",
            rounding="nearest_even",
            compatibility_level="L2",
            scale_block_size=16,
            scale_dtype="float8_e4m3fn",
        ),
    )


class TestMiniMaxMusic3Contracts(unittest.TestCase):
    def test_adapters_are_registered(self):
        self.assertEqual(get_adapter("minimax_music3_dit").family, "minimax_music3_dit")
        self.assertEqual(
            get_adapter("minimax_music3_text_encoder").family,
            "minimax_music3_text_encoder",
        )

    def test_dit_contract_matches_published_comfyui_checkpoint(self):
        contract = build_minimax_music3_dit_static_contract()
        self.assertEqual(
            contract.schema_version, "minimax_music3_dit_static_contract.v1"
        )
        self.assertEqual(
            [(group.prefix, group.count) for group in contract.block_groups],
            [
                ("diffusion_transformer.transformer.layers", 36),
            ],
        )

        adapter = get_adapter("minimax_music3_dit")
        _inspection, graph = adapter.inspect(
            ModelSource(
                family=adapter.family,
                model_id=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
            )
        )
        tensors = _tensors_by_name(graph)
        self.assertEqual(
            tensors[
                "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight"
            ].shape,
            [6144, 2048],
        )
        self.assertEqual(
            tensors[
                "diffusion_transformer.transformer.layers.35.ff.ff.0.proj.weight"
            ].shape,
            [16384, 2048],
        )
        self.assertEqual(
            tensors["diffusion_transformer.transformer.layers.35.ff.ff.2.weight"].shape,
            [2048, 8192],
        )

    def test_text_encoder_contract_matches_published_comfyui_checkpoint(self):
        contract = build_minimax_music3_text_encoder_static_contract()
        self.assertEqual(
            contract.schema_version,
            "minimax_music3_text_encoder_static_contract.v1",
        )
        self.assertEqual(
            [(group.prefix, group.count) for group in contract.block_groups],
            [
                ("model.layers", 36),
                ("model.audio_decoder.layers", 4),
            ],
        )

        adapter = get_adapter("minimax_music3_text_encoder")
        _inspection, graph = adapter.inspect(
            ModelSource(
                family=adapter.family,
                model_id=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
            )
        )
        tensors = _tensors_by_name(graph)
        self.assertEqual(
            tensors["model.layers.0.self_attn.qkv_proj.weight"].shape, [6144, 4096]
        )
        self.assertEqual(
            tensors["model.layers.35.mlp.gate_up_proj.weight"].shape, [24576, 4096]
        )
        self.assertEqual(
            tensors["model.audio_decoder.layers.3.self_attn.qkv_proj.weight"].shape,
            [12288, 4096],
        )
        self.assertEqual(
            tensors["model.audio_decoder.layers.3.mlp.down_proj.weight"].shape,
            [4096, 6144],
        )

    def test_nvfp4_selection_is_block_aligned_and_keeps_sensitive_heads(self):
        expected_counts = {
            "minimax_music3_dit": 36 * 4,
            "minimax_music3_text_encoder": 36 * 4 + 4 * 4,
        }
        for family, expected_count in expected_counts.items():
            with self.subTest(family=family):
                index = _nvfp4_index(family)
                selected = {row["name"] for row in index["tensors"]}
                self.assertEqual(index["format"]["name"], "nvfp4")
                self.assertEqual(len(selected), expected_count)
                for row in index["tensors"]:
                    self.assertEqual(row["shape"][1] % 16, 0, row["name"])
                    self.assertEqual(row["scale"]["block_size"], 16)

        text_selected = {
            row["name"]
            for row in _nvfp4_index("minimax_music3_text_encoder")["tensors"]
        }
        self.assertNotIn("model.embed_tokens_prefill.weight", text_selected)
        self.assertNotIn("model.embed_tokens_audio.weight", text_selected)
        self.assertNotIn("model.lm_head_pruned.weight", text_selected)
        self.assertNotIn("model.audio_decoder.audio_heads.0.weight", text_selected)
        self.assertNotIn("model.audio_decoder.projection.weight", text_selected)


if __name__ == "__main__":
    unittest.main()
