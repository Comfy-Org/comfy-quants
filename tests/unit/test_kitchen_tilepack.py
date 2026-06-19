import json
import unittest


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


class TestKitchenTilePack(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required for kitchen tile-pack tests")

    def _natural_params(self, n: int = 128, k: int = 128, rank: int = 16):
        torch = self.torch
        from comfy_quants.formats.int4_common import encode_quant_config_tensor, pack_signed_int4_pairs

        dense = (torch.arange(n * k, dtype=torch.int16).remainder(16) - 8).view(n, k).to(torch.int8)
        return {
            "weight": pack_signed_int4_pairs(dense),
            "weight_scale": torch.arange((k // 64) * n, dtype=torch.float32).view(k // 64, n).to(torch.float16),
            "smooth_factor": torch.arange(k, dtype=torch.float32).to(torch.float16),
            "proj_down": torch.arange(k * rank, dtype=torch.float32).view(k, rank).to(torch.float16),
            "proj_up": torch.arange(n * rank, dtype=torch.float32).view(n, rank).to(torch.float16),
            "bias": torch.arange(n, dtype=torch.float32).to(torch.float16),
            "comfy_quant": encode_quant_config_tensor({"format": "svdquant_w4a4", "act_unsigned": True}),
        }

    def test_signed_int4_pair_roundtrip(self):
        torch = self.torch
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs, unpack_signed_int4_pairs

        values = torch.tensor([[-8, -7, -1, 0, 1, 7]], dtype=torch.int8)
        packed = pack_signed_int4_pairs(values)
        restored = unpack_signed_int4_pairs(packed)

        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(list(packed.shape), [1, 3])
        self.assertTrue(torch.equal(restored, values))

    def test_svdquant_params_pack_to_kitchen_tile_layout(self):
        torch = self.torch
        from comfy_quants.formats.int4_common import decode_quant_config_tensor
        from comfy_quants.formats.kitchen_tilepack import (
            KITCHEN_TILEPACK_LAYOUT_NAME,
            to_kitchen_tile_packed_params,
            unpack_n_axis,
            unpack_weight_scale,
            unpack_weight_tile,
        )

        params = self._natural_params()
        packed = to_kitchen_tile_packed_params(params)

        self.assertEqual(tuple(packed["weight"].shape), (1, 2, 32, 128))
        self.assertEqual(tuple(packed["weight_scale"].shape), (1, 2, 128))
        self.assertEqual(packed["weight_scale"].dtype, torch.float16)
        self.assertEqual(tuple(packed["proj_up"].shape), (1, 16, 128))
        self.assertEqual(tuple(packed["bias"].shape), (128,))
        self.assertTrue(torch.equal(unpack_weight_tile(packed["weight"]), params["weight"]))
        self.assertTrue(torch.equal(unpack_weight_scale(packed["weight_scale"]), params["weight_scale"]))
        self.assertTrue(torch.equal(unpack_n_axis(packed["proj_up"]), params["proj_up"]))

        comfy_quant = decode_quant_config_tensor(packed["comfy_quant"])
        self.assertEqual(comfy_quant["format"], "svdquant_w4a4")
        self.assertEqual(comfy_quant["layout"], KITCHEN_TILEPACK_LAYOUT_NAME)
        self.assertIs(comfy_quant["act_unsigned"], True)

    def test_svdquant_fp32_weight_scale_is_stored_as_bfloat16(self):
        torch = self.torch
        from comfy_quants.formats.kitchen_tilepack import to_kitchen_tile_packed_params, unpack_weight_scale

        params = self._natural_params()
        params["weight_scale"] = params["weight_scale"].to(torch.float32)

        packed = to_kitchen_tile_packed_params(params)

        self.assertEqual(tuple(packed["weight_scale"].shape), (1, 2, 128))
        self.assertEqual(packed["weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(unpack_weight_scale(packed["weight_scale"]).dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(unpack_weight_scale(packed["weight_scale"]).float(), params["weight_scale"], atol=1e-2, rtol=1e-2))

    def test_state_dict_repack_finds_svdquant_prefix(self):
        from comfy_quants.formats.kitchen_tilepack import repack_svdquant_state_dict

        params = self._natural_params(rank=8)
        state_dict = {f"layer.{key}": value for key, value in params.items()}

        prefixes = repack_svdquant_state_dict(state_dict)

        self.assertEqual(prefixes, ["layer"])
        self.assertEqual(tuple(state_dict["layer.weight"].shape), (1, 2, 32, 128))
        self.assertEqual(tuple(state_dict["layer.weight_scale"].shape), (1, 2, 128))
        self.assertIn(state_dict["layer.weight_scale"].dtype, {self.torch.float16, self.torch.bfloat16})
        self.assertEqual(tuple(state_dict["layer.proj_up"].shape), (1, 8, 128))

    def test_registered_int4_formats_are_model_agnostic_specs(self):
        from comfy_quants.formats.registry import get_format

        svdquant = get_format("svdquant_w4a4")
        awq = get_format("awq_w4a16")

        self.assertEqual(svdquant.metadata["layout"], "kitchen_tile_packed_w4a4")
        self.assertEqual(svdquant.metadata["group_size"], 64)
        self.assertEqual(awq.metadata["group_size"], 64)
        self.assertIn("qwen_image_edit", svdquant.compatible_families)
        self.assertIn("qwen_image_edit", awq.compatible_families)

    def test_quant_config_tensor_uses_json_payload(self):
        from comfy_quants.formats.int4_common import encode_quant_config_tensor

        tensor = encode_quant_config_tensor({"format": "svdquant_w4a4", "layout": "kitchen_tile_packed_w4a4"})
        decoded = json.loads(bytes(tensor.tolist()).decode("utf-8"))

        self.assertEqual(decoded["format"], "svdquant_w4a4")
        self.assertEqual(decoded["layout"], "kitchen_tile_packed_w4a4")

    def test_svdquant_quant_config_defaults_to_raw_lowrank_basis(self):
        from comfy_quants.formats.svdquant_w4a4 import svdquant_w4a4_checkpoint_quant_config

        config = svdquant_w4a4_checkpoint_quant_config()

        self.assertEqual(config["format"], "svdquant_w4a4")
        self.assertEqual(config["layout"], "kitchen_tile_packed_w4a4")
        self.assertEqual(config["lowrank_branch_input_basis"], "raw")
        self.assertIs(config["proj_down_smooth_folded"], True)

    def test_svdquant_quant_config_allows_explicit_post_smoothing_basis(self):
        from comfy_quants.formats.svdquant_w4a4 import svdquant_w4a4_checkpoint_quant_config

        config = svdquant_w4a4_checkpoint_quant_config(
            lowrank_branch_input_basis="post_smoothing",
            proj_down_smooth_folded=False,
        )

        self.assertEqual(config["lowrank_branch_input_basis"], "post_smoothing")
        self.assertIs(config["proj_down_smooth_folded"], False)

    def test_svdquant_quant_config_allows_explicit_raw_folded_basis(self):
        from comfy_quants.formats.svdquant_w4a4 import svdquant_w4a4_checkpoint_quant_config

        config = svdquant_w4a4_checkpoint_quant_config(
            act_unsigned=True,
            lowrank_branch_input_basis="raw",
            proj_down_smooth_folded=True,
        )

        self.assertEqual(config["lowrank_branch_input_basis"], "raw")
        self.assertIs(config["proj_down_smooth_folded"], True)
        self.assertIs(config["act_unsigned"], True)

    def test_svdquant_quant_config_rejects_unknown_lowrank_basis(self):
        from comfy_quants.formats.svdquant_w4a4 import svdquant_w4a4_checkpoint_quant_config

        with self.assertRaisesRegex(ValueError, "unsupported low-rank branch input basis"):
            svdquant_w4a4_checkpoint_quant_config(lowrank_branch_input_basis="runtime_guess")


if __name__ == "__main__":
    unittest.main()
