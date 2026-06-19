import unittest


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


class TestInt4SvdquantRuntimeReference(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required")

    def test_signed_activation_w4_uses_absmax_over_group(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import quantize_activation_w4_signed
        from comfy_quants.formats.int4_common import unpack_signed_int4_pairs

        inputs = torch.tensor([[-7.0, -3.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        result = quantize_activation_w4_signed(inputs, group_size=4)

        expected_q = torch.tensor([[-7, -3, 0, 7, 0, 0, 0, 0]], dtype=torch.int8)
        self.assertTrue(torch.equal(result.q_values, expected_q))
        self.assertTrue(torch.allclose(result.scale, torch.tensor([[1.0, 1.0]], dtype=torch.float32)))
        self.assertTrue(torch.equal(unpack_signed_int4_pairs(result.packed), expected_q))
        self.assertTrue(torch.allclose(result.dequantized, inputs))
        self.assertEqual(result.signedness, "signed")

    def test_unsigned_activation_w4_saturates_negative_values(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import quantize_activation_w4_unsigned
        from comfy_quants.formats.int4_common import unpack_uint4_pairs

        inputs = torch.tensor([[0.0, 5.0, 15.0, 10.0, -1.0, 0.0, 1.2, 2.0]], dtype=torch.float32)

        result = quantize_activation_w4_unsigned(inputs, group_size=4)

        expected_q = torch.tensor([[0, 5, 15, 10, 0, 0, 9, 15]], dtype=torch.int8)
        expected_scale = torch.tensor([[1.0, 2.0 / 15.0]], dtype=torch.float32)
        expected_dequant = torch.tensor([[0.0, 5.0, 15.0, 10.0, 0.0, 0.0, 1.2, 2.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(result.q_values, expected_q))
        self.assertTrue(torch.allclose(result.scale, expected_scale))
        self.assertTrue(torch.equal(unpack_uint4_pairs(result.packed), expected_q))
        self.assertTrue(torch.allclose(result.dequantized, expected_dequant))
        self.assertEqual(result.signedness, "unsigned")

    def test_runtime_linear_no_branch_matches_manual_activation_w4_path(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import (
            quantize_activation_w4_signed,
            reference_svdquant_w4a4_linear_runtime,
        )
        from comfy_quants.algorithms.int4_svdquant.weight_quant import dequantize_natural_svdquant_weight
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs

        n, k = 8, 64
        codes = torch.arange(n * k, dtype=torch.int32).reshape(n, k).remainder(9).sub(4).to(torch.int8)
        packed_weight = pack_signed_int4_pairs(codes)
        weight_scale = torch.linspace(0.05, 0.12, n, dtype=torch.float32).reshape(1, n)
        smooth_factor = torch.linspace(0.75, 1.5, k, dtype=torch.float32)
        inputs = torch.randn((3, k), generator=torch.Generator().manual_seed(1101), dtype=torch.float32)
        bias = torch.linspace(-0.2, 0.2, n, dtype=torch.float32)

        activation_quant = quantize_activation_w4_signed(inputs / smooth_factor.reshape(1, k), group_size=64)
        dense_weight = dequantize_natural_svdquant_weight(packed_weight, weight_scale, group_size=64)
        expected = torch.matmul(activation_quant.dequantized, dense_weight.t()) + bias.reshape(1, n)

        actual = reference_svdquant_w4a4_linear_runtime(
            inputs,
            packed_weight,
            weight_scale,
            smooth_factor,
            bias=bias,
            group_size=64,
            activation_signedness="signed",
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_unsigned_runtime_applies_gelu_shift_to_main_path_but_not_raw_branch(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import (
            GELU_UNSIGNED_SHIFT,
            quantize_activation_w4_unsigned,
            reference_svdquant_w4a4_linear_runtime,
        )
        from comfy_quants.algorithms.int4_svdquant.weight_quant import dequantize_natural_svdquant_weight
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs

        n, k, rank = 8, 64, 2
        codes = torch.arange(n * k, dtype=torch.int32).reshape(n, k).remainder(15).sub(7).to(torch.int8)
        packed_weight = pack_signed_int4_pairs(codes)
        weight_scale = torch.linspace(0.04, 0.13, n, dtype=torch.float32).reshape(1, n)
        smooth_factor = torch.linspace(0.65, 1.35, k, dtype=torch.float32)
        inputs = torch.linspace(-1.5, 1.0, 2 * k, dtype=torch.float32).reshape(2, k)
        proj_down = torch.randn((k, rank), generator=torch.Generator().manual_seed(1401), dtype=torch.float32) * 0.04
        proj_up = torch.randn((n, rank), generator=torch.Generator().manual_seed(1402), dtype=torch.float32) * 0.05
        bias = torch.linspace(-0.03, 0.04, n, dtype=torch.float32)

        main_inputs = inputs + GELU_UNSIGNED_SHIFT
        activation_quant = quantize_activation_w4_unsigned(main_inputs / smooth_factor.reshape(1, k), group_size=64)
        dense_weight = dequantize_natural_svdquant_weight(packed_weight, weight_scale, group_size=64)
        lowrank = torch.matmul(torch.matmul(inputs, proj_down), proj_up.t())
        expected = torch.matmul(activation_quant.dequantized, dense_weight.t()) + lowrank + bias.reshape(1, n)

        actual = reference_svdquant_w4a4_linear_runtime(
            inputs,
            packed_weight,
            weight_scale,
            smooth_factor,
            proj_down,
            proj_up,
            bias=bias,
            group_size=64,
            activation_signedness="unsigned",
            branch_input_basis="raw",
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

        old_unshifted_behavior = reference_svdquant_w4a4_linear_runtime(
            inputs,
            packed_weight,
            weight_scale,
            smooth_factor,
            proj_down,
            proj_up,
            bias=bias,
            group_size=64,
            activation_signedness="unsigned",
            branch_input_basis="raw",
            apply_unsigned_activation_shift=False,
        )
        self.assertGreater(float((actual - old_unshifted_behavior).abs().max().item()), 1e-4)

    def test_raw_branch_with_smooth_folded_down_matches_post_smoothing_branch(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import reference_svdquant_w4a4_linear_runtime
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs

        n, k, rank = 8, 64, 3
        packed_zero_weight = pack_signed_int4_pairs(torch.zeros((n, k), dtype=torch.int8))
        weight_scale = torch.ones((1, n), dtype=torch.float32)
        smooth_factor = torch.linspace(0.8, 1.6, k, dtype=torch.float32)
        inputs = torch.randn((2, 5, k), generator=torch.Generator().manual_seed(1201), dtype=torch.float32)
        proj_down_post = torch.randn((k, rank), generator=torch.Generator().manual_seed(1202), dtype=torch.float32) * 0.05
        proj_down_raw = proj_down_post / smooth_factor.reshape(k, 1)
        proj_up = torch.randn((n, rank), generator=torch.Generator().manual_seed(1203), dtype=torch.float32) * 0.07

        raw_basis = reference_svdquant_w4a4_linear_runtime(
            inputs,
            packed_zero_weight,
            weight_scale,
            smooth_factor,
            proj_down_raw,
            proj_up,
            group_size=64,
            branch_input_basis="raw",
        )
        post_smoothing_basis = reference_svdquant_w4a4_linear_runtime(
            inputs,
            packed_zero_weight,
            weight_scale,
            smooth_factor,
            proj_down_post,
            proj_up,
            group_size=64,
            branch_input_basis="post_smoothing",
        )

        self.assertTrue(torch.allclose(raw_basis, post_smoothing_basis, atol=1e-5, rtol=1e-5))

    def test_runtime_linear_accepts_kitchen_tile_packed_checkpoint_tensors(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import reference_svdquant_w4a4_linear_runtime
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs
        from comfy_quants.formats.kitchen_tilepack import to_kitchen_tile_packed_params

        n, k, rank = 128, 128, 2
        codes = torch.arange(n * k, dtype=torch.int32).reshape(n, k).remainder(11).sub(5).to(torch.int8)
        natural = {
            "weight": pack_signed_int4_pairs(codes),
            "weight_scale": torch.linspace(0.02, 0.2, (k // 64) * n, dtype=torch.float32).reshape(k // 64, n).to(torch.bfloat16),
            "smooth_factor": torch.linspace(0.9, 1.7, k, dtype=torch.float32),
            "proj_down": torch.randn((k, rank), generator=torch.Generator().manual_seed(1301), dtype=torch.float32) * 0.02,
            "proj_up": torch.randn((n, rank), generator=torch.Generator().manual_seed(1302), dtype=torch.float32) * 0.03,
            "bias": torch.randn((n,), generator=torch.Generator().manual_seed(1303), dtype=torch.float32) * 0.01,
        }
        packed = to_kitchen_tile_packed_params(natural)
        inputs = torch.randn((2, k), generator=torch.Generator().manual_seed(1304), dtype=torch.float32)

        natural_out = reference_svdquant_w4a4_linear_runtime(
            inputs,
            natural["weight"],
            natural["weight_scale"],
            natural["smooth_factor"],
            natural["proj_down"],
            natural["proj_up"],
            bias=natural["bias"],
            branch_input_basis="post_smoothing",
        )
        packed_out = reference_svdquant_w4a4_linear_runtime(
            inputs,
            packed["weight"],
            packed["weight_scale"],
            packed["smooth_factor"],
            packed["proj_down"],
            packed["proj_up"],
            bias=packed["bias"],
            branch_input_basis="post_smoothing",
        )

        self.assertTrue(torch.allclose(packed_out, natural_out, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
