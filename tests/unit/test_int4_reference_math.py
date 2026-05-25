import unittest


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


class TestInt4ReferenceMath(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required")

    def test_svdquant_reference_linear_matches_repo_formula_for_natural_tensors(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.reference import (
            dequantize_svdquant_w4a4_effective_weight,
            reference_svdquant_w4a4_linear,
        )
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs

        n, k, rank = 128, 128, 3
        codes = torch.arange(n * k, dtype=torch.int32).reshape(n, k).remainder(16).sub(8).to(torch.int8)
        packed_weight = pack_signed_int4_pairs(codes)
        weight_scale = torch.linspace(0.05, 0.2, (k // 64) * n, dtype=torch.float32).reshape(k // 64, n)
        smooth_factor = torch.linspace(0.75, 1.5, k, dtype=torch.float32)
        proj_down = torch.randn((k, rank), generator=torch.Generator().manual_seed(31)) * 0.01
        proj_up = torch.randn((n, rank), generator=torch.Generator().manual_seed(32)) * 0.02
        bias = torch.linspace(-0.5, 0.5, n, dtype=torch.float32)
        inputs = torch.randn((2, 5, k), generator=torch.Generator().manual_seed(33), dtype=torch.float32)

        effective_weight = dequantize_svdquant_w4a4_effective_weight(
            packed_weight,
            weight_scale,
            proj_down=proj_down,
            proj_up=proj_up,
        )
        expected = torch.matmul(inputs / smooth_factor.reshape(1, 1, k), effective_weight.t()) + bias.reshape(1, 1, n)
        actual = reference_svdquant_w4a4_linear(
            inputs,
            packed_weight,
            weight_scale,
            smooth_factor,
            proj_down,
            proj_up,
            bias=bias,
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_svdquant_reference_accepts_kitchen_tile_packed_checkpoint_tensors(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.reference import reference_svdquant_w4a4_linear
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs
        from comfy_quants.formats.kitchen_tilepack import to_kitchen_tile_packed_params

        n, k, rank = 128, 128, 2
        codes = torch.arange(n * k, dtype=torch.int32).reshape(n, k).remainder(9).sub(4).to(torch.int8)
        natural = {
            "weight": pack_signed_int4_pairs(codes),
            "weight_scale": torch.full((k // 64, n), 0.125, dtype=torch.float32),
            "smooth_factor": torch.linspace(1.0, 2.0, k, dtype=torch.float32),
            "proj_down": torch.randn((k, rank), generator=torch.Generator().manual_seed(41), dtype=torch.float32) * 0.03,
            "proj_up": torch.randn((n, rank), generator=torch.Generator().manual_seed(42), dtype=torch.float32) * 0.02,
            "bias": torch.randn((n,), generator=torch.Generator().manual_seed(43), dtype=torch.float32) * 0.01,
        }
        packed = to_kitchen_tile_packed_params(natural)
        inputs = torch.randn((4, k), generator=torch.Generator().manual_seed(44), dtype=torch.float32)

        natural_out = reference_svdquant_w4a4_linear(
            inputs,
            natural["weight"],
            natural["weight_scale"],
            natural["smooth_factor"],
            natural["proj_down"],
            natural["proj_up"],
            bias=natural["bias"],
        )
        packed_out = reference_svdquant_w4a4_linear(
            inputs,
            packed["weight"],
            packed["weight_scale"],
            packed["smooth_factor"],
            packed["proj_down"],
            packed["proj_up"],
            bias=packed["bias"],
        )

        self.assertTrue(torch.allclose(packed_out, natural_out, atol=1e-5, rtol=1e-5))

    def test_awq_reference_linear_matches_kitchen_formula(self):
        torch = self.torch
        from comfy_quants.algorithms.awq_w4a16.reference import reference_awq_w4a16_linear
        from comfy_quants.algorithms.awq_w4a16.weight_quant import dequantize_awq_w4a16_weight, quantize_linear_weight_to_awq_w4a16

        weight = torch.randn((12, 128), generator=torch.Generator().manual_seed(51), dtype=torch.float32) * 0.25
        tensors = quantize_linear_weight_to_awq_w4a16(weight, scale_dtype="float32")
        inputs = torch.randn((3, 7, 128), generator=torch.Generator().manual_seed(52), dtype=torch.float32)
        bias = torch.randn((12,), generator=torch.Generator().manual_seed(53), dtype=torch.float32) * 0.01

        dense_weight = dequantize_awq_w4a16_weight(tensors.weight, tensors.weight_scale, tensors.weight_zero)
        expected = torch.matmul(inputs, dense_weight.t()) + bias.reshape(1, 1, 12)
        actual = reference_awq_w4a16_linear(inputs, tensors.weight, tensors.weight_scale, tensors.weight_zero, bias=bias)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_reference_helpers_validate_input_width(self):
        torch = self.torch
        from comfy_quants.algorithms.awq_w4a16.reference import reference_awq_w4a16_linear
        from comfy_quants.algorithms.awq_w4a16.weight_quant import quantize_linear_weight_to_awq_w4a16

        tensors = quantize_linear_weight_to_awq_w4a16(torch.randn((4, 64), generator=torch.Generator().manual_seed(61)))
        with self.assertRaisesRegex(ValueError, "input K=63"):
            reference_awq_w4a16_linear(torch.randn((2, 63), generator=torch.Generator().manual_seed(62)), tensors.weight, tensors.weight_scale, tensors.weight_zero)


if __name__ == "__main__":
    unittest.main()
