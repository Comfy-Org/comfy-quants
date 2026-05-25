import unittest


def _torch_dep():
    try:
        import torch
    except ImportError:
        return None
    return torch


class TestInt4Gptq(unittest.TestCase):
    def setUp(self):
        torch = _torch_dep()
        if torch is None:
            self.skipTest("torch is required")
        self.torch = torch

    def test_hessian_builder_matches_normalized_xtx(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.gptq import build_gptq_hessian_from_activations

        first = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        second = torch.tensor([[7.0, 8.0, 9.0]])
        divisor = torch.tensor([1.0, 2.0, 3.0])
        stats = build_gptq_hessian_from_activations(
            [first, second],
            input_channel_divisor=divisor,
            hessian_block_size=1,
            normalization_sample_count=3,
        )

        rows = torch.cat([first, second], dim=0) / divisor.reshape(1, -1)
        expected = (2.0 / 3.0) * rows.t().matmul(rows)
        self.assertEqual(stats.channel_count, 3)
        self.assertEqual(stats.sample_count, 2)
        self.assertEqual(stats.row_count, 3)
        self.assertTrue(torch.allclose(stats.hessian, expected))

    def test_grouped_gptq_emits_valid_int4_payload(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.gptq import (
            GptqConfig,
            quantize_linear_weight_grouped_signed_int4_gptq,
        )

        weight = torch.tensor(
            [
                [0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70, -0.80],
                [0.25, -0.15, 0.05, -0.35, 0.45, -0.55, 0.65, -0.75],
                [-0.80, 0.70, -0.60, 0.50, -0.40, 0.30, -0.20, 0.10],
                [-0.75, 0.65, -0.55, 0.45, -0.35, 0.25, -0.15, 0.05],
            ],
            dtype=torch.float32,
        )
        result = quantize_linear_weight_grouped_signed_int4_gptq(
            weight,
            hessian=torch.eye(8),
            group_size=4,
            scale_dtype="float32",
            config=GptqConfig(block_size=3),
        )

        self.assertEqual(tuple(result.packed_weight.shape), (4, 4))
        self.assertEqual(tuple(result.weight_scale.shape), (2, 4))
        self.assertEqual(tuple(result.quantized_weight.shape), tuple(weight.shape))
        self.assertFalse(result.used_rtn_fallback)
        self.assertGreaterEqual(int(result.quantized_weight.min()), -8)
        self.assertLessEqual(int(result.quantized_weight.max()), 7)

        scale = result.weight_scale.t().unsqueeze(-1).expand(4, 2, 4).reshape_as(weight)
        expected_dequant = result.quantized_weight.to(torch.float32) * scale
        self.assertTrue(torch.allclose(result.dequantized_weight, expected_dequant))

    def test_dead_hessian_columns_are_zeroed_before_quantization(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.gptq import quantize_linear_weight_grouped_signed_int4_gptq

        weight = torch.randn((4, 8), generator=torch.Generator().manual_seed(7))
        hessian = torch.eye(8)
        hessian[3, 3] = 0.0
        result = quantize_linear_weight_grouped_signed_int4_gptq(weight, hessian=hessian, group_size=4)

        self.assertEqual(result.dead_column_count, 1)
        self.assertTrue(torch.equal(result.quantized_weight[:, 3], torch.zeros(4, dtype=torch.int8)))

    def test_cholesky_failure_can_fall_back_to_rtn(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.gptq import (
            GptqConfig,
            quantize_linear_weight_grouped_signed_int4_gptq,
        )

        weight = torch.randn((4, 8), generator=torch.Generator().manual_seed(8))
        result = quantize_linear_weight_grouped_signed_int4_gptq(
            weight,
            hessian=torch.eye(8),
            group_size=4,
            config=GptqConfig(num_inv_tries=0, fallback_to_rtn=True),
        )

        self.assertTrue(result.used_rtn_fallback)
        self.assertEqual(result.hessian_inverse_attempts, 0)
        self.assertEqual(tuple(result.packed_weight.shape), (4, 4))

    def test_gptq_natural_builder_uses_smoothed_activation_samples(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.gptq import GptqConfig
        from comfy_quants.algorithms.int4_svdquant.stats import ActivationStats
        from comfy_quants.algorithms.int4_svdquant.weight_quant import (
            dequantize_natural_svdquant_weight,
            quantize_linear_weight_to_gptq_natural_svdquant,
        )

        generator = torch.Generator().manual_seed(13)
        weight = torch.randn((8, 8), generator=generator, dtype=torch.float32) * 0.2
        samples = [torch.randn((5, 8), generator=generator, dtype=torch.float32)]
        stats = ActivationStats(input_amax=samples[0].abs().amax(dim=0), sample_count=1, element_count=5)

        natural = quantize_linear_weight_to_gptq_natural_svdquant(
            weight,
            activation_stats=stats,
            activation_samples=samples,
            rank=2,
            group_size=4,
            scale_dtype="float32",
            gptq_config=GptqConfig(block_size=4, hessian_block_size=2),
        )

        dequantized = dequantize_natural_svdquant_weight(natural.weight, natural.weight_scale, group_size=4)
        branch = natural.proj_up @ natural.proj_down.t()
        smoothed_reference = weight * natural.smooth_factor.to(torch.float32).reshape(1, -1)
        without_branch_error = torch.linalg.vector_norm(smoothed_reference - dequantized)
        with_branch_error = torch.linalg.vector_norm(smoothed_reference - dequantized - branch)

        self.assertEqual(tuple(natural.weight.shape), (8, 4))
        self.assertEqual(tuple(natural.proj_down.shape), (8, 2))
        self.assertEqual(tuple(natural.proj_up.shape), (8, 2))
        self.assertLessEqual(float(with_branch_error), float(without_branch_error) + 1.0e-6)

    def test_gptq_natural_builder_supports_output_error_lowrank_calibration(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.gptq import GptqConfig
        from comfy_quants.algorithms.int4_svdquant.stats import ActivationStats
        from comfy_quants.algorithms.int4_svdquant.weight_quant import (
            dequantize_natural_svdquant_weight,
            quantize_linear_weight_to_gptq_natural_svdquant,
        )

        generator = torch.Generator().manual_seed(37)
        weight = torch.randn((8, 8), generator=generator, dtype=torch.float32) * 0.25
        samples = [
            torch.randn((6, 8), generator=generator, dtype=torch.float32),
            torch.randn((5, 8), generator=generator, dtype=torch.float32),
        ]
        stats = ActivationStats(
            input_amax=torch.cat(samples, dim=0).abs().amax(dim=0),
            sample_count=2,
            element_count=11,
        )

        natural = quantize_linear_weight_to_gptq_natural_svdquant(
            weight,
            activation_stats=stats,
            activation_samples=(sample for sample in samples),
            rank=2,
            group_size=4,
            scale_dtype="float32",
            gptq_config=GptqConfig(block_size=4, hessian_block_size=3),
            lowrank_calibration="output_error",
            lowrank_ridge=1.0e-7,
        )

        dequantized = dequantize_natural_svdquant_weight(natural.weight, natural.weight_scale, group_size=4)
        branch = natural.proj_up @ natural.proj_down.t()
        smoothed_reference = weight * natural.smooth_factor.to(torch.float32).reshape(1, -1)

        self.assertEqual(tuple(natural.weight.shape), (8, 4))
        self.assertEqual(tuple(natural.proj_down.shape), (8, 2))
        self.assertEqual(tuple(natural.proj_up.shape), (8, 2))
        self.assertGreater(float(branch.abs().amax()), 0.0)
        self.assertTrue(torch.isfinite(dequantized).all())
        self.assertTrue(torch.isfinite(smoothed_reference - dequantized - branch).all())

    def test_output_error_lowrank_calibration_requires_activation_samples(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.stats import ActivationStats
        from comfy_quants.algorithms.int4_svdquant.weight_quant import quantize_linear_weight_to_gptq_natural_svdquant

        weight = torch.randn((8, 8), generator=torch.Generator().manual_seed(38), dtype=torch.float32) * 0.2
        stats = ActivationStats(input_amax=torch.ones((8,), dtype=torch.float32), sample_count=1, element_count=1)

        with self.assertRaisesRegex(ValueError, "requires activation_samples"):
            quantize_linear_weight_to_gptq_natural_svdquant(
                weight,
                activation_stats=stats,
                gptq_hessian=torch.eye(8),
                rank=2,
                group_size=4,
                lowrank_calibration="output_error",
            )


if __name__ == "__main__":
    unittest.main()
