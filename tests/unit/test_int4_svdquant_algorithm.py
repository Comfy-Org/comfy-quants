import tempfile
import unittest
from pathlib import Path


def _torch_dep():
    try:
        import torch
    except ImportError:
        return None
    return torch


class TestInt4SvdquantAlgorithm(unittest.TestCase):
    def setUp(self):
        torch = _torch_dep()
        if torch is None:
            self.skipTest("torch is required")
        self.torch = torch

    def test_activation_stats_reduce_merge_and_json_roundtrip(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.stats import (
            activation_amax_from_samples,
            load_activation_stats_map,
            merge_activation_stats,
            write_activation_stats_map,
        )

        first = activation_amax_from_samples([torch.tensor([[1.0, -2.0], [3.0, 4.0]])])
        second = activation_amax_from_samples([torch.tensor([[-5.0, 1.0]])])
        merged = merge_activation_stats([first, second])

        self.assertTrue(torch.equal(merged.input_amax, torch.tensor([5.0, 4.0])))
        self.assertEqual(merged.sample_count, 2)
        self.assertEqual(merged.element_count, 3)
        self.assertTrue(torch.allclose(merged.input_rms, torch.sqrt(torch.tensor([(1.0 + 9.0 + 25.0) / 3.0, (4.0 + 16.0 + 1.0) / 3.0]))))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            write_activation_stats_map(path, {"transformer_blocks.0.attn.to_q": merged})
            loaded = load_activation_stats_map(path)
        self.assertIn("transformer_blocks.0.attn.to_q", loaded)
        self.assertTrue(torch.equal(loaded["transformer_blocks.0.attn.to_q"].input_amax, merged.input_amax))

    def test_smoothing_solver_clamps_and_scales_columns(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.smoothing import solve_smooth_factor

        weight = torch.ones((4, 4), dtype=torch.float32)
        stats = torch.tensor([1.0, 4.0, 16.0, 64.0], dtype=torch.float32)
        result = solve_smooth_factor(weight, stats, alpha=0.5, min_value=0.25, max_value=8.0)

        self.assertEqual(tuple(result.smooth_factor.shape), (4,))
        self.assertTrue(torch.isfinite(result.smooth_factor).all())
        self.assertTrue(torch.all(result.smooth_factor[1:] >= result.smooth_factor[:-1]))
        self.assertTrue(torch.allclose(result.smoothed_weight[0], result.smooth_factor.to(torch.float32)))

    def test_lowrank_solver_reconstructs_rank_one_residual(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.lowrank import solve_lowrank_residual_branch

        left = torch.tensor([[1.0], [2.0], [3.0]])
        right = torch.tensor([[4.0], [-1.0]])
        residual = left @ right.t()
        branch = solve_lowrank_residual_branch(residual, rank=1, dtype=torch.float32)
        reconstructed = branch.proj_up @ branch.proj_down.t()

        self.assertEqual(tuple(branch.proj_down.shape), (2, 1))
        self.assertEqual(tuple(branch.proj_up.shape), (3, 1))
        self.assertTrue(torch.allclose(reconstructed, residual, atol=1e-5, rtol=1e-5))

    def test_output_error_lowrank_solver_reconstructs_rank_limited_branch(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.lowrank import solve_lowrank_output_error_branch

        generator = torch.Generator().manual_seed(31)
        inputs = torch.randn((16, 5), generator=generator, dtype=torch.float32)
        true_down = torch.randn((5, 2), generator=generator, dtype=torch.float32)
        true_up = torch.randn((4, 2), generator=generator, dtype=torch.float32)
        output_residual = inputs @ true_down @ true_up.t()

        branch = solve_lowrank_output_error_branch(inputs, output_residual, rank=2, dtype=torch.float32, ridge=0.0)
        reconstructed_output = inputs @ branch.proj_down @ branch.proj_up.t()

        self.assertEqual(tuple(branch.proj_down.shape), (5, 2))
        self.assertEqual(tuple(branch.proj_up.shape), (4, 2))
        self.assertTrue(torch.allclose(reconstructed_output, output_residual, atol=2.0e-5, rtol=2.0e-5))

    def test_output_error_lowrank_solver_supports_smoothing_divisor(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.lowrank import solve_lowrank_output_error_branch

        generator = torch.Generator().manual_seed(32)
        raw_inputs = torch.randn((18, 6), generator=generator, dtype=torch.float32)
        smooth = torch.tensor([1.0, 2.0, 0.5, 4.0, 1.5, 0.75], dtype=torch.float32)
        post_smoothing_inputs = raw_inputs / smooth.reshape(1, -1)
        true_down = torch.randn((6, 2), generator=generator, dtype=torch.float32)
        true_up = torch.randn((3, 2), generator=generator, dtype=torch.float32)
        output_residual = post_smoothing_inputs @ true_down @ true_up.t()

        branch = solve_lowrank_output_error_branch(
            raw_inputs,
            output_residual,
            rank=2,
            dtype=torch.float32,
            ridge=0.0,
            input_channel_divisor=smooth,
        )
        reconstructed_output = post_smoothing_inputs @ branch.proj_down @ branch.proj_up.t()

        self.assertTrue(torch.allclose(reconstructed_output, output_residual, atol=2.0e-5, rtol=2.0e-5))

    def test_calibrated_builder_emits_non_identity_side_tensors(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.stats import ActivationStats
        from comfy_quants.algorithms.int4_svdquant.weight_quant import (
            dequantize_natural_svdquant_weight,
            quantize_linear_weight_to_calibrated_natural_svdquant,
        )

        generator = torch.Generator().manual_seed(11)
        weight = torch.randn((128, 128), generator=generator, dtype=torch.float32) * 0.17
        stats = ActivationStats(input_amax=torch.linspace(0.5, 4.0, 128))
        natural = quantize_linear_weight_to_calibrated_natural_svdquant(
            weight,
            activation_stats=stats,
            rank=4,
            scale_dtype="float32",
        )

        dequantized = dequantize_natural_svdquant_weight(natural.weight, natural.weight_scale)
        branch = natural.proj_up @ natural.proj_down.t()
        smoothed_reference = weight * natural.smooth_factor.to(torch.float32).reshape(1, -1)
        without_branch_error = torch.linalg.vector_norm(smoothed_reference - dequantized)
        with_branch_error = torch.linalg.vector_norm(smoothed_reference - dequantized - branch)

        self.assertEqual(tuple(natural.weight.shape), (128, 64))
        self.assertFalse(torch.allclose(natural.smooth_factor, torch.ones_like(natural.smooth_factor)))
        self.assertGreater(float(natural.proj_down.abs().amax()), 0.0)
        self.assertGreater(float(natural.proj_up.abs().amax()), 0.0)
        self.assertLess(float(with_branch_error), float(without_branch_error))

    def test_signed_svdquant_weight_quantizer_does_not_emit_negative_eight(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.weight_quant import quantize_linear_weight_grouped_signed_int4

        weight = torch.zeros((128, 64), dtype=torch.float32)
        weight[:, 0] = -8.0
        weight[:, 1] = 7.0

        result = quantize_linear_weight_grouped_signed_int4(weight, group_size=64, scale_dtype="float32")

        self.assertGreaterEqual(int(result.quantized_weight.min().item()), -7)
        self.assertLessEqual(int(result.quantized_weight.max().item()), 7)
        self.assertTrue(torch.allclose(result.weight_scale[0], torch.full((128,), 8.0 / 7.0)))

    def test_gptq_signed_int4_clamp_does_not_emit_negative_eight(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.gptq import torch_round_clamp_int4

        values = torch.tensor([-100.0, -8.0, -7.6, -7.5, -7.0, 0.0, 7.0, 100.0])
        quantized = torch_round_clamp_int4(values)

        self.assertEqual(quantized.tolist(), [-7, -7, -7, -7, -7, 0, 7, 7])

    def test_signed_activation_oracle_does_not_emit_negative_eight(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import quantize_activation_w4_signed

        inputs = torch.zeros((2, 64), dtype=torch.float32)
        inputs[:, 0] = -8.0
        inputs[:, 1] = 7.0

        result = quantize_activation_w4_signed(inputs, group_size=64)

        self.assertGreaterEqual(int(result.q_values.min().item()), -7)
        self.assertLessEqual(int(result.q_values.max().item()), 7)
        self.assertTrue(torch.allclose(result.scale, torch.full((2, 1), 8.0 / 7.0)))


if __name__ == "__main__":
    unittest.main()
