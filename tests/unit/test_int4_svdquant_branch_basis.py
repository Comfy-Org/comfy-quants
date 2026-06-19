import unittest


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


class TestInt4SvdquantBranchBasis(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required")

    def test_fold_raw_branch_matches_post_smoothing_branch(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.branch_basis import fold_proj_down_for_raw_branch
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import reference_svdquant_w4a4_linear_runtime
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs

        n, k, rank = 8, 64, 5
        inputs = torch.randn((3, k), generator=torch.Generator().manual_seed(2301), dtype=torch.float32)
        smooth_factor = torch.linspace(0.6, 1.9, k, dtype=torch.float32)
        proj_down_post = torch.randn((k, rank), generator=torch.Generator().manual_seed(2302), dtype=torch.float16) * 0.02
        proj_up = torch.randn((n, rank), generator=torch.Generator().manual_seed(2303), dtype=torch.float16) * 0.03
        packed_zero_weight = pack_signed_int4_pairs(torch.zeros((n, k), dtype=torch.int8))
        weight_scale = torch.ones((1, n), dtype=torch.float16)

        proj_down_raw = fold_proj_down_for_raw_branch(proj_down_post, smooth_factor)
        self.assertEqual(proj_down_raw.dtype, torch.float16)

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

        self.assertTrue(torch.allclose(raw_basis, post_smoothing_basis, atol=5e-5, rtol=5e-5))

    def test_unfold_is_inverse_of_fold(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.branch_basis import (
            fold_proj_down_for_raw_branch,
            unfold_proj_down_for_post_smoothing_branch,
        )

        proj_down_post = torch.randn((64, 3), generator=torch.Generator().manual_seed(2311), dtype=torch.float32)
        smooth_factor = torch.linspace(0.75, 2.0, 64, dtype=torch.float32)

        folded = fold_proj_down_for_raw_branch(proj_down_post, smooth_factor)
        unfolded = unfold_proj_down_for_post_smoothing_branch(folded, smooth_factor)

        self.assertTrue(torch.allclose(unfolded, proj_down_post, atol=1e-6, rtol=1e-6))

    def test_fold_rejects_invalid_smooth_factor(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.branch_basis import fold_proj_down_for_raw_branch

        proj_down_post = torch.ones((4, 2), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "does not match"):
            fold_proj_down_for_raw_branch(proj_down_post, torch.ones((3,), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "must not contain zero"):
            fold_proj_down_for_raw_branch(proj_down_post, torch.tensor([1.0, 0.0, 1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
