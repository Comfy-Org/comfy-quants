import unittest


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


class TestMxFp8Blocked(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required")

    def test_to_blocked_output_shape(self):
        from comfy_quants.formats.mxfp8_blocked import to_blocked

        torch = self.torch
        for (h, w, exp_rows, exp_cols) in [
            (4, 2, 128, 4),     # padded grid: 128*ceil(4/128)=128, 4*ceil(2/4)=4
            (130, 2, 256, 4),   # 128*ceil(130/128)=256
            (128, 8, 128, 8),   # 4*ceil(8/4)=8
        ]:
            grid = torch.randint(0, 255, (h, w), dtype=torch.uint8)
            out = to_blocked(grid, flatten=False)
            self.assertEqual(list(out.shape), [exp_rows, exp_cols])
            self.assertEqual(out.dtype, torch.uint8)

    def test_to_blocked_from_blocked_roundtrip(self):
        from comfy_quants.formats.mxfp8_blocked import from_blocked, to_blocked

        torch = self.torch
        for (h, w) in [(4, 2), (130, 2), (256, 9), (1, 1)]:
            grid = torch.randint(0, 255, (h, w), dtype=torch.uint8)
            swizzled = to_blocked(grid, flatten=False)
            recovered = from_blocked(swizzled, h, w)
            self.assertEqual(list(recovered.shape), [h, w])
            self.assertTrue(torch.equal(recovered, grid))

    def test_e8m0_to_f32_known_values(self):
        from comfy_quants.formats.mxfp8_blocked import e8m0_to_f32

        torch = self.torch
        e = torch.tensor([0, 126, 127, 128, 129], dtype=torch.uint8)
        f = e8m0_to_f32(e)
        self.assertEqual(f.dtype, torch.float32)
        self.assertEqual(f.tolist(), [0.0, 0.5, 1.0, 2.0, 4.0])

    def test_quantize_shapes_and_dtypes(self):
        from comfy_quants.formats.mxfp8_blocked import quantize_mxfp8_block

        torch = self.torch
        w = torch.randn(64, 128, dtype=torch.float32)
        qweight, weight_scale = quantize_mxfp8_block(w)
        self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
        self.assertEqual(list(qweight.shape), [64, 128])
        self.assertEqual(weight_scale.dtype, torch.uint8)
        # padded grid: 128*ceil(64/128)=128 ; in/32=4 blocks -> 4*ceil(4/4)=4
        self.assertEqual(list(weight_scale.shape), [128, 4])

    def test_quantize_is_deterministic(self):
        from comfy_quants.formats.mxfp8_blocked import quantize_mxfp8_block

        torch = self.torch
        w = torch.randn(32, 256, dtype=torch.float32)
        q1, s1 = quantize_mxfp8_block(w)
        q2, s2 = quantize_mxfp8_block(w)
        self.assertTrue(torch.equal(q1.view(torch.uint8), q2.view(torch.uint8)))
        self.assertTrue(torch.equal(s1, s2))

    def test_quantize_requires_block_aligned_in_features(self):
        from comfy_quants.core.errors import PayloadWriteError
        from comfy_quants.formats.mxfp8_blocked import quantize_mxfp8_block

        torch = self.torch
        with self.assertRaises(PayloadWriteError):
            quantize_mxfp8_block(torch.randn(8, 33, dtype=torch.float32))

    def test_dequant_reconstruction_is_close(self):
        from comfy_quants.formats.mxfp8_blocked import (
            BLOCK_SIZE,
            e8m0_to_f32,
            from_blocked,
            quantize_mxfp8_block,
        )

        torch = self.torch
        out_f, in_f = 64, 256
        w = torch.randn(out_f, in_f, dtype=torch.float32)
        qweight, weight_scale = quantize_mxfp8_block(w)
        blocks = in_f // BLOCK_SIZE
        grid = from_blocked(weight_scale, out_f, blocks)  # (out, blocks) uint8
        scale = e8m0_to_f32(grid)  # (out, blocks)
        dq = (qweight.to(torch.float32).reshape(out_f, blocks, BLOCK_SIZE) * scale.unsqueeze(-1)).reshape(out_f, in_f)
        rel = (dq - w).norm() / w.norm()
        self.assertLess(rel.item(), 0.1)

    def test_zero_block_encodes_zero_scale(self):
        from comfy_quants.formats.mxfp8_blocked import quantize_mxfp8_block

        torch = self.torch
        w = torch.zeros(32, 64, dtype=torch.float32)
        qweight, weight_scale = quantize_mxfp8_block(w)
        # all-zero weight -> all E8M0 exponents 0 and all fp8 elements 0
        self.assertTrue(torch.equal(weight_scale, torch.zeros_like(weight_scale)))
        self.assertTrue(torch.equal(qweight.to(torch.float32), torch.zeros(32, 64)))


if __name__ == "__main__":
    unittest.main()
