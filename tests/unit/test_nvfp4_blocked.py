import unittest


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


class TestNvFp4Blocked(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required")

    def test_pack_unpack_roundtrip(self):
        from comfy_quants.formats.nvfp4_blocked import pack_uint4, unpack_uint4

        torch = self.torch
        nibbles = torch.randint(0, 16, (4, 64), dtype=torch.uint8)
        packed = pack_uint4(nibbles)
        self.assertEqual(list(packed.shape), [4, 32])
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertTrue(torch.equal(unpack_uint4(packed), nibbles))

    def test_pack_order_high_even_low_odd(self):
        from comfy_quants.formats.nvfp4_blocked import pack_uint4

        torch = self.torch
        nibbles = torch.tensor([[0x0A, 0x0B]], dtype=torch.uint8)  # even=A (high), odd=B (low)
        packed = pack_uint4(nibbles)
        self.assertEqual(int(packed[0, 0]), 0xAB)

    def test_e2m1_lut_decode_all_codes(self):
        from comfy_quants.formats.nvfp4_blocked import E2M1_VALUES, e2m1_to_f32

        torch = self.torch
        codes = torch.arange(16, dtype=torch.uint8)
        vals = e2m1_to_f32(codes)
        self.assertEqual(vals.tolist(), list(E2M1_VALUES))

    def test_f32_to_floatx_encodes_grid_exactly(self):
        from comfy_quants.formats.nvfp4_blocked import f32_to_floatx_unpacked

        torch = self.torch
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
        codes = f32_to_floatx_unpacked(grid, 2, 1)
        self.assertEqual(codes.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
        neg = torch.tensor([-0.5, -1.0, -6.0], dtype=torch.float32)
        self.assertEqual(f32_to_floatx_unpacked(neg, 2, 1).tolist(), [9, 10, 15])

    def test_f32_to_floatx_roundtrip_via_lut(self):
        from comfy_quants.formats.nvfp4_blocked import e2m1_to_f32, f32_to_floatx_unpacked

        torch = self.torch
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                             -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
        codes = f32_to_floatx_unpacked(grid, 2, 1)
        self.assertTrue(torch.equal(e2m1_to_f32(codes).abs(), grid.abs()))

    def test_quantize_shapes_and_dtypes(self):
        from comfy_quants.formats.nvfp4_blocked import quantize_nvfp4_block

        torch = self.torch
        w = torch.randn(64, 256, dtype=torch.float32)
        weight, weight_scale, weight_scale_2 = quantize_nvfp4_block(w)
        self.assertEqual(weight.dtype, torch.uint8)
        self.assertEqual(list(weight.shape), [64, 128])  # packed 2/byte
        self.assertEqual(weight_scale.dtype, torch.float8_e4m3fn)
        # block grid [64, 256/16=16] -> to_blocked (128*ceil(64/128)=128, 4*ceil(16/4)=16)
        self.assertEqual(list(weight_scale.shape), [128, 16])
        self.assertEqual(weight_scale_2.dtype, torch.float32)
        self.assertEqual(weight_scale_2.dim(), 0)  # per-tensor scalar

    def test_quantize_is_deterministic(self):
        from comfy_quants.formats.nvfp4_blocked import quantize_nvfp4_block

        torch = self.torch
        w = torch.randn(32, 128, dtype=torch.float32)
        w1, s1, t1 = quantize_nvfp4_block(w)
        w2, s2, t2 = quantize_nvfp4_block(w)
        self.assertTrue(torch.equal(w1, w2))
        self.assertTrue(torch.equal(s1.view(torch.uint8), s2.view(torch.uint8)))
        self.assertTrue(torch.equal(t1, t2))

    def test_quantize_requires_block_aligned_in_features(self):
        from comfy_quants.core.errors import PayloadWriteError
        from comfy_quants.formats.nvfp4_blocked import quantize_nvfp4_block

        torch = self.torch
        with self.assertRaises(PayloadWriteError):
            quantize_nvfp4_block(torch.randn(8, 24, dtype=torch.float32))  # 24 % 16 != 0

    def test_dequant_reconstruction_is_close(self):
        from comfy_quants.formats.nvfp4_blocked import (
            BLOCK_SIZE,
            e2m1_to_f32,
            from_blocked,
            quantize_nvfp4_block,
            unpack_uint4,
        )

        torch = self.torch
        out_f, in_f = 64, 256
        w = torch.randn(out_f, in_f, dtype=torch.float32)
        weight, weight_scale, weight_scale_2 = quantize_nvfp4_block(w)
        codes = unpack_uint4(weight)  # [out, in]
        vals = e2m1_to_f32(codes)
        blocks = in_f // BLOCK_SIZE
        block_scale = from_blocked(weight_scale, out_f, blocks).to(torch.float32)  # [out, blocks]
        total = weight_scale_2.to(torch.float32) * block_scale
        dq = (vals.reshape(out_f, blocks, BLOCK_SIZE) * total.unsqueeze(-1)).reshape(out_f, in_f)
        rel = (dq - w).norm() / w.norm()
        self.assertLess(rel.item(), 0.2)  # 4-bit E2M1 is coarse

    def test_zero_weight_encodes_zero(self):
        from comfy_quants.formats.nvfp4_blocked import quantize_nvfp4_block

        torch = self.torch
        w = torch.zeros(32, 64, dtype=torch.float32)
        weight, weight_scale, weight_scale_2 = quantize_nvfp4_block(w)
        self.assertTrue(torch.equal(weight, torch.zeros_like(weight)))
        self.assertTrue(torch.equal(weight_scale.view(torch.uint8), torch.zeros_like(weight_scale.view(torch.uint8))))
        self.assertEqual(float(weight_scale_2), 0.0)


if __name__ == "__main__":
    unittest.main()
