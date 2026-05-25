import unittest


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


class TestAwqW4A16(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required for AWQ tests")

    def test_unsigned_int4_pair_roundtrip(self):
        torch = self.torch
        from comfy_quants.formats.int4_common import pack_uint4_pairs, unpack_uint4_pairs

        values = torch.tensor([[0, 1, 7, 8, 14, 15]], dtype=torch.int8)
        packed = pack_uint4_pairs(values)
        restored = unpack_uint4_pairs(packed)

        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(list(packed.shape), [1, 3])
        self.assertTrue(torch.equal(restored, values))

    def test_awq_quantizer_shapes_and_dequant(self):
        torch = self.torch
        from comfy_quants.algorithms.awq_w4a16.weight_quant import (
            dequantize_awq_w4a16_weight,
            quantize_linear_weight_to_awq_w4a16_debug,
            quantize_linear_weight_to_awq_w4a16,
        )
        from comfy_quants.formats.int4_common import unpack_uint4_pairs

        weight = torch.linspace(-2.0, 3.0, 12 * 128, dtype=torch.float32).view(12, 128).to(torch.float16)
        tensors = quantize_linear_weight_to_awq_w4a16(weight, scale_dtype="float16")

        self.assertEqual(tuple(tensors.weight.shape), (12, 64))
        self.assertEqual(tuple(tensors.weight_scale.shape), (2, 12))
        self.assertEqual(tuple(tensors.weight_zero.shape), (2, 12))
        self.assertEqual(tensors.weight.dtype, torch.int8)
        self.assertEqual(tensors.weight_scale.dtype, torch.float16)
        self.assertEqual(tensors.weight_zero.dtype, torch.float16)
        unpacked = unpack_uint4_pairs(tensors.weight)
        self.assertGreaterEqual(int(unpacked.min()), 0)
        self.assertLessEqual(int(unpacked.max()), 15)
        restored = dequantize_awq_w4a16_weight(tensors.weight, tensors.weight_scale, tensors.weight_zero)
        self.assertEqual(tuple(restored.shape), tuple(weight.shape))
        self.assertLess(float((restored - weight.float()).abs().mean()), 0.2)

        debug = quantize_linear_weight_to_awq_w4a16_debug(weight, scale_dtype="float32")
        manual = (
            (unpack_uint4_pairs(debug.packed_weight).float().view(12, 2, 64) - 8.0)
            * debug.weight_scale.t().float().view(12, 2, 1)
            + debug.weight_zero.t().float().view(12, 2, 1)
        ).view(12, 128)
        self.assertTrue(torch.allclose(manual, debug.dequantized_weight, atol=1e-6, rtol=1e-6))

    def test_qwen_modulation_reorder_helper(self):
        torch = self.torch
        from comfy_quants.algorithms.awq_w4a16.qwen_modulation import reorder_qwen_modulation_awq_tensors

        params = {
            "weight": torch.arange(12 * 4, dtype=torch.int8).view(12, 4),
            "weight_scale": torch.arange(2 * 12, dtype=torch.float16).view(2, 12),
            "weight_zero": (torch.arange(2 * 12, dtype=torch.float16).view(2, 12) + 100),
        }
        bias = torch.arange(12, dtype=torch.float16)
        reordered, reordered_bias = reorder_qwen_modulation_awq_tensors(params, bias=bias)

        expected_weight = params["weight"].view(2, 6, 4).transpose(0, 1).reshape(12, 4)
        expected_scale = params["weight_scale"].view(2, 2, 6).transpose(1, 2).reshape(2, 12)
        expected_zero = params["weight_zero"].view(2, 2, 6).transpose(1, 2).reshape(2, 12)
        expected_bias = bias.view(2, 6).transpose(0, 1).reshape(12)
        self.assertTrue(torch.equal(reordered["weight"], expected_weight))
        self.assertTrue(torch.equal(reordered["weight_scale"], expected_scale))
        self.assertTrue(torch.equal(reordered["weight_zero"], expected_zero))
        self.assertTrue(torch.equal(reordered_bias, expected_bias))


if __name__ == "__main__":
    unittest.main()
