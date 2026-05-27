import unittest


def _torch():
    try:
        import torch
    except ImportError:
        return None
    if not hasattr(torch, "float8_e4m3fn"):
        return None
    return torch


class TestFP8TorchReference(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch.float8_e4m3fn is unavailable")

    def test_per_channel_fp8_e4m3_roundtrip(self):
        torch = self.torch
        from comfy_quants.backends.torch_ref import dequantize_fp8_e4m3_payload, quantize_tensor_fp8_e4m3

        source = torch.tensor(
            [
                [0.0, 1.0, -1.0, 2.0],
                [0.5, -0.25, 0.125, 0.0],
            ],
            dtype=torch.float32,
        )
        quantized = quantize_tensor_fp8_e4m3(source, granularity="per_channel", axis="out_features")
        restored = dequantize_fp8_e4m3_payload(quantized.payload, quantized.scale, axis=quantized.scale_axis)

        self.assertEqual(quantized.payload.dtype, torch.uint8)
        self.assertEqual(list(quantized.payload.shape), [2, 4])
        self.assertEqual(list(quantized.scale.shape), [2])
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue(torch.allclose(restored, source, rtol=0.06, atol=0.01))

    def test_zero_channel_scale_is_finite(self):
        torch = self.torch
        from comfy_quants.backends.torch_ref import quantize_tensor_fp8_e4m3

        source = torch.zeros((3, 4), dtype=torch.float32)
        quantized = quantize_tensor_fp8_e4m3(source, granularity="per_channel", axis="out_features")

        self.assertTrue(torch.isfinite(quantized.scale).all())
        self.assertTrue(torch.equal(quantized.scale, torch.ones_like(quantized.scale)))
        self.assertEqual(int(quantized.payload.max().item()), 0)

    def test_per_tensor_metadata(self):
        torch = self.torch
        from comfy_quants.backends.torch_ref import quantize_tensor_fp8_e4m3

        source = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32)
        quantized = quantize_tensor_fp8_e4m3(source, granularity="per_tensor", axis=None)
        metadata = quantized.to_metadata()

        self.assertEqual(list(quantized.scale.shape), [])
        self.assertEqual(metadata["payload_shape"], [2, 2])
        self.assertEqual(metadata["scale_shape"], [])
        self.assertEqual(metadata["quant_dtype"], "fp8_e4m3")
        self.assertEqual(metadata["storage_dtype"], "uint8")

    def test_per_channel_fp8_e5m2_roundtrip(self):
        torch = self.torch
        if not hasattr(torch, "float8_e5m2"):
            self.skipTest("torch.float8_e5m2 is unavailable")
        from comfy_quants.backends.torch_ref import dequantize_fp8_e5m2_payload, quantize_tensor_fp8_e5m2

        source = torch.tensor(
            [
                [0.0, 1.0, -1.0, 2.0],
                [0.5, -0.25, 0.125, 0.0],
            ],
            dtype=torch.float32,
        )
        quantized = quantize_tensor_fp8_e5m2(source, granularity="per_channel", axis="out_features")
        restored = dequantize_fp8_e5m2_payload(quantized.payload, quantized.scale, axis=quantized.scale_axis)

        self.assertEqual(quantized.payload.dtype, torch.uint8)
        self.assertEqual(list(quantized.payload.shape), [2, 4])
        self.assertEqual(list(quantized.scale.shape), [2])
        self.assertEqual(quantized.quant_dtype, "fp8_e5m2")
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue(torch.allclose(restored, source, rtol=0.18, atol=0.05))

    def test_generic_fp8_dispatch_metadata(self):
        torch = self.torch
        if not hasattr(torch, "float8_e5m2"):
            self.skipTest("torch.float8_e5m2 is unavailable")
        from comfy_quants.backends.torch_ref import quantize_tensor_fp8

        source = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32)
        quantized = quantize_tensor_fp8(source, quant_dtype="fp8_e5m2", granularity="per_tensor", axis=None)
        metadata = quantized.to_metadata()

        self.assertEqual(metadata["quant_dtype"], "fp8_e5m2")
        self.assertEqual(metadata["storage_dtype"], "uint8")
        self.assertEqual(metadata["payload_shape"], [2, 2])


if __name__ == "__main__":
    unittest.main()
