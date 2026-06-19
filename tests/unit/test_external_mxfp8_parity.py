import importlib.util
import os
import unittest
from pathlib import Path


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _load_comfy_float():
    """Load ComfyUI's comfy/float.py as the MXFP8 parity oracle.

    comfy/float.py imports only torch, so it loads standalone. SkipTest if the
    ComfyUI source is unavailable. Set COMFY_QUANTS_COMFYUI_SOURCE to override.
    """
    root = Path(os.environ.get("COMFY_QUANTS_COMFYUI_SOURCE", str(Path.cwd().parent / "external" / "ComfyUI")))
    path = root / "comfy" / "float.py"
    if not path.is_file():
        raise unittest.SkipTest(f"ComfyUI comfy/float.py oracle is not available at {path}")
    spec = importlib.util.spec_from_file_location("_comfy_quants_comfy_float_oracle", path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"cannot import comfy/float.py oracle at {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise unittest.SkipTest(f"comfy/float.py oracle import failed (missing dep?): {exc}")
    if not hasattr(module, "to_blocked") or not hasattr(module, "stochastic_round_quantize_mxfp8_by_block"):
        raise unittest.SkipTest("comfy/float.py oracle lacks the MXFP8 helpers (update ComfyUI)")
    return module


class TestExternalMxFp8Parity(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required")
        self.oracle = _load_comfy_float()

    def test_to_blocked_matches_comfy(self):
        from comfy_quants.formats.mxfp8_blocked import to_blocked

        torch = self.torch
        torch.manual_seed(0)
        for (h, w) in [(4, 4), (130, 9), (64, 2), (256, 16)]:
            grid = torch.randint(0, 255, (h, w), dtype=torch.uint8)
            ours = to_blocked(grid, flatten=False)
            theirs = self.oracle.to_blocked(grid, flatten=False)
            self.assertEqual(ours.shape, theirs.shape)
            self.assertTrue(torch.equal(ours, theirs), f"to_blocked mismatch at ({h},{w})")

    def test_block_scale_matches_comfy(self):
        """Our deterministic E8M0 swizzled block scale == comfy's (scale path is
        deterministic; only comfy's FP8 element rounding is stochastic)."""
        from comfy_quants.formats.mxfp8_blocked import quantize_mxfp8_block

        torch = self.torch
        torch.manual_seed(0)
        for (out_f, in_f) in [(64, 128), (128, 256), (96, 64)]:
            w = torch.randn(out_f, in_f, dtype=torch.float32)
            _our_fp8, our_scale = quantize_mxfp8_block(w)
            # comfy returns (fp8, block_scale_e8m0fnu) with pad_32x=False (in%32==0).
            _comfy_fp8, comfy_scale = self.oracle.stochastic_round_quantize_mxfp8_by_block(
                w.clone(), pad_32x=False, seed=0
            )
            comfy_scale_u8 = comfy_scale.view(torch.uint8)
            self.assertEqual(our_scale.shape, comfy_scale_u8.shape)
            self.assertTrue(torch.equal(our_scale, comfy_scale_u8), f"E8M0 scale mismatch at ({out_f},{in_f})")

    def test_weight_matches_deterministic_oracle_from_comfy_scale(self):
        """Our FP8 weight == the deterministic RTN weight derived from comfy's own
        E8M0 scale (validates the encode path against the oracle's scale, without
        depending on comfy's stochastic element rounding)."""
        from comfy_quants.formats.mxfp8_blocked import (
            BLOCK_SIZE,
            F8_E4M3_MAX,
            e8m0_to_f32,
            from_blocked,
            quantize_mxfp8_block,
        )

        torch = self.torch
        torch.manual_seed(1)
        out_f, in_f = 64, 128
        w = torch.randn(out_f, in_f, dtype=torch.float32)
        our_fp8, _our_scale = quantize_mxfp8_block(w)

        _comfy_fp8, comfy_scale = self.oracle.stochastic_round_quantize_mxfp8_by_block(w.clone(), pad_32x=False, seed=0)
        blocks = in_f // BLOCK_SIZE
        grid = from_blocked(comfy_scale.view(torch.uint8), out_f, blocks)  # (out, blocks) uint8
        sf = e8m0_to_f32(grid)
        sf = torch.where(grid == 0, torch.ones_like(sf), sf)
        xb = w.reshape(out_f, blocks, BLOCK_SIZE)
        oracle_fp8 = (xb / sf.unsqueeze(-1)).reshape(out_f, in_f).clamp(-F8_E4M3_MAX, F8_E4M3_MAX).to(torch.float8_e4m3fn)
        self.assertTrue(torch.equal(our_fp8.view(torch.uint8), oracle_fp8.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
