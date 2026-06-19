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


def _load_ck_float_utils():
    """Load comfy-kitchen's float_utils.py as the NVFP4 parity oracle.

    float_utils.py imports only torch, so it loads standalone. SkipTest if the
    comfy-kitchen source is unavailable. Set COMFY_QUANTS_COMFY_KITCHEN_SOURCE to override.
    """
    root = Path(os.environ.get("COMFY_QUANTS_COMFY_KITCHEN_SOURCE", str(Path.cwd().parent / "external" / "comfy-kitchen")))
    path = root / "comfy_kitchen" / "float_utils.py"
    if not path.is_file():
        raise unittest.SkipTest(f"comfy-kitchen float_utils.py oracle is not available at {path}")
    spec = importlib.util.spec_from_file_location("_comfy_quants_ck_float_utils_oracle", path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"cannot import float_utils.py oracle at {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise unittest.SkipTest(f"float_utils.py oracle import failed (missing dep?): {exc}")
    for fn in ("_f32_to_floatx_unpacked", "pack_uint4", "to_blocked", "_float8_round"):
        if not hasattr(module, fn):
            raise unittest.SkipTest(f"comfy-kitchen float_utils lacks {fn} (update comfy-kitchen)")
    return module


def _oracle_quantize_nvfp4(fu, torch, w):
    """Inline composition of comfy-kitchen's eager quantize_nvfp4 using the
    standalone float_utils primitives (the eager module itself uses relative
    imports and is not standalone-importable). per_tensor = amax/(448*6)."""
    F4 = fu.F4_E2M1_MAX
    F8 = fu.F8_E4M3_MAX
    out_f, in_f = w.shape
    per_tensor = w.abs().amax().to(torch.float32) / (F8 * F4)
    xb = w.reshape(out_f, -1, 16).float()
    block_scale = xb.abs().amax(dim=-1) / F4
    scaled_fp8 = (block_scale / per_tensor).clamp(max=F8)
    total = per_tensor * fu._float8_round(scaled_fp8)
    total_safe = torch.where(total == 0, torch.ones_like(total), total)
    ds = xb / total_safe.unsqueeze(-1)
    ds = torch.where((total == 0).unsqueeze(-1), torch.zeros_like(ds), ds)
    ds = ds.reshape(out_f, in_f).clamp(-F4, F4)
    nibbles = fu._f32_to_floatx_unpacked(ds.contiguous(), 2, 1)
    weight = fu.pack_uint4(nibbles)
    weight_scale = fu.to_blocked(scaled_fp8.to(torch.float8_e4m3fn), flatten=False)
    return weight, weight_scale, per_tensor


class TestExternalNvFp4Parity(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required")
        self.fu = _load_ck_float_utils()

    def test_f32_to_floatx_matches_oracle(self):
        from comfy_quants.formats.nvfp4_blocked import f32_to_floatx_unpacked

        torch = self.torch
        torch.manual_seed(0)
        x = (torch.rand(257, dtype=torch.float32) - 0.5) * 12.0  # span [-6, 6]
        ours = f32_to_floatx_unpacked(x.contiguous(), 2, 1)
        theirs = self.fu._f32_to_floatx_unpacked(x.contiguous(), 2, 1)
        self.assertTrue(torch.equal(ours, theirs))

    def test_pack_uint4_matches_oracle(self):
        from comfy_quants.formats.nvfp4_blocked import pack_uint4

        torch = self.torch
        nibbles = torch.randint(0, 16, (8, 64), dtype=torch.uint8)
        self.assertTrue(torch.equal(pack_uint4(nibbles), self.fu.pack_uint4(nibbles)))

    def test_to_blocked_matches_oracle(self):
        from comfy_quants.formats.nvfp4_blocked import to_blocked

        torch = self.torch
        grid = torch.randint(0, 255, (130, 9), dtype=torch.uint8).to(torch.float8_e4m3fn)
        ours = to_blocked(grid, flatten=False)
        theirs = self.fu.to_blocked(grid, flatten=False)
        self.assertTrue(torch.equal(ours.view(torch.uint8), theirs.view(torch.uint8)))

    def test_full_quantize_matches_oracle(self):
        from comfy_quants.formats.nvfp4_blocked import quantize_nvfp4_block

        torch = self.torch
        torch.manual_seed(1)
        for (out_f, in_f) in [(64, 128), (128, 256), (96, 64)]:
            w = torch.randn(out_f, in_f, dtype=torch.float32)
            our_w, our_s, our_s2 = quantize_nvfp4_block(w)
            ora_w, ora_s, ora_s2 = _oracle_quantize_nvfp4(self.fu, torch, w)
            self.assertTrue(torch.equal(our_w, ora_w), f"weight mismatch at ({out_f},{in_f})")
            self.assertTrue(torch.equal(our_s.view(torch.uint8), ora_s.view(torch.uint8)), f"block scale mismatch at ({out_f},{in_f})")
            self.assertTrue(torch.equal(our_s2, ora_s2.to(torch.float32)), f"per-tensor scale mismatch at ({out_f},{in_f})")


if __name__ == "__main__":
    unittest.main()
