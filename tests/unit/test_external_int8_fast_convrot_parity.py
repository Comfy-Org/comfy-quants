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


def _load_int8_fast_convrot():
    """Load ComfyUI-INT8-Fast's convrot.py as the parity oracle.

    SkipTest if the source is unavailable or its (top-level) scipy import fails.
    int8_quant.py is intentionally NOT loaded — it imports comfy.* / folder_paths.
    """
    root = Path(os.environ.get("COMFY_QUANTS_INT8_FAST_SOURCE", str(Path.cwd().parent / "external" / "int8-fast")))
    path = root / "convrot.py"
    if not path.is_file():
        raise unittest.SkipTest(f"ComfyUI-INT8-Fast convrot oracle is not available at {path}")
    spec = importlib.util.spec_from_file_location("_comfy_quants_int8_fast_convrot_oracle", path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"cannot import INT8-Fast convrot oracle at {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # convrot.py imports scipy at module level
        raise unittest.SkipTest(f"INT8-Fast convrot oracle import failed (missing dep?): {exc}")
    return module


def _oracle_quant(oracle, torch, w, *, convrot, group_size=256):
    """Inline reimplementation of INT8-Fast's offline weight quant (int8_quant.py),
    which is not importable directly. Uses the oracle's own Hadamard/rotation."""
    w = w.detach().to(torch.float32)
    rotated = False
    if convrot and w.shape[1] % group_size == 0:
        H = oracle.build_hadamard(group_size, device=w.device, dtype=w.dtype)
        w = oracle.rotate_weight(w, H, group_size)
        rotated = True
    scale = (w.abs().amax(dim=1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    q = w.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale, rotated


class TestExternalInt8FastConvRotParity(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required")
        self.oracle = _load_int8_fast_convrot()

    def test_build_hadamard_matches_int8_fast(self):
        from comfy_quants.formats.convrot import build_hadamard

        torch = self.torch
        for size in (4, 16, 64, 256):
            ours = build_hadamard(size)
            theirs = self.oracle.build_hadamard(size)
            self.assertTrue(torch.equal(ours, theirs), f"Hadamard mismatch at size {size}")

    def test_rotate_weight_matches_int8_fast(self):
        from comfy_quants.formats.convrot import build_hadamard, rotate_weight

        torch = self.torch
        torch.manual_seed(0)
        W = torch.randn(512, 1024)
        H_ours = build_hadamard(256)
        H_theirs = self.oracle.build_hadamard(256)
        self.assertTrue(torch.equal(rotate_weight(W, H_ours, 256), self.oracle.rotate_weight(W, H_theirs, 256)))

    def test_offline_rotate_plus_row_int8_is_bit_faithful(self):
        from comfy_quants.backends.int8_w8a8_model_export import _quantize_int8_per_row

        torch = self.torch
        torch.manual_seed(1)
        W = torch.randn(512, 1024)
        W[:, ::97] *= 12.0  # inject channel outliers ConvRot is meant to tame

        for convrot in (True, False):
            q, s, rotated = _quantize_int8_per_row(W, convrot=convrot, group_size=256)
            q_ref, s_ref, rotated_ref = _oracle_quant(self.oracle, torch, W, convrot=convrot, group_size=256)
            self.assertEqual(rotated, rotated_ref)
            self.assertTrue(torch.equal(q, q_ref), f"int8 weight mismatch (convrot={convrot})")
            self.assertTrue(torch.equal(s, s_ref), f"scale mismatch (convrot={convrot})")
            self.assertEqual(q.dtype, torch.int8)
            self.assertEqual(list(s.shape), [512, 1])

        # Non-divisible in_features: ConvRot must auto-disable.
        W2 = torch.randn(64, 300)
        q2, s2, rotated2 = _quantize_int8_per_row(W2, convrot=True, group_size=256)
        self.assertFalse(rotated2)
        q2_ref, s2_ref, _ = _oracle_quant(self.oracle, torch, W2, convrot=True, group_size=256)
        self.assertTrue(torch.equal(q2, q2_ref))
        self.assertTrue(torch.equal(s2, s2_ref))

    def test_convrot_lowers_w8a8_error_numeric_sanity(self):
        # Soft sanity: ConvRot should reduce W8A8 reconstruction error on outlier-heavy weights.
        from comfy_quants.backends.int8_w8a8_model_export import _quantize_int8_per_row
        from comfy_quants.formats.convrot import build_hadamard, rotate_activation

        torch = self.torch
        torch.manual_seed(2)
        gs, out_f, in_f, tok = 256, 256, 1024, 64
        W = torch.randn(out_f, in_f); W[:, ::97] *= 12.0
        x = torch.randn(tok, in_f); x[:, ::97] *= 12.0
        y_ref = x @ W.T
        H = build_hadamard(gs)

        def w8a8(q_w, s_w, rotate):
            xx = rotate_activation(x, H, gs) if rotate else x
            xs = (xx.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
            xq = xx.mul(1.0 / xs).round_().clamp_(-128, 127)
            acc = xq.float() @ q_w.float().T  # exact int8 GEMM via float (no overflow at this size)
            return acc.mul(xs).mul(s_w.T)

        qc, sc, _ = _quantize_int8_per_row(W, convrot=True, group_size=gs)
        qp, sp, _ = _quantize_int8_per_row(W, convrot=False, group_size=gs)
        err_convrot = (w8a8(qc, sc, True) - y_ref).norm() / y_ref.norm()
        err_plain = (w8a8(qp, sp, False) - y_ref).norm() / y_ref.norm()
        self.assertLess(float(err_convrot), float(err_plain))


if __name__ == "__main__":
    unittest.main()
