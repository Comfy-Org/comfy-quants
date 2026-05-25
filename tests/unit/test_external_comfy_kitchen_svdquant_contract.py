import functools
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


@functools.lru_cache(maxsize=1)
def _load_comfy_kitchen_eager_svdquant():
    root = Path(os.environ.get("COMFY_QUANTS_COMFY_KITCHEN_SOURCE", str(Path.cwd().parent / "external" / "comfy-kitchen-hk416-awq")))
    path = root / "comfy_kitchen" / "backends" / "eager" / "svdquant.py"
    if not path.is_file():
        raise unittest.SkipTest(f"comfy-kitchen SVDQuant eager oracle is not available at {path}")
    spec = importlib.util.spec_from_file_location("_comfy_quants_ck_eager_svdquant_oracle", path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"cannot import comfy-kitchen SVDQuant eager oracle at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExternalComfyKitchenSvdquantContract(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required")
        self.ck_eager = _load_comfy_kitchen_eager_svdquant()

    def _external_eager_forward(self, *, inputs, weight, weight_scale, smooth_factor, proj_down, proj_up, bias, act_unsigned):
        x = inputs.reshape(-1, inputs.shape[-1]).contiguous()
        if act_unsigned:
            x_main = x + float(self.ck_eager._GELU_UNSIGNED_SHIFT)
            lora_x = x
        else:
            x_main = x
            lora_x = None
        q_x, ascales, lora_act = self.ck_eager.quantize_svdquant_w4a4(
            x_main,
            smooth=smooth_factor,
            lora_down=proj_down,
            pad_size=1,
            act_unsigned=act_unsigned,
            lora_x=lora_x,
        )
        out = self.ck_eager.scaled_mm_svdquant_w4a4(
            act=q_x,
            wgt=weight,
            ascales=ascales,
            wscales=weight_scale,
            lora_act_in=lora_act,
            lora_up=proj_up,
            bias=bias,
            act_unsigned=act_unsigned,
        )
        return out.reshape(*inputs.shape[:-1], -1).contiguous()

    def test_signed_runtime_reference_matches_comfy_kitchen_eager_contract(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import reference_svdquant_w4a4_linear_runtime
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs

        n, k, rank = 128, 64, 4
        codes = torch.arange(n * k, dtype=torch.int16).reshape(n, k).remainder(15).sub(7).to(torch.int8)
        weight = pack_signed_int4_pairs(codes)
        weight_scale = torch.linspace(0.01, 0.04, n, dtype=torch.float32).reshape(1, n)
        smooth = torch.linspace(0.8, 1.4, k, dtype=torch.float32)
        proj_down = torch.randn((k, rank), generator=torch.Generator().manual_seed(5101), dtype=torch.float32) * 0.02
        proj_up = torch.randn((n, rank), generator=torch.Generator().manual_seed(5102), dtype=torch.float32) * 0.03
        bias = torch.randn((n,), generator=torch.Generator().manual_seed(5103), dtype=torch.float32) * 0.01
        inputs = torch.randn((3, k), generator=torch.Generator().manual_seed(5104), dtype=torch.float32) * 0.8

        ours = reference_svdquant_w4a4_linear_runtime(
            inputs,
            weight,
            weight_scale,
            smooth,
            proj_down,
            proj_up,
            bias=bias,
            activation_signedness="signed",
            branch_input_basis="raw",
        )
        theirs = self._external_eager_forward(
            inputs=inputs,
            weight=weight,
            weight_scale=weight_scale,
            smooth_factor=smooth,
            proj_down=proj_down,
            proj_up=proj_up,
            bias=bias,
            act_unsigned=False,
        )
        self.assertTrue(torch.allclose(ours, theirs, atol=1e-5, rtol=1e-5))

    def test_unsigned_runtime_reference_matches_shifted_comfy_kitchen_eager_contract(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.runtime_reference import (
            GELU_UNSIGNED_SHIFT,
            reference_svdquant_w4a4_linear_runtime,
        )
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs
        from comfy_quants.formats.kitchen_tilepack import to_kitchen_tile_packed_params

        n, k, rank = 128, 128, 3
        codes = torch.arange(n * k, dtype=torch.int16).reshape(n, k).remainder(15).sub(7).to(torch.int8)
        natural = {
            "weight": pack_signed_int4_pairs(codes),
            "weight_scale": (torch.linspace(0.008, 0.035, (k // 64) * n, dtype=torch.float32).reshape(k // 64, n)).to(torch.bfloat16),
            "smooth_factor": torch.linspace(0.75, 1.55, k, dtype=torch.float32),
            "proj_down": (torch.randn((k, rank), generator=torch.Generator().manual_seed(5201), dtype=torch.float32) * 0.015).to(torch.bfloat16),
            "proj_up": (torch.randn((n, rank), generator=torch.Generator().manual_seed(5202), dtype=torch.float32) * 0.025).to(torch.bfloat16),
            "bias": (torch.randn((n,), generator=torch.Generator().manual_seed(5203), dtype=torch.float32) * 0.01).to(torch.bfloat16),
        }
        packed = to_kitchen_tile_packed_params(natural)
        inputs = torch.linspace(-1.1, 0.9, 4 * k, dtype=torch.float32).reshape(4, k).to(torch.bfloat16)

        ours = reference_svdquant_w4a4_linear_runtime(
            inputs,
            packed["weight"],
            packed["weight_scale"],
            packed["smooth_factor"],
            packed["proj_down"],
            packed["proj_up"],
            bias=packed["bias"],
            activation_signedness="unsigned",
            branch_input_basis="raw",
        )
        theirs = self._external_eager_forward(
            inputs=inputs,
            weight=packed["weight"],
            weight_scale=packed["weight_scale"],
            smooth_factor=packed["smooth_factor"],
            proj_down=packed["proj_down"],
            proj_up=packed["proj_up"],
            bias=packed["bias"],
            act_unsigned=True,
        )
        old_unshifted = reference_svdquant_w4a4_linear_runtime(
            inputs,
            packed["weight"],
            packed["weight_scale"],
            packed["smooth_factor"],
            packed["proj_down"],
            packed["proj_up"],
            bias=packed["bias"],
            activation_signedness="unsigned",
            branch_input_basis="raw",
            apply_unsigned_activation_shift=False,
        )
        self.assertEqual(GELU_UNSIGNED_SHIFT, float(self.ck_eager._GELU_UNSIGNED_SHIFT))
        self.assertTrue(torch.allclose(ours, theirs.float(), atol=2e-2, rtol=2e-2))
        self.assertGreater(float((ours - old_unshifted).abs().max().item()), 1e-4)


if __name__ == "__main__":
    unittest.main()
