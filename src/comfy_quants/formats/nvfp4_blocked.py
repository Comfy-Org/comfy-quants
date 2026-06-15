"""Pure-torch NVFP4 (FP4 E2M1 microscaling) block quantization.

Offline math for the native-ComfyUI NVFP4 producer. NVFP4 stores **FP4-E2M1**
elements (packed 2-per-byte) with **two-level scaling**: a per-tensor float32 scale
and a per-block-16 FP8-E4M3 scale (laid out in the cuBLAS ``to_blocked`` swizzle).
ComfyUI loads it via ``QUANT_ALGOS["nvfp4"]`` / ``TensorCoreNVFP4Layout`` on Blackwell.

Bit-faithful to comfy-kitchen's deterministic eager ``quantize_nvfp4`` and its
``float_utils`` (``_f32_to_floatx_unpacked`` / ``pack_uint4`` / ``_float8_round`` /
``to_blocked``). The ``to_blocked``/``from_blocked`` swizzle is shared with the MXFP8
exporter (:mod:`comfy_quants.formats.mxfp8_blocked`). Pure torch (behind
``_require_torch()``); no comfy_kitchen, no CUDA dependency.

Portions of ``_f32_to_floatx_unpacked`` are derived from PyTorch AO
(https://github.com/pytorch/ao, BSD 3-Clause) via comfy-kitchen's ``float_utils``.

Reference layout: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
"""

from __future__ import annotations

from typing import Any

from comfy_quants.core.errors import PayloadWriteError

# Reuse the identical cuBLAS block-scale swizzle from the MXFP8 exporter.
from comfy_quants.formats.mxfp8_blocked import from_blocked, to_blocked

# OCP NVFP4 constants (match comfy-kitchen float_utils / ComfyUI comfy/float.py).
BLOCK_SIZE = 16
F4_E2M1_MAX = 6.0
F8_E4M3_MAX = 448.0

# IEEE-754 fp32 field widths (for the sub-byte float encoder below).
_EBITS_F32 = 8
_MBITS_F32 = 23
_F32_EXP_BIAS = (1 << (_EBITS_F32 - 1)) - 1  # 127

# Signed E2M1 magnitude grid, indexed by the 4-bit code (sign<<3)|(exp<<1)|mantissa.
E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise PayloadWriteError("torch is required for NVFP4 block quantization") from exc
    return torch


def _n_ones(n: int) -> int:
    return (1 << n) - 1


def _float8_round(x):
    """Round a float32 tensor to the float8_e4m3fn grid, returned as float32."""
    torch = _require_torch()
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def f32_to_floatx_unpacked(x, ebits: int, mbits: int):
    """Convert float32 to sub-byte float codes (round-to-nearest-even), one per byte.

    Returns a uint8 tensor with the ``1+ebits+mbits``-bit code in the least
    significant bits. Bit-faithful port of comfy-kitchen ``float_utils
    ._f32_to_floatx_unpacked`` (PyTorch-AO derived). For E2M1 call with
    ``ebits=2, mbits=1``.
    """
    torch = _require_torch()
    if x.dtype != torch.float32:
        raise PayloadWriteError("f32_to_floatx_unpacked requires a float32 tensor")
    if 1 + ebits + mbits > 8:
        raise PayloadWriteError("sub-byte float must fit in a byte")

    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)
    magic_adder = _n_ones(_MBITS_F32 - mbits - 1)

    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (_F32_EXP_BIAS - exp_bias) + (_MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << _MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32).to(x.device)

    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float32)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x = denormal_x - denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (_MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - _F32_EXP_BIAS) << _MBITS_F32) + magic_adder
    normal_x = normal_x + val_to_add
    normal_x = normal_x + mant_odd
    normal_x = normal_x >> (_MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    out = torch.full_like(x, max_int, dtype=torch.uint8)
    out = torch.where(denormal_mask, denormal_x, out)
    out = torch.where(normal_mask, normal_x, out)

    sign_lp = sign >> (_MBITS_F32 + _EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    sign_lp = sign_lp & sign_mask
    out = out | sign_lp
    return out.to(torch.uint8)


def pack_uint4(nibbles):
    """Pack a uint8 nibble tensor 2-per-byte: HIGH nibble = even index, LOW = odd.

    Last dim must be even; output last dim is halved. Matches comfy-kitchen
    ``float_utils.pack_uint4`` and ComfyUI's ``comfy/float.py`` pack order.
    """
    torch = _require_torch()
    shape = nibbles.shape
    if shape[-1] % 2 != 0:
        raise PayloadWriteError("pack_uint4 requires an even last dimension")
    flat = nibbles.contiguous().view(-1)
    packed = (flat[::2] << 4) | flat[1::2]
    return packed.view(*shape[:-1], shape[-1] // 2)


def unpack_uint4(packed):
    """Inverse of :func:`pack_uint4` (HIGH nibble first), for round-trip tests."""
    torch = _require_torch()
    shape = packed.shape
    hi = (packed >> 4).to(torch.uint8)
    lo = (packed & 0x0F).to(torch.uint8)
    return torch.stack([hi, lo], dim=-1).view(*shape[:-1], shape[-1] * 2)


def e2m1_to_f32(codes):
    """Decode uint8 E2M1 nibble codes (0..15) to float32 via the value LUT."""
    torch = _require_torch()
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    return lut[codes.long()]


def quantize_nvfp4_block(weight) -> tuple[Any, Any, Any]:
    """Quantize a 2D weight to NVFP4: packed FP4-E2M1 + two-level scales.

    Returns ``(weight_uint8, weight_scale_fp8, weight_scale_2_fp32)``:
      * ``weight_uint8``  — FP4-E2M1 nibbles packed 2/byte, shape ``[out, in//2]``
      * ``weight_scale_fp8`` — per-block-16 scale, ``float8_e4m3fn``, ``to_blocked``
        swizzled, shape ``(128*ceil(out/128), 4*ceil((in/16)/4))``
      * ``weight_scale_2_fp32`` — per-tensor scale, 0-dim ``float32``

    Deterministic round-to-nearest-even. Bit-faithful to comfy-kitchen eager
    ``quantize_nvfp4(W, per_tensor_scale, pad_16x=False)`` for ``in % 16 == 0``
    (true for the Qwen-Image families); padding non-aligned weights is future work.
    """
    torch = _require_torch()
    w = weight.detach()
    if w.dim() != 2:
        raise PayloadWriteError("NVFP4 export requires a rank-2 weight tensor")
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    if in_f % BLOCK_SIZE != 0:
        raise PayloadWriteError(f"NVFP4 in_features {in_f} must be a multiple of block size {BLOCK_SIZE}")

    wf = w.to(torch.float32)
    # Per-tensor scale = amax / (448*6); guard /0 for the divisions only (never
    # triggers for non-zero weights, so parity with the oracle is unaffected).
    per_tensor = wf.abs().amax() / (F8_E4M3_MAX * F4_E2M1_MAX)
    per_tensor_div = per_tensor.clamp(min=2.0**-126)

    xb = wf.reshape(out_f, in_f // BLOCK_SIZE, BLOCK_SIZE)
    block_amax = xb.abs().amax(dim=-1)  # [out, in/16]
    block_scale = block_amax / F4_E2M1_MAX
    scaled_fp8 = (block_scale / per_tensor_div).clamp(max=F8_E4M3_MAX)  # stored (->fp8)

    total = per_tensor_div * _float8_round(scaled_fp8)
    total_safe = torch.where(total == 0, torch.ones_like(total), total)
    data_scaled = xb / total_safe.unsqueeze(-1)
    data_scaled = torch.where((total == 0).unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled)
    data_scaled = data_scaled.reshape(out_f, in_f).clamp(-F4_E2M1_MAX, F4_E2M1_MAX)

    nibbles = f32_to_floatx_unpacked(data_scaled.contiguous(), 2, 1)
    weight_uint8 = pack_uint4(nibbles)
    weight_scale = to_blocked(scaled_fp8.to(torch.float8_e4m3fn), flatten=False)
    weight_scale_2 = per_tensor.to(torch.float32)
    return weight_uint8.contiguous(), weight_scale.contiguous(), weight_scale_2


__all__ = [
    "BLOCK_SIZE",
    "F4_E2M1_MAX",
    "F8_E4M3_MAX",
    "E2M1_VALUES",
    "f32_to_floatx_unpacked",
    "pack_uint4",
    "unpack_uint4",
    "e2m1_to_f32",
    "quantize_nvfp4_block",
    "to_blocked",
    "from_blocked",
]
