"""Pure-torch MXFP8 block quantization + cuBLAS ``to_blocked`` scale swizzle.

Offline math for the native-ComfyUI MXFP8 producer. The OCP microscaling FP8
format stores FP8-E4M3 elements with one **E8M0 power-of-2 block scale per 32
consecutive elements** (along the input dim), the block-scale grid laid out in the
cuBLAS ``to_blocked`` swizzle. ComfyUI loads it via ``QUANT_ALGOS["mxfp8"]`` and
``TensorCoreMXFP8Layout`` on Blackwell.

Bit-faithful to ComfyUI's own reference quantizer
(``comfy/float.py``: ``to_blocked`` and ``stochastic_round_quantize_mxfp8_by_block``)
and comfy-kitchen's pure-torch ``float_utils`` — except we use deterministic
round-to-nearest-even (``.to(float8_e4m3fn)``) instead of stochastic rounding, to
match ``rounding: nearest_even`` and our FP8 writer. Pure torch (behind
``_require_torch()``); no comfy_kitchen, no CUDA dependency.

Reference layout: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
"""

from __future__ import annotations

from typing import Any

from comfy_quants.core.errors import PayloadWriteError

# OCP MXFP8 constants (match comfy/float.py:225-227).
BLOCK_SIZE = 32
F8_E4M3_MAX = 448.0
E8M0_BIAS = 127


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise PayloadWriteError("torch is required for MXFP8 block quantization") from exc
    return torch


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def roundup(value: int, multiple: int) -> int:
    """Round ``value`` up to the nearest ``multiple`` (matches comfy/float.py)."""
    return ((value + multiple - 1) // multiple) * multiple


def to_blocked(input_matrix, flatten: bool = False):
    """Swizzle an ``(H, W)`` block-scale grid into the cuBLAS d-block layout.

    Output shape ``(128*ceil(H/128), 4*ceil(W/4))`` — the zero-padded grid shape,
    rearranged in place. Bit-faithful to ``comfy/float.py:99-137`` (we default
    ``flatten=False`` since we store the 2D form on disk). NB: comfy's docstring
    quotes ``(32*ceil(H/128), 16*ceil(W/4))`` but its code returns the padded grid
    shape ``(128*nr, 4*nc)`` (same element count, different shape).
    """
    torch = _require_torch()
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype)
        padded[:rows, :cols] = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    if flatten:
        return rearranged.flatten()
    return rearranged.reshape(padded_rows, padded_cols)


def from_blocked(blocked, num_rows: int, num_cols: int):
    """Inverse of :func:`to_blocked` — recover the ``(num_rows, num_cols)`` grid.

    Reverses the exact reshape/permute sequence and crops away the swizzle padding.
    Provided for round-trip tests and reference dequant.
    """
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    step = blocked.reshape(-1, 32, 16)
    step = step.reshape(-1, 32, 4, 4).transpose(1, 2)  # (-1, 4, 32, 4)
    step = step.reshape(n_row_blocks, n_col_blocks, 128, 4).permute(0, 2, 1, 3)
    unblocked = step.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols].contiguous()


def e8m0_to_f32(e8m0):
    """Decode E8M0 uint8 exponents to float32 scales (``2^(e-127)``; ``e==0 -> 0``).

    Matches ``comfy/float.py:239`` (``(e8m0.int() << 23).view(float32)``).
    """
    torch = _require_torch()
    return (e8m0.to(torch.int32) << 23).view(torch.float32)


def quantize_mxfp8_block(weight) -> tuple[Any, Any]:
    """Quantize a 2D weight to MXFP8: FP8-E4M3 elements + swizzled E8M0 block scales.

    Returns ``(weight_fp8, weight_scale_uint8)`` where ``weight_fp8`` is
    ``torch.float8_e4m3fn`` with the input shape ``[out, in]`` and
    ``weight_scale_uint8`` is ``torch.uint8`` in the ``to_blocked`` swizzle layout
    ``(128*ceil(out/128), 4*ceil((in/32)/4))``. Deterministic round-to-nearest-even.

    ``in_features`` must be a multiple of :data:`BLOCK_SIZE` (true for the Qwen-Image
    families); padding non-aligned weights is a future extension.
    """
    torch = _require_torch()
    w = weight.detach()
    if w.dim() != 2:
        raise PayloadWriteError("MXFP8 export requires a rank-2 weight tensor")
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    if in_f % BLOCK_SIZE != 0:
        raise PayloadWriteError(f"MXFP8 in_features {in_f} must be a multiple of block size {BLOCK_SIZE}")

    wf = w.to(torch.float32)
    xb = wf.reshape(out_f, in_f // BLOCK_SIZE, BLOCK_SIZE)
    max_abs = xb.abs().amax(dim=-1)  # [out, in/32]

    # E8M0 block scales (power-of-2 exponents); CEIL + bias 127, clamp [0, 254].
    scale_needed = (max_abs / F8_E4M3_MAX).clamp(min=2.0**-127)
    exp_biased = (torch.ceil(torch.log2(scale_needed)).to(torch.int32) + E8M0_BIAS).clamp(0, 254)
    block_scales_e8m0 = exp_biased.to(torch.uint8)  # [out, in/32]

    zero_mask = max_abs == 0
    scale_f32 = e8m0_to_f32(block_scales_e8m0)
    scale_f32 = torch.where(zero_mask, torch.ones_like(scale_f32), scale_f32)

    data_scaled = (xb / scale_f32.unsqueeze(-1)).reshape(out_f, in_f)
    qdata = data_scaled.clamp(-F8_E4M3_MAX, F8_E4M3_MAX).to(torch.float8_e4m3fn)

    block_scales_e8m0 = torch.where(zero_mask, torch.zeros_like(block_scales_e8m0), block_scales_e8m0)
    weight_scale = to_blocked(block_scales_e8m0, flatten=False)  # uint8 swizzled
    return qdata.contiguous(), weight_scale.contiguous()
