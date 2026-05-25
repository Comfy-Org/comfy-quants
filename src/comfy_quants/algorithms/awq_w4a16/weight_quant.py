"""AWQ W4A16 tensor quantization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from comfy_quants.formats.awq_w4a16 import AWQ_W4A16_GROUP_SIZE
from comfy_quants.formats.int4_common import pack_uint4_pairs, unpack_uint4_pairs


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("torch is required for AWQ W4A16 quantization") from exc
    return torch


@dataclass(frozen=True)
class AwqW4A16LinearTensors:
    """Natural-layout tensors for one AWQ W4A16 linear layer."""

    weight: Any
    weight_scale: Any
    weight_zero: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight,
            "weight_scale": self.weight_scale,
            "weight_zero": self.weight_zero,
        }


@dataclass(frozen=True)
class AwqW4A16QuantizationDebug:
    """Intermediate tensors useful for tests and local calibration checks."""

    packed_weight: Any
    weight_scale: Any
    weight_zero: Any
    quantized_weight: Any
    dequantized_weight: Any


def _resolve_scale_dtype(torch: Any, source_dtype: Any, scale_dtype: str) -> Any:
    if scale_dtype == "source":
        if source_dtype in {torch.float16, torch.bfloat16, torch.float32}:
            return source_dtype
        return torch.float16
    if scale_dtype == "float16":
        return torch.float16
    if scale_dtype == "bfloat16":
        return torch.bfloat16
    if scale_dtype == "float32":
        return torch.float32
    raise ValueError(f"unsupported scale dtype: {scale_dtype}")


def _validate_linear_weight_shape(weight: Any, *, group_size: int) -> tuple[int, int]:
    if int(weight.ndim) != 2:
        raise ValueError(f"linear weight must be rank 2, got shape {tuple(weight.shape)}")
    n = int(weight.shape[0])
    k = int(weight.shape[1])
    if n <= 0 or k <= 0:
        raise ValueError(f"linear weight dimensions must be positive, got {tuple(weight.shape)}")
    if k % int(group_size) != 0:
        raise ValueError(f"input dimension K={k} is not divisible by group size {group_size}")
    if k % 2 != 0:
        raise ValueError(f"input dimension K={k} must be even for W4 pair packing")
    return n, k


def quantize_linear_weight_to_awq_w4a16_debug(
    weight: Any,
    *,
    group_size: int = AWQ_W4A16_GROUP_SIZE,
    scale_dtype: str = "source",
) -> AwqW4A16QuantizationDebug:
    """Quantize a dense linear weight to natural AWQ W4A16 tensors.

    This is a static checkpoint writer helper for the kitchen-native AWQ W4A16
    tensor contract.  It uses groupwise asymmetric uint4 quantization over each
    output row and K-axis group.  The stored zero is an additive floating-point
    group center.  Dequantization follows the target layout formula:

    ``(uint4_weight - 8) * weight_scale + weight_zero``.
    """
    torch = _require_torch()
    group_size = int(group_size)
    n, k = _validate_linear_weight_shape(weight, group_size=group_size)
    out_dtype = _resolve_scale_dtype(torch, weight.dtype, scale_dtype)
    groups = k // group_size

    working = weight.detach().to(dtype=torch.float32).contiguous().view(n, groups, group_size)
    w_min = working.amin(dim=2)
    w_max = working.amax(dim=2)
    span = w_max - w_min
    scale = torch.where(span > 0, span / 15.0, torch.ones_like(span))
    zero = torch.where(span > 0, w_min + (8.0 * scale), w_min)
    centered = torch.round((working - zero.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(-8, 7)
    quant = (centered + 8).to(torch.int8).contiguous()
    dequant = (centered * scale.unsqueeze(-1) + zero.unsqueeze(-1)).view(n, k).contiguous()
    return AwqW4A16QuantizationDebug(
        packed_weight=pack_uint4_pairs(quant.view(n, k), validate=False),
        weight_scale=scale.t().contiguous().to(dtype=out_dtype),
        weight_zero=zero.t().contiguous().to(dtype=out_dtype),
        quantized_weight=quant.view(n, k).contiguous(),
        dequantized_weight=dequant,
    )


def quantize_linear_weight_to_awq_w4a16(
    weight: Any,
    *,
    group_size: int = AWQ_W4A16_GROUP_SIZE,
    scale_dtype: str = "source",
) -> AwqW4A16LinearTensors:
    """Quantize one dense linear weight to natural AWQ W4A16 checkpoint tensors."""
    quantized = quantize_linear_weight_to_awq_w4a16_debug(weight, group_size=group_size, scale_dtype=scale_dtype)
    return AwqW4A16LinearTensors(
        weight=quantized.packed_weight,
        weight_scale=quantized.weight_scale,
        weight_zero=quantized.weight_zero,
    )


def dequantize_awq_w4a16_weight(
    packed_weight: Any,
    weight_scale: Any,
    weight_zero: Any,
    *,
    group_size: int = AWQ_W4A16_GROUP_SIZE,
):
    """Dequantize natural AWQ W4A16 tensors using the kitchen-native formula."""
    torch = _require_torch()
    if int(weight_scale.ndim) != 2:
        raise ValueError(f"weight_scale must have shape (K/{group_size}, N), got {tuple(weight_scale.shape)}")
    if int(weight_zero.ndim) != 2:
        raise ValueError(f"weight_zero must have shape (K/{group_size}, N), got {tuple(weight_zero.shape)}")
    quant = unpack_uint4_pairs(packed_weight).to(dtype=torch.float32)
    n, k = _validate_linear_weight_shape(quant, group_size=int(group_size))
    groups = k // int(group_size)
    expected = (groups, n)
    if tuple(int(x) for x in weight_scale.shape) != expected:
        raise ValueError(f"weight_scale shape {tuple(weight_scale.shape)} does not match quantized weight shape {(n, k)}")
    if tuple(int(x) for x in weight_zero.shape) != expected:
        raise ValueError(f"weight_zero shape {tuple(weight_zero.shape)} does not match quantized weight shape {(n, k)}")
    scale = weight_scale.transpose(0, 1).to(device=quant.device, dtype=torch.float32).reshape(n, groups, 1)
    zero = weight_zero.transpose(0, 1).to(device=quant.device, dtype=torch.float32).reshape(n, groups, 1)
    return ((quant.view(n, groups, int(group_size)) - 8.0) * scale + zero).view(n, k).contiguous()
