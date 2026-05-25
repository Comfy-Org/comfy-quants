"""Smoothing solver for INT4 SVDQuant linear layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from comfy_quants.algorithms.int4_svdquant.stats import as_activation_vector


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("torch is required for INT4 SVDQuant smoothing") from exc
    return torch


@dataclass(frozen=True)
class SmoothResult:
    """Column smoothing result for one dense linear weight."""

    smooth_factor: Any
    smoothed_weight: Any


def solve_smooth_factor(
    weight: Any,
    activation_amax: Any,
    *,
    alpha: float = 0.5,
    min_value: float = 1.0 / 64.0,
    max_value: float = 64.0,
    eps: float = 1.0e-12,
    output_dtype: Any | None = None,
) -> SmoothResult:
    """Return a SmoothQuant-style factor and the column-scaled weight.

    The factor has one value per input channel.  The returned weight is scaled
    column-wise as ``weight * smooth_factor`` so the paired runtime can divide
    the input channel by the same factor before low-bit matmul.
    """
    torch = _require_torch()
    if int(weight.ndim) != 2:
        raise ValueError(f"linear weight must be rank 2, got shape {tuple(weight.shape)}")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if float(min_value) <= 0 or float(max_value) <= 0 or float(min_value) > float(max_value):
        raise ValueError(f"invalid smooth clamp range: min={min_value}, max={max_value}")
    if float(eps) <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    n, k = int(weight.shape[0]), int(weight.shape[1])
    if n <= 0 or k <= 0:
        raise ValueError(f"linear weight dimensions must be positive, got {tuple(weight.shape)}")
    act = as_activation_vector(activation_amax, name="activation_amax", device=weight.device).abs()
    if int(act.numel()) != k:
        raise ValueError(f"activation_amax length {int(act.numel())} does not match input dimension K={k}")

    working = weight.detach().to(dtype=torch.float32)
    weight_col_amax = working.abs().amax(dim=0)
    safe_act = torch.clamp(act, min=float(eps))
    safe_weight = torch.clamp(weight_col_amax, min=float(eps))
    smooth = safe_act.pow(float(alpha)) / safe_weight.pow(1.0 - float(alpha))
    smooth = torch.clamp(smooth, min=float(min_value), max=float(max_value))
    smooth = torch.where(torch.isfinite(smooth), smooth, torch.ones_like(smooth))

    smoothed_weight = (working * smooth.reshape(1, k)).contiguous()
    if output_dtype is None:
        output_dtype = weight.dtype if weight.dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.float16
    return SmoothResult(
        smooth_factor=smooth.to(dtype=output_dtype).contiguous(),
        smoothed_weight=smoothed_weight,
    )
