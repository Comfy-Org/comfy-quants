"""Qwen modulation-layout helpers for AWQ W4A16 tensors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def reorder_qwen_modulation_awq_tensors(
    params: Mapping[str, Any],
    *,
    bias: Any | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """Transpose Qwen modulation rows from ``[dim, 6]`` order to ``[6, dim]``.

    This helper is intentionally explicit.  Direct dense checkpoints that are
    already in the target row order should not call it; bridge exporters from
    runtimes that store modulation rows interleaved per channel can opt in.
    """
    required = ("weight", "weight_scale", "weight_zero")
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f"missing AWQ modulation tensors: {', '.join(missing)}")
    weight = params["weight"]
    weight_scale = params["weight_scale"]
    weight_zero = params["weight_zero"]
    n_orig = int(weight.shape[0])
    if n_orig <= 0 or n_orig % 6 != 0:
        raise ValueError(f"Qwen modulation output dimension must be positive and divisible by 6, got {n_orig}")
    dim = n_orig // 6
    if int(weight_scale.shape[1]) != n_orig:
        raise ValueError(f"weight_scale output dimension {int(weight_scale.shape[1])} does not match N={n_orig}")
    if int(weight_zero.shape[1]) != n_orig:
        raise ValueError(f"weight_zero output dimension {int(weight_zero.shape[1])} does not match N={n_orig}")
    out = {
        "weight": weight.view(dim, 6, -1).transpose(0, 1).reshape(n_orig, -1).contiguous(),
        "weight_scale": weight_scale.view(-1, dim, 6).transpose(1, 2).reshape(-1, n_orig).contiguous(),
        "weight_zero": weight_zero.view(-1, dim, 6).transpose(1, 2).reshape(-1, n_orig).contiguous(),
    }
    out_bias = None
    if bias is not None:
        if int(bias.shape[0]) != n_orig:
            raise ValueError(f"bias output dimension {int(bias.shape[0])} does not match N={n_orig}")
        out_bias = bias.view(dim, 6).transpose(0, 1).reshape(n_orig).contiguous()
    return out, out_bias
