"""Group-wise regular-Hadamard rotation (ConvRot) for INT8 weight quantization.

Offline half of the ConvRot scheme used by the downstream ComfyUI-INT8-Fast
runtime: rotate each weight row block by a normalized *regular* Hadamard matrix
before per-row INT8 quantization, spreading channel outliers so the int8 grid is
used more evenly. The matching online activation rotation lives in the runtime
(out of scope here).

Bit-faithful to ComfyUI-INT8-Fast ``convrot.py`` (QuaRot 2024 / ConvRot 2025):
the regular H4 has no all-ones column, and larger matrices are the Kronecker
power of H4 normalized by ``1/sqrt(size)``. Pure-torch (``torch.kron``); the
upstream ``scipy`` import is vestigial, so this module needs only torch.
"""

from __future__ import annotations

import math
from typing import Any

from comfy_quants.core.errors import PayloadWriteError

# Cache by (size, device-str, dtype) to avoid recomputation, mirroring upstream.
_HADAMARD_CACHE: dict[tuple[int, str, Any], Any] = {}

# Default ConvRot group size (must match ComfyUI-INT8-Fast CONVROT_GROUP_SIZE).
CONVROT_GROUP_SIZE = 256


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise PayloadWriteError("torch is required for ConvRot weight rotation") from exc
    return torch


def is_power_of_four(size: int) -> bool:
    return size >= 4 and (size & (size - 1)) == 0 and math.log(size, 4) % 1 == 0


def build_hadamard(size: int, *, device: str | Any = "cpu", dtype: Any = None):
    """Return a normalized REGULAR orthogonal Hadamard matrix of shape (size, size).

    ``size`` must be a power of four (4, 16, 64, 256, ...). Built as the Kronecker
    power of the regular H4 and normalized by ``1/sqrt(size)`` (so it is symmetric
    and orthogonal: ``H @ H == I``). Cached by (size, device, dtype).
    """
    torch = _require_torch()
    if dtype is None:
        dtype = torch.float32
    if not is_power_of_four(size):
        raise PayloadWriteError(f"ConvRot Hadamard size must be a power of four, got {size}")

    cache_key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4

    h_normalized = h / (size ** 0.5)
    _HADAMARD_CACHE[cache_key] = h_normalized
    return h_normalized


def rotate_weight(weight, hadamard, group_size: int):
    """Rotate a weight matrix offline: ``W_rot = (W grouped) @ H.T``.

    For ``Linear(in, out)`` weight of shape ``(out, in)``, each row is split into
    blocks of ``group_size`` along the input dimension and rotated by ``H.T``.
    Returns a tensor of the same shape.
    """
    torch = _require_torch()
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise PayloadWriteError(f"in_features {in_f} not divisible by ConvRot group_size {group_size}")
    n_groups = in_f // group_size
    grouped = weight.view(out_f, n_groups, group_size)
    h_t = hadamard.T.to(dtype=weight.dtype, device=weight.device)
    rotated = torch.matmul(grouped, h_t)
    return rotated.reshape(out_f, in_f)


def rotate_activation(x, hadamard, group_size: int):
    """Rotate activations online: ``x_rot = (x grouped) @ H``.

    Provided for parity tests / reference; the production online rotation lives in
    the downstream runtime. Last dim must be divisible by ``group_size``.
    """
    torch = _require_torch()
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise PayloadWriteError(f"features {features} not divisible by ConvRot group_size {group_size}")
    n_groups = features // group_size
    grouped = x.view(*orig_shape[:-1], n_groups, group_size)
    h_dev = hadamard.to(dtype=x.dtype, device=x.device)
    return torch.matmul(grouped, h_dev).view(orig_shape)
