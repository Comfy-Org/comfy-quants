"""Low-rank branch basis transforms for SVDQuant W4A4 artifacts.

These helpers only transform already-solved low-rank tensors between explicit
artifact bases.  They do not import or depend on any model runtime.
"""

from __future__ import annotations

from typing import Any


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("torch is required for SVDQuant low-rank branch basis transforms") from exc
    return torch


def _as_proj_down_tensor(value: Any):
    torch = _require_torch()
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if int(tensor.ndim) != 2:
        raise ValueError(f"proj_down must have shape (K, R), got {tuple(tensor.shape)}")
    if int(tensor.shape[0]) <= 0:
        raise ValueError(f"proj_down input dimension K must be positive, got {int(tensor.shape[0])}")
    if int(tensor.shape[1]) < 0:
        raise ValueError(f"proj_down rank must be non-negative, got {int(tensor.shape[1])}")
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    if bool((~torch.isfinite(tensor.detach())).any().item()):
        raise ValueError("proj_down contains NaN or Inf values")
    return tensor


def _as_smooth_vector(value: Any, *, k: int, device: Any):
    torch = _require_torch()
    smooth = value if torch.is_tensor(value) else torch.as_tensor(value)
    smooth = smooth.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if int(smooth.numel()) != int(k):
        raise ValueError(f"smooth_factor length {int(smooth.numel())} does not match proj_down K={int(k)}")
    if bool((~torch.isfinite(smooth)).any().item()):
        raise ValueError("smooth_factor contains NaN or Inf values")
    if bool((smooth == 0).any().item()):
        raise ValueError("smooth_factor must not contain zero values")
    return smooth


def _output_dtype_for_proj_down(proj_down: Any, torch: Any):
    if hasattr(proj_down, "dtype") and proj_down.dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        return proj_down.dtype
    return torch.float32


def fold_proj_down_for_raw_branch(
    proj_down_post_smoothing: Any,
    smooth_factor: Any,
):
    """Return a raw-input-basis ``proj_down`` tensor.

    Some solver artifacts describe the low-rank branch in the post-smoothing
    basis:

    ```text
    branch = (x / smooth_factor) @ proj_down_post_smoothing @ proj_up.T
    ```

    The target Kitchen/Nunchaku runtime contract computes the low-rank down
    projection from the raw activation, so exported artifacts default to the
    equivalent raw-input basis:

    ```text
    branch = x @ proj_down_raw @ proj_up.T
    proj_down_raw = proj_down_post_smoothing / smooth_factor[:, None]
    ```

    The result keeps the floating dtype of ``proj_down_post_smoothing`` while
    performing the divisor math in float32.
    """

    torch = _require_torch()
    down = _as_proj_down_tensor(proj_down_post_smoothing)
    smooth = _as_smooth_vector(smooth_factor, k=int(down.shape[0]), device=down.device)
    out_dtype = _output_dtype_for_proj_down(down, torch)
    folded = down.detach().to(dtype=torch.float32) / smooth.reshape(int(down.shape[0]), 1)
    if bool((~torch.isfinite(folded)).any().item()):
        raise ValueError("folded proj_down contains NaN or Inf values")
    return folded.to(dtype=out_dtype).contiguous()


def unfold_proj_down_for_post_smoothing_branch(
    proj_down_raw: Any,
    smooth_factor: Any,
):
    """Return a post-smoothing-basis ``proj_down`` tensor from raw basis.

    This is the inverse of :func:`fold_proj_down_for_raw_branch`:

    ```text
    proj_down_post_smoothing = proj_down_raw * smooth_factor[:, None]
    ```
    """

    torch = _require_torch()
    down = _as_proj_down_tensor(proj_down_raw)
    smooth = _as_smooth_vector(smooth_factor, k=int(down.shape[0]), device=down.device)
    out_dtype = _output_dtype_for_proj_down(down, torch)
    unfolded = down.detach().to(dtype=torch.float32) * smooth.reshape(int(down.shape[0]), 1)
    if bool((~torch.isfinite(unfolded)).any().item()):
        raise ValueError("unfolded proj_down contains NaN or Inf values")
    return unfolded.to(dtype=out_dtype).contiguous()
