"""Runtime-independent reference math for SVDQuant W4A4 tensors.

These helpers execute the tensor contract produced by this package using plain
PyTorch operations.  They are intentionally not a ComfyUI, comfy-kitchen, or
DeepCompressor runtime binding.  The functions are useful for deterministic
unit tests and local oracle comparisons, but they do not by themselves prove
that an external fused runtime has the same layout, zero, activation, or dtype
semantics.
"""

from __future__ import annotations

from typing import Any

from comfy_quants.algorithms.int4_svdquant.weight_quant import dequantize_natural_svdquant_weight
from comfy_quants.formats.kitchen_tilepack import KITCHEN_GROUP_SIZE, unpack_n_axis, unpack_weight_scale, unpack_weight_tile


SVDQUANT_W4A4_REFERENCE_STATE = "repo_reference_math_runtime_unverified"


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("torch is required for SVDQuant reference execution") from exc
    return torch


def _as_tensor(value: Any, *, name: str):
    torch = _require_torch()
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if bool((~torch.isfinite(value.detach().to(dtype=torch.float32))).any().item()):
        raise ValueError(f"{name} contains NaN or Inf values")
    return value


def _natural_weight_and_scale(weight: Any, weight_scale: Any, *, group_size: int):
    if int(group_size) != KITCHEN_GROUP_SIZE:
        raise ValueError(f"SVDQuant W4A4 reference requires group size {KITCHEN_GROUP_SIZE}")
    natural_weight = unpack_weight_tile(weight) if int(weight.ndim) == 4 else weight
    natural_scale = unpack_weight_scale(weight_scale) if int(weight_scale.ndim) == 3 else weight_scale
    return natural_weight, natural_scale


def _natural_proj_up(proj_up: Any):
    return unpack_n_axis(proj_up) if int(proj_up.ndim) >= 3 else proj_up


def dequantize_svdquant_w4a4_effective_weight(
    weight: Any,
    weight_scale: Any,
    *,
    proj_down: Any | None = None,
    proj_up: Any | None = None,
    group_size: int = KITCHEN_GROUP_SIZE,
):
    """Return the dense effective SVDQuant weight under the repo reference math.

    ``weight`` and ``weight_scale`` may be either natural layout or kitchen
    tile-packed layout.  When ``proj_down`` and ``proj_up`` are supplied, the
    returned effective weight is:

    ```text
    dequantized_int4_weight + proj_up @ proj_down.T
    ```

    This is the execution order assumed by the current static artifact writer.
    It is still marked runtime-unverified until checked against a compatible
    fused external runtime.
    """
    torch = _require_torch()
    weight = _as_tensor(weight, name="weight")
    weight_scale = _as_tensor(weight_scale, name="weight_scale")
    natural_weight, natural_scale = _natural_weight_and_scale(weight, weight_scale, group_size=int(group_size))
    dense = dequantize_natural_svdquant_weight(natural_weight, natural_scale, group_size=int(group_size)).to(dtype=torch.float32)
    if (proj_down is None) != (proj_up is None):
        raise ValueError("proj_down and proj_up must be provided together")
    if proj_down is None:
        return dense

    proj_down = _as_tensor(proj_down, name="proj_down").to(device=dense.device, dtype=torch.float32)
    proj_up = _natural_proj_up(_as_tensor(proj_up, name="proj_up")).to(device=dense.device, dtype=torch.float32)
    if int(proj_down.ndim) != 2:
        raise ValueError(f"proj_down must have shape (K, R), got {tuple(proj_down.shape)}")
    if int(proj_up.ndim) != 2:
        raise ValueError(f"proj_up must have shape (N, R), got {tuple(proj_up.shape)}")
    n, k = int(dense.shape[0]), int(dense.shape[1])
    if tuple(int(x) for x in proj_down.shape) != (k, int(proj_up.shape[1])):
        raise ValueError(
            f"proj_down shape {tuple(proj_down.shape)} is incompatible with dense weight shape {(n, k)} "
            f"and proj_up shape {tuple(proj_up.shape)}"
        )
    if int(proj_up.shape[0]) != n:
        raise ValueError(f"proj_up N={int(proj_up.shape[0])} does not match weight N={n}")
    return (dense + proj_up.matmul(proj_down.t())).contiguous()


def reference_svdquant_w4a4_linear(
    inputs: Any,
    weight: Any,
    weight_scale: Any,
    smooth_factor: Any,
    proj_down: Any,
    proj_up: Any,
    *,
    bias: Any | None = None,
    group_size: int = KITCHEN_GROUP_SIZE,
):
    """Execute one SVDQuant W4A4 linear layer with plain PyTorch math.

    The repo reference formula is:

    ```text
    y = (x / smooth_factor) @ (dequant(weight, weight_scale) + proj_up @ proj_down.T).T + bias
    ```

    ``weight`` / ``weight_scale`` / ``proj_up`` may be natural tensors or the
    kitchen tile-packed tensors stored in a checkpoint.  The helper deliberately
    does not model target-runtime dynamic activation W4 kernels; that oracle
    remains a separate validation gate.
    """
    torch = _require_torch()
    x = _as_tensor(inputs, name="inputs").to(dtype=torch.float32)
    if int(x.ndim) == 0:
        raise ValueError("inputs must have at least one dimension")
    k = int(x.shape[-1])
    smooth = _as_tensor(smooth_factor, name="smooth_factor").to(device=x.device, dtype=torch.float32).reshape(-1)
    if int(smooth.numel()) != k:
        raise ValueError(f"smooth_factor length {int(smooth.numel())} does not match input K={k}")
    if bool((smooth == 0).any().item()):
        raise ValueError("smooth_factor must not contain zero values")
    effective_weight = dequantize_svdquant_w4a4_effective_weight(
        weight,
        weight_scale,
        proj_down=proj_down,
        proj_up=proj_up,
        group_size=int(group_size),
    ).to(device=x.device, dtype=torch.float32)
    if int(effective_weight.shape[1]) != k:
        raise ValueError(f"effective weight K={int(effective_weight.shape[1])} does not match input K={k}")
    output = torch.matmul(x / smooth.reshape(*([1] * (int(x.ndim) - 1)), k), effective_weight.t())
    if bias is not None:
        bias_tensor = _as_tensor(bias, name="bias").to(device=output.device, dtype=torch.float32).reshape(-1)
        if int(bias_tensor.numel()) != int(effective_weight.shape[0]):
            raise ValueError(f"bias length {int(bias_tensor.numel())} does not match output N={int(effective_weight.shape[0])}")
        output = output + bias_tensor.reshape(*([1] * (int(output.ndim) - 1)), int(bias_tensor.numel()))
    return output.contiguous()
