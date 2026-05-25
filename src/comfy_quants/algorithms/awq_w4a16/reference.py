"""Runtime-independent reference math for AWQ W4A16 tensors."""

from __future__ import annotations

from typing import Any

from comfy_quants.algorithms.awq_w4a16.weight_quant import dequantize_awq_w4a16_weight
from comfy_quants.formats.awq_w4a16 import AWQ_W4A16_GROUP_SIZE


AWQ_W4A16_REFERENCE_STATE = "kitchen_awq_w4a16_reference_math_runtime_unverified"


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("torch is required for AWQ W4A16 reference execution") from exc
    return torch


def _as_tensor(value: Any, *, name: str):
    torch = _require_torch()
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if bool((~torch.isfinite(value.detach().to(dtype=torch.float32))).any().item()):
        raise ValueError(f"{name} contains NaN or Inf values")
    return value


def reference_awq_w4a16_linear(
    inputs: Any,
    weight: Any,
    weight_scale: Any,
    weight_zero: Any,
    *,
    bias: Any | None = None,
    group_size: int = AWQ_W4A16_GROUP_SIZE,
):
    """Execute one AWQ W4A16 linear layer with the kitchen-native formula.

    The reference formula is:

    ```text
    dense_weight = (uint4_weight - 8) * weight_scale + weight_zero
    y = x @ dense_weight.T + bias
    ```

    This validates the package's checkpoint tensor contract with plain PyTorch.
    Exact parity with a target fused external AWQ runtime remains a separate
    validation gate until that runtime branch is available for full inference.
    """
    torch = _require_torch()
    x = _as_tensor(inputs, name="inputs").to(dtype=torch.float32)
    if int(x.ndim) == 0:
        raise ValueError("inputs must have at least one dimension")
    dense_weight = dequantize_awq_w4a16_weight(weight, weight_scale, weight_zero, group_size=int(group_size)).to(
        device=x.device,
        dtype=torch.float32,
    )
    if int(x.shape[-1]) != int(dense_weight.shape[1]):
        raise ValueError(f"input K={int(x.shape[-1])} does not match AWQ weight K={int(dense_weight.shape[1])}")
    output = torch.matmul(x, dense_weight.t())
    if bias is not None:
        bias_tensor = _as_tensor(bias, name="bias").to(device=output.device, dtype=torch.float32).reshape(-1)
        if int(bias_tensor.numel()) != int(dense_weight.shape[0]):
            raise ValueError(f"bias length {int(bias_tensor.numel())} does not match output N={int(dense_weight.shape[0])}")
        output = output + bias_tensor.reshape(*([1] * (int(output.ndim) - 1)), int(bias_tensor.numel()))
    return output.contiguous()
