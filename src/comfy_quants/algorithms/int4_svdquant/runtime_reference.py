"""Runtime-like PyTorch oracles for experimental SVDQuant W4A4 layers.

The helpers in this module model the layer math that a fused W4A4 runtime is
expected to execute: activation W4 quantization, W4 weight dequantization, and
the separate low-rank branch.  They are local test oracles only.  They do not
import, wrap, or validate against any external runtime package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from comfy_quants.algorithms.int4_svdquant.reference import dequantize_svdquant_w4a4_effective_weight
from comfy_quants.formats.int4_common import pack_signed_int4_pairs, pack_uint4_pairs
from comfy_quants.formats.kitchen_tilepack import KITCHEN_GROUP_SIZE, unpack_n_axis

ActivationSignedness = Literal["signed", "unsigned"]
LowRankBranchInputBasis = Literal["raw", "post_smoothing"]

SVDQUANT_W4A4_RUNTIME_REFERENCE_STATE = "repo_runtime_like_activation_w4_branch_oracle_runtime_unverified"
GELU_UNSIGNED_SHIFT = 0.171875


@dataclass(frozen=True)
class ActivationW4Quantization:
    """Natural-layout activation W4 quantization result."""

    q_values: Any
    packed: Any
    scale: Any
    dequantized: Any
    signedness: ActivationSignedness


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("torch is required for SVDQuant runtime reference execution") from exc
    return torch


def _as_float_tensor(value: Any, *, name: str, device: Any | None = None):
    torch = _require_torch()
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if device is not None:
        value = value.to(device=device)
    value = value.to(dtype=torch.float32)
    if bool((~torch.isfinite(value.detach())).any().item()):
        raise ValueError(f"{name} contains NaN or Inf values")
    return value


def _validate_activation_shape(inputs: Any, *, group_size: int) -> tuple[int, int]:
    if int(inputs.ndim) == 0:
        raise ValueError("activation inputs must have at least one dimension")
    k = int(inputs.shape[-1])
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if k <= 0:
        raise ValueError(f"activation input K must be positive, got {k}")
    if k % group_size != 0:
        raise ValueError(f"activation input K={k} is not divisible by group size {group_size}")
    if k % 2 != 0:
        raise ValueError(f"activation input K={k} must be even for INT4 pair packing")
    return k, k // group_size


def _quantize_activation_w4(
    inputs: Any,
    *,
    group_size: int,
    signedness: ActivationSignedness,
) -> ActivationW4Quantization:
    torch = _require_torch()
    x = _as_float_tensor(inputs, name="inputs")
    k, groups = _validate_activation_shape(x, group_size=int(group_size))
    grouped = x.contiguous().view(*x.shape[:-1], groups, int(group_size))
    amax = grouped.abs().amax(dim=-1)
    if signedness == "signed":
        scale = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
        q_values = torch.round(grouped / scale.unsqueeze(-1)).clamp(-7, 7).to(torch.int8).view(*x.shape).contiguous()
        packed = pack_signed_int4_pairs(q_values, validate=False)
    elif signedness == "unsigned":
        scale = torch.where(amax > 0, amax / 15.0, torch.ones_like(amax))
        q_values = torch.round(grouped / scale.unsqueeze(-1)).clamp(0, 15).to(torch.int8).view(*x.shape).contiguous()
        packed = pack_uint4_pairs(q_values, validate=False)
    else:
        raise ValueError(f"unsupported activation signedness: {signedness}")
    dequantized = (q_values.to(dtype=torch.float32).view(*x.shape[:-1], groups, int(group_size)) * scale.unsqueeze(-1)).view(
        *x.shape[:-1],
        k,
    )
    return ActivationW4Quantization(
        q_values=q_values,
        packed=packed,
        scale=scale.contiguous(),
        dequantized=dequantized.contiguous(),
        signedness=signedness,
    )


def quantize_activation_w4_signed(
    inputs: Any,
    *,
    group_size: int = KITCHEN_GROUP_SIZE,
) -> ActivationW4Quantization:
    """Quantize activations with a signed W4 absmax oracle.

    The natural reference scale is stored as ``inputs.shape[:-1] + (K/G,)`` and
    uses ``absmax / 7`` for non-zero groups.  Zero groups use scale ``1``.
    """

    return _quantize_activation_w4(inputs, group_size=int(group_size), signedness="signed")


def quantize_activation_w4_unsigned(
    inputs: Any,
    *,
    group_size: int = KITCHEN_GROUP_SIZE,
) -> ActivationW4Quantization:
    """Quantize activations with an unsigned W4 absmax oracle.

    This helper deliberately models unsigned saturation.  Negative values map to
    code ``0`` unless a caller applies a runtime-compatible shift before calling
    the oracle.
    """

    return _quantize_activation_w4(inputs, group_size=int(group_size), signedness="unsigned")


def _smooth_inputs(inputs: Any, smooth_factor: Any):
    x = _as_float_tensor(inputs, name="inputs")
    if int(x.ndim) == 0:
        raise ValueError("inputs must have at least one dimension")
    k = int(x.shape[-1])
    smooth = _as_float_tensor(smooth_factor, name="smooth_factor", device=x.device).reshape(-1)
    if int(smooth.numel()) != k:
        raise ValueError(f"smooth_factor length {int(smooth.numel())} does not match input K={k}")
    if bool((smooth == 0).any().item()):
        raise ValueError("smooth_factor must not contain zero values")
    return x, smooth, x / smooth.reshape(*([1] * (int(x.ndim) - 1)), k)


def _main_path_inputs(
    raw_inputs: Any,
    smooth: Any,
    *,
    activation_signedness: ActivationSignedness,
    apply_unsigned_activation_shift: bool,
    unsigned_activation_shift: float,
):
    if activation_signedness == "unsigned" and bool(apply_unsigned_activation_shift):
        main_inputs = raw_inputs + float(unsigned_activation_shift)
    elif activation_signedness == "unsigned" or activation_signedness == "signed":
        main_inputs = raw_inputs
    else:
        raise ValueError(f"unsupported activation signedness: {activation_signedness}")
    k = int(raw_inputs.shape[-1])
    smooth_view = smooth.reshape(*([1] * (int(raw_inputs.ndim) - 1)), k)
    return main_inputs.contiguous(), (main_inputs / smooth_view).contiguous()


def _lowrank_branch(
    raw_inputs: Any,
    post_smoothing_inputs: Any,
    *,
    proj_down: Any | None,
    proj_up: Any | None,
    branch_input_basis: LowRankBranchInputBasis,
):
    torch = _require_torch()
    if (proj_down is None) != (proj_up is None):
        raise ValueError("proj_down and proj_up must be provided together")
    if proj_down is None:
        return None

    down = _as_float_tensor(proj_down, name="proj_down", device=raw_inputs.device)
    up_tensor = _as_float_tensor(proj_up, name="proj_up", device=raw_inputs.device)
    up = unpack_n_axis(up_tensor) if int(up_tensor.ndim) >= 3 else up_tensor
    if int(down.ndim) != 2:
        raise ValueError(f"proj_down must have shape (K, R), got {tuple(down.shape)}")
    if int(up.ndim) != 2:
        raise ValueError(f"proj_up must have shape (N, R), got {tuple(up.shape)}")
    k = int(raw_inputs.shape[-1])
    if int(down.shape[0]) != k:
        raise ValueError(f"proj_down K={int(down.shape[0])} does not match input K={k}")
    if int(down.shape[1]) != int(up.shape[1]):
        raise ValueError(f"proj_down rank {int(down.shape[1])} does not match proj_up rank {int(up.shape[1])}")

    if branch_input_basis == "raw":
        branch_inputs = raw_inputs
    elif branch_input_basis == "post_smoothing":
        branch_inputs = post_smoothing_inputs
    else:
        raise ValueError(f"unsupported low-rank branch input basis: {branch_input_basis}")

    branch = torch.matmul(torch.matmul(branch_inputs, down), up.t())
    return branch.contiguous()


def reference_svdquant_w4a4_linear_runtime(
    inputs: Any,
    weight: Any,
    weight_scale: Any,
    smooth_factor: Any,
    proj_down: Any | None = None,
    proj_up: Any | None = None,
    *,
    bias: Any | None = None,
    group_size: int = KITCHEN_GROUP_SIZE,
    activation_signedness: ActivationSignedness = "signed",
    branch_input_basis: LowRankBranchInputBasis = "raw",
    apply_unsigned_activation_shift: bool = True,
    unsigned_activation_shift: float = GELU_UNSIGNED_SHIFT,
):
    """Execute a runtime-like SVDQuant W4A4 linear oracle.

    The main path is:

    ```text
    x_main = x
    if act_unsigned:
        x_main = x + 0.171875
    x_quant = x_main / smooth_factor
    qx, ascales = activation_w4_quantize(x_quant)
    main = dequant(qx, ascales) @ dequant(weight, weight_scale).T
    ```

    The unsigned activation shift follows the Qwen/GEGLU SVDQuant runtime
    contract used by the target W4A4 kernels: the main activation quantizer sees
    ``x + 0.171875`` for unsigned layers, but the low-rank branch remains
    mathematically defined on the unshifted activation.  The branch input basis
    is explicit because runtimes may compute it from raw activations with a
    smooth-folded ``proj_down`` tensor, while dense math references often
    describe it in the post-smoothing basis:

    ```text
    raw basis:            branch = x @ proj_down @ proj_up.T
    post_smoothing basis: branch = (x / smooth_factor) @ proj_down @ proj_up.T
    ```

    The helper is intentionally marked runtime-unverified.  It is a local oracle
    for closing layer math before comparing against an external fused runtime.
    """

    torch = _require_torch()
    raw_inputs, smooth, post_smoothing_inputs = _smooth_inputs(inputs, smooth_factor)
    _main_inputs, main_post_smoothing_inputs = _main_path_inputs(
        raw_inputs,
        smooth,
        activation_signedness=activation_signedness,
        apply_unsigned_activation_shift=bool(apply_unsigned_activation_shift),
        unsigned_activation_shift=float(unsigned_activation_shift),
    )
    activation_quant = _quantize_activation_w4(
        main_post_smoothing_inputs,
        group_size=int(group_size),
        signedness=activation_signedness,
    )
    dense_weight = dequantize_svdquant_w4a4_effective_weight(
        weight,
        weight_scale,
        group_size=int(group_size),
    ).to(device=raw_inputs.device, dtype=torch.float32)
    if int(dense_weight.shape[1]) != int(raw_inputs.shape[-1]):
        raise ValueError(f"dequantized weight K={int(dense_weight.shape[1])} does not match input K={int(raw_inputs.shape[-1])}")

    output = torch.matmul(activation_quant.dequantized, dense_weight.t())
    branch = _lowrank_branch(
        raw_inputs,
        post_smoothing_inputs,
        proj_down=proj_down,
        proj_up=proj_up,
        branch_input_basis=branch_input_basis,
    )
    if branch is not None:
        if int(branch.shape[-1]) != int(dense_weight.shape[0]):
            raise ValueError(f"low-rank branch N={int(branch.shape[-1])} does not match weight N={int(dense_weight.shape[0])}")
        output = output + branch

    if bias is not None:
        bias_tensor = _as_float_tensor(bias, name="bias", device=output.device).reshape(-1)
        if int(bias_tensor.numel()) != int(dense_weight.shape[0]):
            raise ValueError(f"bias length {int(bias_tensor.numel())} does not match output N={int(dense_weight.shape[0])}")
        output = output + bias_tensor.reshape(*([1] * (int(output.ndim) - 1)), int(bias_tensor.numel()))
    return output.contiguous()
