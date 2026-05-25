"""Weight quantization helpers for natural SVDQuant W4A4 tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from comfy_quants.algorithms.int4_svdquant.gptq import (
    GptqConfig,
    GptqHessianStats,
    build_gptq_hessian_from_activations,
    quantize_linear_weight_grouped_signed_int4_gptq,
    transform_gptq_hessian_input_basis,
)
from comfy_quants.algorithms.int4_svdquant.lowrank import (
    solve_lowrank_output_error_branch,
    solve_lowrank_residual_branch,
)
from comfy_quants.algorithms.int4_svdquant.smoothing import solve_smooth_factor
from comfy_quants.algorithms.int4_svdquant.stats import ActivationStats
from comfy_quants.formats.int4_common import pack_signed_int4_pairs, unpack_signed_int4_pairs
from comfy_quants.formats.kitchen_tilepack import KITCHEN_GROUP_SIZE


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("torch is required for INT4 SVDQuant weight quantization") from exc
    return torch


@dataclass(frozen=True)
class NaturalSvdquantLinearTensors:
    """Natural-layout tensors for one SVDQuant W4A4 linear layer."""

    weight: Any
    weight_scale: Any
    smooth_factor: Any
    proj_down: Any
    proj_up: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight,
            "weight_scale": self.weight_scale,
            "smooth_factor": self.smooth_factor,
            "proj_down": self.proj_down,
            "proj_up": self.proj_up,
        }


@dataclass(frozen=True)
class GroupedInt4WeightQuantization:
    """Intermediate groupwise signed-INT4 quantization result."""

    packed_weight: Any
    weight_scale: Any
    quantized_weight: Any
    dequantized_weight: Any


LowRankCalibrationMode = Literal["weight_residual", "output_error"]


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
    return n, k


def _materialize_activation_samples(torch: Any, activation_samples: Any | None) -> list[Any] | None:
    if activation_samples is None:
        return None
    if torch.is_tensor(activation_samples):
        return [activation_samples]
    if isinstance(activation_samples, dict | str | bytes):
        raise ValueError("activation_samples must be a tensor or an iterable of tensors")
    return list(activation_samples)


def _activation_samples_for_hessian(torch: Any, activation_samples: Any | None):
    if activation_samples is None:
        return None
    if torch.is_tensor(activation_samples):
        return [activation_samples]
    if isinstance(activation_samples, dict | str | bytes):
        raise ValueError("activation_samples must be a tensor or an iterable of tensors")
    return activation_samples


def _iter_output_error_samples_for_weight_residual(
    activation_samples: list[Any],
    *,
    weight_residual: Any,
    smooth_factor: Any,
    channel_dim: int,
):
    torch = _require_torch()
    residual = weight_residual.detach().to(dtype=torch.float32)
    smooth = smooth_factor.detach().to(device=residual.device, dtype=torch.float32).reshape(1, -1)
    k = int(residual.shape[1])
    for sample in activation_samples:
        if not torch.is_tensor(sample):
            sample = torch.as_tensor(sample)
        if int(sample.ndim) == 0:
            raise ValueError("activation sample must have at least one dimension")
        dim = int(channel_dim)
        if dim < 0:
            dim += int(sample.ndim)
        if dim < 0 or dim >= int(sample.ndim):
            raise ValueError(f"activation_channel_dim {channel_dim} is out of range for shape {tuple(sample.shape)}")
        tensor = sample.detach()
        if dim != int(tensor.ndim) - 1:
            tensor = tensor.movedim(dim, -1)
        if int(tensor.shape[-1]) != k:
            raise ValueError(f"activation channel count {int(tensor.shape[-1])} does not match weight input dim {k}")
        rows = tensor.reshape(-1, k).to(device=residual.device, dtype=torch.float32)
        output_rows = (rows / smooth).matmul(residual.t()).contiguous()
        yield output_rows.reshape(*tuple(int(x) for x in tensor.shape[:-1]), int(residual.shape[0]))


def quantize_linear_weight_grouped_signed_int4(
    weight: Any,
    *,
    group_size: int = KITCHEN_GROUP_SIZE,
    scale_dtype: str = "source",
) -> GroupedInt4WeightQuantization:
    """Quantize a dense linear weight with per-output-row/per-K-group scales."""
    torch = _require_torch()
    group_size = int(group_size)
    n, k = _validate_linear_weight_shape(weight, group_size=group_size)
    source_dtype = weight.dtype
    out_dtype = _resolve_scale_dtype(torch, source_dtype, scale_dtype)
    groups = k // group_size
    working = weight.detach().to(dtype=torch.float32).contiguous().view(n, groups, group_size)
    amax = working.abs().amax(dim=2)
    scale = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
    quant = torch.round(working / scale.unsqueeze(-1)).clamp(-7, 7).to(torch.int8).contiguous()
    dequant = (quant.to(dtype=torch.float32) * scale.unsqueeze(-1)).view(n, k).contiguous()
    return GroupedInt4WeightQuantization(
        packed_weight=pack_signed_int4_pairs(quant.view(n, k), validate=False),
        weight_scale=scale.t().contiguous().to(dtype=out_dtype),
        quantized_weight=quant.view(n, k).contiguous(),
        dequantized_weight=dequant,
    )


def dequantize_natural_svdquant_weight(
    packed_weight: Any,
    weight_scale: Any,
    *,
    group_size: int = KITCHEN_GROUP_SIZE,
):
    """Dequantize natural-layout signed INT4 weight bytes using natural scales."""
    torch = _require_torch()
    if int(weight_scale.ndim) != 2:
        raise ValueError(f"weight_scale must have shape (K/{group_size}, N), got {tuple(weight_scale.shape)}")
    quant = unpack_signed_int4_pairs(packed_weight).to(dtype=torch.float32)
    n, k = _validate_linear_weight_shape(quant, group_size=int(group_size))
    groups = k // int(group_size)
    if tuple(int(x) for x in weight_scale.shape) != (groups, n):
        raise ValueError(f"weight_scale shape {tuple(weight_scale.shape)} does not match quantized weight shape {(n, k)}")
    scale = weight_scale.transpose(0, 1).to(device=quant.device, dtype=torch.float32).reshape(n, groups, 1)
    return (quant.view(n, groups, int(group_size)) * scale).view(n, k).contiguous()


def quantize_linear_weight_to_natural_svdquant(
    weight: Any,
    *,
    rank: int,
    group_size: int = KITCHEN_GROUP_SIZE,
    scale_dtype: str = "source",
) -> NaturalSvdquantLinearTensors:
    """Quantize one dense linear weight to natural SVDQuant W4A4 tensors.

    This calibration-free mode emits identity smoothing and a zero low-rank
    branch while preserving the same tensor contract used by calibrated solvers.
    """
    torch = _require_torch()
    n, k = _validate_linear_weight_shape(weight, group_size=int(group_size))
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")

    out_dtype = _resolve_scale_dtype(torch, weight.dtype, scale_dtype)
    quantized = quantize_linear_weight_grouped_signed_int4(weight, group_size=group_size, scale_dtype=scale_dtype)
    return NaturalSvdquantLinearTensors(
        weight=quantized.packed_weight,
        weight_scale=quantized.weight_scale,
        smooth_factor=torch.ones((k,), dtype=out_dtype, device=weight.device),
        proj_down=torch.zeros((k, rank), dtype=out_dtype, device=weight.device),
        proj_up=torch.zeros((n, rank), dtype=out_dtype, device=weight.device),
    )


def quantize_linear_weight_to_calibrated_natural_svdquant(
    weight: Any,
    *,
    activation_stats: ActivationStats | Any,
    rank: int,
    group_size: int = KITCHEN_GROUP_SIZE,
    scale_dtype: str = "source",
    smooth_alpha: float = 0.5,
    smooth_min: float = 1.0 / 64.0,
    smooth_max: float = 64.0,
) -> NaturalSvdquantLinearTensors:
    """Build experimental natural SVDQuant tensors from activation statistics.

    This implementation performs smoothing, groupwise round-to-nearest INT4
    weight quantization, and residual SVD.  It intentionally does not claim to
    be the publishable SVDQuant W4A4 path because that path also requires the
    post-smoothing GPTQ/Hessian solve and the full mixed runtime contract.
    """
    torch = _require_torch()
    _n, _k = _validate_linear_weight_shape(weight, group_size=int(group_size))
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")

    out_dtype = _resolve_scale_dtype(torch, weight.dtype, scale_dtype)
    activation_amax = activation_stats.input_amax if isinstance(activation_stats, ActivationStats) else activation_stats
    smooth = solve_smooth_factor(
        weight,
        activation_amax,
        alpha=smooth_alpha,
        min_value=smooth_min,
        max_value=smooth_max,
        output_dtype=out_dtype,
    )
    quantized = quantize_linear_weight_grouped_signed_int4(
        smooth.smoothed_weight,
        group_size=group_size,
        scale_dtype=scale_dtype,
    )
    residual = smooth.smoothed_weight.detach().to(dtype=torch.float32) - quantized.dequantized_weight
    branch = solve_lowrank_residual_branch(residual, rank=rank, dtype=out_dtype)
    return NaturalSvdquantLinearTensors(
        weight=quantized.packed_weight,
        weight_scale=quantized.weight_scale,
        smooth_factor=smooth.smooth_factor,
        proj_down=branch.proj_down,
        proj_up=branch.proj_up,
    )


def quantize_linear_weight_to_gptq_natural_svdquant(
    weight: Any,
    *,
    activation_stats: ActivationStats | Any,
    activation_samples: Any | None = None,
    gptq_hessian: GptqHessianStats | Any | None = None,
    rank: int,
    group_size: int = KITCHEN_GROUP_SIZE,
    scale_dtype: str = "source",
    smooth_alpha: float = 0.5,
    smooth_min: float = 1.0 / 64.0,
    smooth_max: float = 64.0,
    activation_channel_dim: int = -1,
    gptq_config: GptqConfig | None = None,
    gptq_hessian_input_basis: str = "post_smoothing",
    lowrank_calibration: LowRankCalibrationMode = "weight_residual",
    lowrank_ridge: float = 1.0e-6,
) -> NaturalSvdquantLinearTensors:
    """Build natural SVDQuant tensors using post-smoothing GPTQ.

    ``activation_samples`` are interpreted as the original dense linear inputs;
    the function divides them by the solved ``smooth_factor`` before building
    the Hessian.  If ``gptq_hessian`` is supplied in the default
    ``post_smoothing`` basis, it is consumed as-is.  If it is supplied in the
    ``raw_activation`` basis, the function applies the same smoothing divisor
    before the GPTQ solve.

    By default the low-rank branch is initialized from a weight-space RTN
    residual and is subtracted before the GPTQ solve.  Passing
    ``lowrank_calibration="output_error"`` instead fits the branch against the
    RTN output residual on ``activation_samples`` before subtracting the branch
    effective weight and running GPTQ.  A precomputed Hessian alone is
    insufficient for that output-error branch solve; activation samples are
    still required.
    """
    torch = _require_torch()
    _n, _k = _validate_linear_weight_shape(weight, group_size=int(group_size))
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if lowrank_calibration not in {"weight_residual", "output_error"}:
        raise ValueError(f"unsupported low-rank calibration mode: {lowrank_calibration}")
    if activation_samples is None and gptq_hessian is None:
        raise ValueError("GPTQ SVDQuant requires activation_samples or a precomputed gptq_hessian")
    if lowrank_calibration == "output_error" and activation_samples is None:
        raise ValueError("output-error low-rank calibration requires activation_samples; a GPTQ Hessian alone is insufficient")
    if gptq_hessian_input_basis not in {"post_smoothing", "raw_activation"}:
        raise ValueError(f"unsupported GPTQ Hessian input basis: {gptq_hessian_input_basis}")

    out_dtype = _resolve_scale_dtype(torch, weight.dtype, scale_dtype)
    gptq_cfg = gptq_config or GptqConfig()
    materialized_samples = None
    hessian_samples = None
    if lowrank_calibration == "output_error":
        materialized_samples = _materialize_activation_samples(torch, activation_samples)
        hessian_samples = materialized_samples
    else:
        hessian_samples = _activation_samples_for_hessian(torch, activation_samples)
    activation_amax = activation_stats.input_amax if isinstance(activation_stats, ActivationStats) else activation_stats
    smooth = solve_smooth_factor(
        weight,
        activation_amax,
        alpha=smooth_alpha,
        min_value=smooth_min,
        max_value=smooth_max,
        output_dtype=out_dtype,
    )
    hessian = gptq_hessian
    if hessian is None:
        hessian = build_gptq_hessian_from_activations(
            hessian_samples,
            channel_dim=activation_channel_dim,
            input_channel_divisor=smooth.smooth_factor,
            hessian_block_size=gptq_cfg.hessian_block_size,
            device=weight.device,
            dtype=torch.float32,
        )
    elif gptq_hessian_input_basis == "raw_activation":
        hessian = transform_gptq_hessian_input_basis(hessian, input_channel_divisor=smooth.smooth_factor)
    branch_seed_quantized = quantize_linear_weight_grouped_signed_int4(
        smooth.smoothed_weight,
        group_size=group_size,
        scale_dtype=scale_dtype,
    )
    branch_seed_residual = smooth.smoothed_weight.detach().to(dtype=torch.float32) - branch_seed_quantized.dequantized_weight
    if lowrank_calibration == "output_error":
        if materialized_samples is None:
            raise ValueError("output-error low-rank calibration requires activation_samples")
        output_residual = _iter_output_error_samples_for_weight_residual(
            materialized_samples,
            weight_residual=branch_seed_residual,
            smooth_factor=smooth.smooth_factor,
            channel_dim=activation_channel_dim,
        )
        branch = solve_lowrank_output_error_branch(
            materialized_samples,
            output_residual,
            rank=rank,
            dtype=out_dtype,
            ridge=lowrank_ridge,
            input_channel_divisor=smooth.smooth_factor,
            input_channel_dim=activation_channel_dim,
            output_channel_dim=-1,
        )
    else:
        branch = solve_lowrank_residual_branch(branch_seed_residual, rank=rank, dtype=out_dtype)
    branch_weight = branch.proj_up.to(dtype=torch.float32).matmul(branch.proj_down.to(dtype=torch.float32).t())
    residual_weight = (smooth.smoothed_weight.detach().to(dtype=torch.float32) - branch_weight).contiguous()

    quantized = quantize_linear_weight_grouped_signed_int4_gptq(
        residual_weight,
        hessian=hessian,
        group_size=group_size,
        scale_dtype=scale_dtype,
        config=gptq_cfg,
    )
    return NaturalSvdquantLinearTensors(
        weight=quantized.packed_weight,
        weight_scale=quantized.weight_scale,
        smooth_factor=smooth.smooth_factor,
        proj_down=branch.proj_down,
        proj_up=branch.proj_up,
    )
