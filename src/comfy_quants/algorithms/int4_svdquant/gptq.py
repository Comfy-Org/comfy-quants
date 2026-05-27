"""GPTQ helpers for grouped signed-INT4 SVDQuant weights.

This module is intentionally runtime-independent.  It operates on dense PyTorch
tensors, activation samples or precomputed Hessians, and the byte-level INT4
packing contract used by the rest of this package.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from comfy_quants.formats.int4_common import pack_signed_int4_pairs
from comfy_quants.formats.kitchen_tilepack import KITCHEN_GROUP_SIZE


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("torch is required for GPTQ INT4 quantization") from exc
    return torch


@dataclass(frozen=True)
class GptqConfig:
    """Numerical options for the grouped GPTQ weight solve."""

    damp_percentage: float = 0.01
    block_size: int = 128
    num_inv_tries: int = 250
    hessian_block_size: int = 512
    use_importance_ordering: bool = True
    fallback_to_rtn: bool = True

    def validate(self) -> None:
        if float(self.damp_percentage) < 0.0:
            raise ValueError(f"damp_percentage must be non-negative, got {self.damp_percentage}")
        if int(self.block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if int(self.num_inv_tries) < 0:
            raise ValueError(f"num_inv_tries must be non-negative, got {self.num_inv_tries}")
        if int(self.hessian_block_size) == 0:
            raise ValueError("hessian_block_size must be positive or negative")


@dataclass(frozen=True)
class GptqHessianStats:
    """Precomputed input Hessian for one linear layer."""

    hessian: Any
    channel_count: int
    sample_count: int
    row_count: int
    normalization_count: int


@dataclass(frozen=True)
class GptqInt4WeightQuantization:
    """Grouped signed-INT4 GPTQ result for one dense linear weight."""

    packed_weight: Any
    weight_scale: Any
    quantized_weight: Any
    dequantized_weight: Any
    hessian_inverse_attempts: int
    used_rtn_fallback: bool
    dead_column_count: int


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


def _validate_weight_shape(weight: Any, *, group_size: int) -> tuple[int, int]:
    if int(weight.ndim) != 2:
        raise ValueError(f"linear weight must be rank 2, got shape {tuple(weight.shape)}")
    n = int(weight.shape[0])
    k = int(weight.shape[1])
    if n <= 0 or k <= 0:
        raise ValueError(f"linear weight dimensions must be positive, got {tuple(weight.shape)}")
    if k % int(group_size) != 0:
        raise ValueError(f"input dimension K={k} is not divisible by group size {group_size}")
    return n, k


def _flatten_channel_last(sample: Any, *, channel_dim: int):
    torch = _require_torch()
    if not torch.is_tensor(sample):
        sample = torch.as_tensor(sample)
    if int(sample.ndim) == 0:
        raise ValueError("activation sample must have at least one dimension")
    dim = int(channel_dim)
    if dim < 0:
        dim += int(sample.ndim)
    if dim < 0 or dim >= int(sample.ndim):
        raise ValueError(f"channel_dim {channel_dim} is out of range for shape {tuple(sample.shape)}")
    tensor = sample.detach()
    if dim != int(tensor.ndim) - 1:
        tensor = tensor.movedim(dim, -1)
    return tensor.reshape(-1, int(tensor.shape[-1]))


def _as_input_divisor(input_channel_divisor: Any | None, *, channel_count: int, device: Any, dtype: Any):
    if input_channel_divisor is None:
        return None
    torch = _require_torch()
    divisor = input_channel_divisor if torch.is_tensor(input_channel_divisor) else torch.as_tensor(input_channel_divisor)
    divisor = divisor.detach().to(device=device, dtype=dtype).reshape(-1)
    if int(divisor.numel()) != int(channel_count):
        raise ValueError(
            f"input_channel_divisor length {int(divisor.numel())} does not match channel count {int(channel_count)}"
        )
    if bool((~torch.isfinite(divisor)).any().item()) or bool((divisor == 0).any().item()):
        raise ValueError("input_channel_divisor must contain finite non-zero values")
    return divisor


def build_gptq_hessian_from_activations(
    samples: Iterable[Any],
    *,
    channel_dim: int = -1,
    input_channel_divisor: Any | None = None,
    hessian_block_size: int = 512,
    normalization_sample_count: int | None = None,
    device: Any | None = None,
    dtype: Any | None = None,
) -> GptqHessianStats:
    """Build a GPTQ Hessian from captured layer input activations.

    ``input_channel_divisor`` is applied after flattening.  For SVDQuant
    smoothing, pass the layer ``smooth_factor`` so GPTQ sees the post-smoothing
    inputs ``x / smooth_factor`` paired with the smoothed weight.
    """
    torch = _require_torch()
    compute_dtype = torch.float32 if dtype is None else dtype
    target_device = None if device is None else torch.device(device)
    hessian = None
    channel_count: int | None = None
    row_count = 0
    sample_count = 0
    divisor = None
    block_size = int(hessian_block_size)

    for sample in samples:
        rows = _flatten_channel_last(sample, channel_dim=channel_dim)
        if int(rows.shape[0]) == 0:
            continue
        if target_device is None:
            target_device = rows.device
        rows = rows.to(device=target_device, dtype=compute_dtype)
        if channel_count is None:
            channel_count = int(rows.shape[1])
            hessian = torch.zeros((channel_count, channel_count), device=target_device, dtype=compute_dtype)
            divisor = _as_input_divisor(
                input_channel_divisor,
                channel_count=channel_count,
                device=target_device,
                dtype=compute_dtype,
            )
        elif int(rows.shape[1]) != channel_count:
            raise ValueError(f"activation channel count changed from {channel_count} to {int(rows.shape[1])}")
        if divisor is not None:
            rows = rows / divisor.reshape(1, channel_count)

        if block_size > 0 and int(rows.shape[0]) > block_size:
            for start in range(0, int(rows.shape[0]), block_size):
                block = rows[start : start + block_size]
                hessian += block.t().matmul(block)
        else:
            hessian += rows.t().matmul(rows)
        row_count += int(rows.shape[0])
        sample_count += 1

    if hessian is None or channel_count is None or row_count <= 0:
        raise ValueError("at least one non-empty activation sample is required")
    normalization_count = int(normalization_sample_count or row_count)
    if normalization_count <= 0:
        raise ValueError(f"normalization_sample_count must be positive, got {normalization_sample_count}")
    hessian = (hessian * (2.0 / float(normalization_count))).contiguous()
    return GptqHessianStats(
        hessian=hessian,
        channel_count=channel_count,
        sample_count=sample_count,
        row_count=row_count,
        normalization_count=normalization_count,
    )


def transform_gptq_hessian_input_basis(
    hessian: GptqHessianStats | Any,
    *,
    input_channel_divisor: Any,
) -> GptqHessianStats | Any:
    """Transform a raw-input Hessian into the basis after channel division.

    SVDQuant smoothing pairs a smoothed weight with inputs divided by the solved
    ``smooth_factor``.  If a manifest stores ``H_raw = X.T @ X * 2 / n``, GPTQ
    must consume ``H_smooth = D^-1 @ H_raw @ D^-1`` where
    ``D = diag(smooth_factor)``.
    """
    torch = _require_torch()
    is_stats = isinstance(hessian, GptqHessianStats)
    tensor = hessian.hessian if is_stats else hessian
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.detach().to(dtype=torch.float32).contiguous()
    if int(tensor.ndim) != 2 or int(tensor.shape[0]) != int(tensor.shape[1]):
        raise ValueError(f"GPTQ Hessian must be square, got shape {tuple(tensor.shape)}")
    channel_count = int(tensor.shape[0])
    divisor = _as_input_divisor(
        input_channel_divisor,
        channel_count=channel_count,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    transformed = (tensor / divisor.reshape(channel_count, 1) / divisor.reshape(1, channel_count)).contiguous()
    if is_stats:
        return GptqHessianStats(
            hessian=transformed,
            channel_count=int(hessian.channel_count),
            sample_count=int(hessian.sample_count),
            row_count=int(hessian.row_count),
            normalization_count=int(hessian.normalization_count),
        )
    return transformed


def _group_scales(weight: Any, *, group_size: int):
    torch = _require_torch()
    n, k = _validate_weight_shape(weight, group_size=group_size)
    groups = k // int(group_size)
    working = weight.detach().to(dtype=torch.float32).contiguous().view(n, groups, int(group_size))
    amax = working.abs().amax(dim=2)
    return torch.where(amax > 0, amax / 7.0, torch.ones_like(amax)).contiguous()


def _quantize_columns_rtn(weight: Any, scale: Any, *, group_size: int):
    torch = _require_torch()
    groups = int(weight.shape[1]) // int(group_size)
    scale_by_column = scale.unsqueeze(-1).expand(int(weight.shape[0]), groups, int(group_size)).reshape_as(weight)
    quantized = torch_round_clamp_int4(weight / scale_by_column)
    dequantized = (quantized.to(dtype=torch.float32) * scale_by_column).contiguous()
    return quantized, dequantized


def torch_round_clamp_int4(values: Any):
    """Round floating values to the SVDQuant signed INT4 emission range."""
    torch = _require_torch()
    return torch.round(values).clamp(-7, 7).to(torch.int8).contiguous()


def _prepare_hessian(hessian: Any, *, expected_k: int, device: Any):
    torch = _require_torch()
    if isinstance(hessian, GptqHessianStats):
        hessian = hessian.hessian
    if not torch.is_tensor(hessian):
        hessian = torch.as_tensor(hessian)
    hessian = hessian.detach().to(device=device, dtype=torch.float32).contiguous()
    if tuple(int(x) for x in hessian.shape) != (int(expected_k), int(expected_k)):
        raise ValueError(f"hessian shape {tuple(hessian.shape)} does not match expected {(int(expected_k), int(expected_k))}")
    if bool((~torch.isfinite(hessian)).any().item()):
        raise ValueError("hessian contains NaN or Inf values")
    return ((hessian + hessian.t()) * 0.5).contiguous()


def _damp_and_factor_hessian(hessian: Any, config: GptqConfig) -> tuple[Any | None, int]:
    torch = _require_torch()
    h = hessian.clone()
    diag = h.diagonal()
    diag_mean = diag.mean()
    if not bool(torch.isfinite(diag_mean).item()) or float(diag_mean.item()) <= 0.0:
        diag_mean = torch.ones((), device=h.device, dtype=h.dtype)
    diag += float(config.damp_percentage) * diag_mean

    hessian_inv = None
    attempts = 0
    while attempts < int(config.num_inv_tries):
        attempts += 1
        try:
            chol = torch.linalg.cholesky(h)
            inv = torch.cholesky_inverse(chol)
            hessian_inv = torch.linalg.cholesky(inv, upper=True)
            if bool(torch.isfinite(hessian_inv).all().item()):
                return hessian_inv.contiguous(), attempts
        except RuntimeError:
            diag += (float(config.damp_percentage) * 0.1) * diag_mean
            continue
        diag += (float(config.damp_percentage) * 0.1) * diag_mean
    return None, attempts


def quantize_linear_weight_grouped_signed_int4_gptq(
    weight: Any,
    *,
    hessian: Any,
    group_size: int = KITCHEN_GROUP_SIZE,
    scale_dtype: str = "source",
    config: GptqConfig | None = None,
) -> GptqInt4WeightQuantization:
    """Quantize a linear weight with per-row/group scales and GPTQ updates."""
    torch = _require_torch()
    cfg = config or GptqConfig()
    cfg.validate()
    group_size = int(group_size)
    n, k = _validate_weight_shape(weight, group_size=group_size)
    source_dtype = weight.dtype
    out_dtype = _resolve_scale_dtype(torch, source_dtype, scale_dtype)
    working = weight.detach().to(dtype=torch.float32).contiguous()
    scale = _group_scales(working, group_size=group_size)
    prepared_hessian = _prepare_hessian(hessian, expected_k=k, device=working.device)

    dead = prepared_hessian.diagonal() <= 0
    dead_column_count = int(dead.sum().item())
    if dead_column_count:
        prepared_hessian[dead, dead] = 1.0
        working[:, dead] = 0.0

    if cfg.use_importance_ordering:
        importance = prepared_hessian.diagonal()
        permute = torch.argsort(importance, descending=True)
    else:
        permute = torch.arange(k, device=working.device)
    inverse_permute = torch.argsort(permute)
    hessian_ordered = prepared_hessian[permute][:, permute].contiguous()
    working_ordered = working[:, permute].contiguous()

    hessian_inv, attempts = _damp_and_factor_hessian(hessian_ordered, cfg)
    if hessian_inv is None:
        if not bool(cfg.fallback_to_rtn):
            raise ValueError(f"failed to factor damped Hessian after {attempts} attempts")
        quantized, dequantized = _quantize_columns_rtn(working, scale, group_size=group_size)
        return GptqInt4WeightQuantization(
            packed_weight=pack_signed_int4_pairs(quantized, validate=False),
            weight_scale=scale.t().contiguous().to(dtype=out_dtype),
            quantized_weight=quantized,
            dequantized_weight=dequantized,
            hessian_inverse_attempts=attempts,
            used_rtn_fallback=True,
            dead_column_count=dead_column_count,
        )

    qtensor = torch.zeros_like(working_ordered)
    block_size = int(cfg.block_size)
    groups = k // group_size
    scale_by_column = scale.unsqueeze(-1).expand(n, groups, group_size).reshape(n, k)
    for c_start in range(0, k, block_size):
        c_end = min(c_start + block_size, k)
        block_weight = working_ordered[:, c_start:c_end].clone()
        block_qtensor = qtensor[:, c_start:c_end]
        block_hessian_inv = hessian_inv[c_start:c_end, c_start:c_end]
        block_error = torch.zeros_like(block_weight)
        for local_col in range(c_end - c_start):
            ordered_col = c_start + local_col
            original_col = int(permute[ordered_col].item())
            column = block_weight[:, local_col]
            pos_diag = block_hessian_inv[local_col, local_col]
            column_scale = scale_by_column[:, original_col]
            qcolumn = torch_round_clamp_int4(column / column_scale)
            block_qtensor[:, local_col] = qcolumn.to(dtype=block_qtensor.dtype)
            dequant_column = qcolumn.to(dtype=torch.float32) * column_scale
            column_error = (column - dequant_column) / pos_diag
            block_error[:, local_col] = column_error
            block_weight[:, local_col:] -= column_error.reshape(-1, 1).matmul(
                block_hessian_inv[local_col, local_col:].reshape(1, -1)
            )
        if c_end < k:
            working_ordered[:, c_end:] -= block_error.matmul(hessian_inv[c_start:c_end, c_end:])

    quantized = qtensor[:, inverse_permute].round().clamp(-7, 7).to(torch.int8).contiguous()
    dequantized = (quantized.to(dtype=torch.float32) * scale_by_column).contiguous()
    return GptqInt4WeightQuantization(
        packed_weight=pack_signed_int4_pairs(quantized, validate=False),
        weight_scale=scale.t().contiguous().to(dtype=out_dtype),
        quantized_weight=quantized,
        dequantized_weight=dequantized,
        hessian_inverse_attempts=attempts,
        used_rtn_fallback=False,
        dead_column_count=dead_column_count,
    )
