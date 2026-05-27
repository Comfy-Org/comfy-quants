"""Low-rank residual solvers for INT4 SVDQuant linear layers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("torch is required for INT4 SVDQuant low-rank solving") from exc
    return torch


@dataclass(frozen=True)
class LowRankBranch:
    """Low-rank tensors where ``residual ≈ proj_up @ proj_down.T``."""

    proj_down: Any
    proj_up: Any


def _flatten_channel_last(sample: Any, *, channel_dim: int):
    torch = _require_torch()
    if not torch.is_tensor(sample):
        sample = torch.as_tensor(sample)
    if int(sample.ndim) == 0:
        raise ValueError("sample must have at least one dimension")
    dim = int(channel_dim)
    if dim < 0:
        dim += int(sample.ndim)
    if dim < 0 or dim >= int(sample.ndim):
        raise ValueError(f"channel_dim {channel_dim} is out of range for shape {tuple(sample.shape)}")
    tensor = sample.detach()
    if dim != int(tensor.ndim) - 1:
        tensor = tensor.movedim(dim, -1)
    return tensor.reshape(-1, int(tensor.shape[-1]))


def _is_sample_sequence(value: Any, torch: Any) -> bool:
    if torch.is_tensor(value):
        return False
    if isinstance(value, dict | str | bytes):
        return False
    return isinstance(value, Iterable)


def _iter_sample_pairs(inputs: Any, output_residual: Any, torch: Any):
    input_is_sequence = _is_sample_sequence(inputs, torch)
    residual_is_sequence = _is_sample_sequence(output_residual, torch)
    if input_is_sequence != residual_is_sequence:
        raise ValueError("inputs and output_residual must both be tensors or both be sample iterables")
    if not input_is_sequence:
        yield inputs, output_residual
        return

    sentinel = object()
    for input_sample, residual_sample in zip_longest(inputs, output_residual, fillvalue=sentinel):
        if input_sample is sentinel or residual_sample is sentinel:
            raise ValueError("inputs and output_residual iterables must contain the same number of samples")
        yield input_sample, residual_sample


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


def solve_lowrank_residual_branch(residual: Any, *, rank: int, dtype: Any | None = None) -> LowRankBranch:
    """Approximate a residual matrix with a rank-limited branch."""
    torch = _require_torch()
    if int(residual.ndim) != 2:
        raise ValueError(f"residual must be rank 2, got shape {tuple(residual.shape)}")
    n, k = int(residual.shape[0]), int(residual.shape[1])
    rank = int(rank)
    if n <= 0 or k <= 0:
        raise ValueError(f"residual dimensions must be positive, got {tuple(residual.shape)}")
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    if dtype is None:
        dtype = residual.dtype if residual.dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64} else torch.float16

    proj_down = torch.zeros((k, rank), dtype=dtype, device=residual.device)
    proj_up = torch.zeros((n, rank), dtype=dtype, device=residual.device)
    if rank == 0:
        return LowRankBranch(proj_down=proj_down, proj_up=proj_up)

    working = residual.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(working).all().item()):
        raise ValueError("residual contains NaN or Inf values")
    effective_rank = min(rank, n, k)
    if effective_rank == 0 or float(working.abs().amax().item()) == 0.0:
        return LowRankBranch(proj_down=proj_down, proj_up=proj_up)

    u, s, vh = torch.linalg.svd(working, full_matrices=False)
    proj_up[:, :effective_rank] = (u[:, :effective_rank] * s[:effective_rank].reshape(1, effective_rank)).to(dtype=dtype)
    proj_down[:, :effective_rank] = vh[:effective_rank, :].transpose(0, 1).to(dtype=dtype)
    return LowRankBranch(proj_down=proj_down.contiguous(), proj_up=proj_up.contiguous())


def solve_lowrank_output_error_branch(
    inputs: Any,
    output_residual: Any,
    *,
    rank: int,
    dtype: Any | None = None,
    ridge: float = 1.0e-6,
    input_channel_divisor: Any | None = None,
    input_channel_dim: int = -1,
    output_channel_dim: int = -1,
) -> LowRankBranch:
    """Fit a low-rank branch from layer inputs and output residuals.

    This solver estimates a dense correction ``B`` from output-space error:

    ``output_residual ≈ (inputs / input_channel_divisor) @ B.T``

    and then factorizes ``B`` into the SVDQuant branch contract
    ``B ≈ proj_up @ proj_down.T``.  It is intentionally runtime-independent:
    callers supply already-captured tensors and decide whether the input basis
    is raw or post-smoothing.
    """
    torch = _require_torch()
    rank = int(rank)
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    ridge = float(ridge)
    if ridge < 0.0:
        raise ValueError(f"ridge must be non-negative, got {ridge}")

    gram = None
    rhs = None
    channel_count: int | None = None
    output_count: int | None = None
    row_count = 0
    divisor = None
    target_device = None
    compute_dtype = torch.float32

    for input_sample, residual_sample in _iter_sample_pairs(inputs, output_residual, torch):
        input_rows = _flatten_channel_last(input_sample, channel_dim=input_channel_dim)
        residual_rows = _flatten_channel_last(residual_sample, channel_dim=output_channel_dim)
        if int(input_rows.shape[0]) != int(residual_rows.shape[0]):
            raise ValueError(
                "inputs and output_residual must flatten to the same row count, "
                f"got {int(input_rows.shape[0])} and {int(residual_rows.shape[0])}"
            )
        if int(input_rows.shape[0]) == 0:
            continue

        if target_device is None:
            target_device = input_rows.device
        input_rows = input_rows.to(device=target_device, dtype=compute_dtype)
        residual_rows = residual_rows.to(device=target_device, dtype=compute_dtype)
        if not bool(torch.isfinite(input_rows).all().item()):
            raise ValueError("inputs contain NaN or Inf values")
        if not bool(torch.isfinite(residual_rows).all().item()):
            raise ValueError("output_residual contains NaN or Inf values")

        if channel_count is None:
            channel_count = int(input_rows.shape[1])
            output_count = int(residual_rows.shape[1])
            if channel_count <= 0 or output_count <= 0:
                raise ValueError(
                    f"input/output channel counts must be positive, got {channel_count} and {output_count}"
                )
            gram = torch.zeros((channel_count, channel_count), device=target_device, dtype=compute_dtype)
            rhs = torch.zeros((channel_count, output_count), device=target_device, dtype=compute_dtype)
            divisor = _as_input_divisor(
                input_channel_divisor,
                channel_count=channel_count,
                device=target_device,
                dtype=compute_dtype,
            )
        elif int(input_rows.shape[1]) != channel_count or int(residual_rows.shape[1]) != output_count:
            raise ValueError(
                "sample channel counts changed: "
                f"expected input/output {(channel_count, output_count)}, "
                f"got {(int(input_rows.shape[1]), int(residual_rows.shape[1]))}"
            )

        if divisor is not None:
            input_rows = input_rows / divisor.reshape(1, channel_count)
        gram += input_rows.t().matmul(input_rows)
        rhs += input_rows.t().matmul(residual_rows)
        row_count += int(input_rows.shape[0])

    if gram is None or rhs is None or channel_count is None or output_count is None or row_count <= 0:
        raise ValueError("at least one non-empty input/output residual sample pair is required")

    if ridge > 0.0:
        gram = gram.clone()
        gram.diagonal().add_(ridge)
    try:
        residual_t = torch.linalg.solve(gram, rhs)
    except RuntimeError:
        residual_t = torch.linalg.lstsq(gram, rhs).solution
    residual = residual_t.t().contiguous()
    if not bool(torch.isfinite(residual).all().item()):
        raise ValueError("output-error low-rank solve produced NaN or Inf values")

    if dtype is None:
        dtype = residual.dtype
    return solve_lowrank_residual_branch(residual, rank=rank, dtype=dtype)
