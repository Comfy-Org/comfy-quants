"""Torch reference export backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from comfy_quants.core.artifact import QuantArtifact
from comfy_quants.formats.fp8_common import get_fp8_runtime_spec


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("torch is required for the torch_ref backend") from exc
    return torch


def _torch_fp8_dtype(torch, quant_dtype: str):
    spec = get_fp8_runtime_spec(quant_dtype)
    if not hasattr(torch, spec.torch_dtype_name):
        raise ImportError(f"torch.{spec.torch_dtype_name} is required for {spec.name} reference quantization")
    return getattr(torch, spec.torch_dtype_name)


def _axis_index(axis: str | int | None, rank: int) -> int | None:
    if axis is None:
        return None
    if isinstance(axis, int):
        index = axis
    elif axis == "out_features":
        index = 0
    elif axis == "in_features":
        index = 1
    else:
        raise ValueError(f"unsupported scale axis: {axis}")
    if index < 0:
        index += rank
    if index < 0 or index >= rank:
        raise ValueError(f"scale axis {axis} is outside tensor rank {rank}")
    return index


def _scale_view_shape(scale, rank: int, axis: str | int | None) -> list[int]:
    index = _axis_index(axis, rank)
    if index is None:
        return [1] * rank
    shape = [1] * rank
    shape[index] = int(scale.numel())
    return shape


@dataclass
class FP8Tensor:
    """Reference FP8 tensor payload and scale."""

    payload: Any
    scale: Any
    scale_axis: str | int | None
    scale_granularity: str
    source_dtype: str
    quant_dtype: str
    storage_dtype: str = "uint8"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "payload_shape": list(self.payload.shape),
            "scale_shape": list(self.scale.shape),
            "scale_axis": self.scale_axis,
            "scale_granularity": self.scale_granularity,
            "source_dtype": self.source_dtype,
            "quant_dtype": self.quant_dtype,
            "storage_dtype": self.storage_dtype,
        }


class FP8E4M3Tensor(FP8Tensor):
    """Reference FP8 E4M3 tensor payload and scale."""


class FP8E5M2Tensor(FP8Tensor):
    """Reference FP8 E5M2 tensor payload and scale."""


def solve_fp8_scale(
    tensor,
    *,
    quant_dtype: str,
    granularity: str = "per_channel",
    axis: str | int | None = "out_features",
    eps: float = 1.0e-12,
):
    """Solve FP32 scales for a supported FP8 quantization format."""
    torch = _require_torch()
    spec = get_fp8_runtime_spec(quant_dtype)
    _torch_fp8_dtype(torch, spec.name)
    values = tensor.detach().to(torch.float32)
    if granularity == "per_tensor":
        amax = values.abs().max()
        return torch.where(amax > 0, torch.clamp(amax / spec.max_finite, min=eps), torch.ones_like(amax, dtype=torch.float32))
    if granularity != "per_channel":
        raise ValueError(f"unsupported scale granularity: {granularity}")
    index = _axis_index(axis, values.ndim)
    if index is None:
        raise ValueError("per_channel scale requires an axis")
    reduce_dims = tuple(dim for dim in range(values.ndim) if dim != index)
    amax = values.abs().amax(dim=reduce_dims)
    scale = amax / spec.max_finite
    return torch.where(amax > 0, torch.clamp(scale, min=eps), torch.ones_like(scale, dtype=torch.float32))


def quantize_fp8_payload(
    tensor,
    scale,
    *,
    quant_dtype: str,
    axis: str | int | None = "out_features",
    rounding: str = "nearest_even",
):
    """Quantize a tensor to uint8 bytes for a supported FP8 format."""
    if rounding != "nearest_even":
        raise ValueError(f"unsupported rounding mode: {rounding}")
    torch = _require_torch()
    spec = get_fp8_runtime_spec(quant_dtype)
    fp8_dtype = _torch_fp8_dtype(torch, spec.name)
    values = tensor.detach().to(torch.float32)
    scale_view = scale.to(torch.float32).reshape(_scale_view_shape(scale, values.ndim, axis))
    normalized = (values / scale_view).clamp(-spec.max_finite, spec.max_finite)
    return normalized.to(fp8_dtype).view(torch.uint8).contiguous()


def dequantize_fp8_payload(
    payload,
    scale,
    *,
    quant_dtype: str,
    axis: str | int | None = "out_features",
):
    """Dequantize uint8 FP8 bytes to FP32 values."""
    torch = _require_torch()
    fp8_dtype = _torch_fp8_dtype(torch, quant_dtype)
    scale_view = scale.to(torch.float32).reshape(_scale_view_shape(scale, payload.ndim, axis))
    values = payload.contiguous().view(fp8_dtype).to(torch.float32)
    return values * scale_view


def quantize_tensor_fp8(
    tensor,
    *,
    quant_dtype: str,
    granularity: str = "per_channel",
    axis: str | int | None = "out_features",
    rounding: str = "nearest_even",
) -> FP8Tensor:
    """Solve scales and produce a reference FP8 payload."""
    spec = get_fp8_runtime_spec(quant_dtype)
    scale = solve_fp8_scale(tensor, quant_dtype=spec.name, granularity=granularity, axis=axis)
    payload = quantize_fp8_payload(tensor, scale, quant_dtype=spec.name, axis=axis, rounding=rounding)
    tensor_cls: type[FP8Tensor]
    if spec.name == "fp8_e4m3":
        tensor_cls = FP8E4M3Tensor
    elif spec.name == "fp8_e5m2":
        tensor_cls = FP8E5M2Tensor
    else:  # pragma: no cover - protected by get_fp8_runtime_spec
        tensor_cls = FP8Tensor
    return tensor_cls(
        payload=payload,
        scale=scale,
        scale_axis=axis,
        scale_granularity=granularity,
        source_dtype=str(tensor.dtype).replace("torch.", ""),
        quant_dtype=spec.name,
    )


def solve_fp8_e4m3_scale(
    tensor,
    *,
    granularity: str = "per_channel",
    axis: str | int | None = "out_features",
    eps: float = 1.0e-12,
):
    """Solve FP32 scales for FP8 E4M3 quantization."""
    return solve_fp8_scale(tensor, quant_dtype="fp8_e4m3", granularity=granularity, axis=axis, eps=eps)


def quantize_fp8_e4m3_payload(
    tensor,
    scale,
    *,
    axis: str | int | None = "out_features",
    rounding: str = "nearest_even",
):
    """Quantize a tensor to uint8 bytes using torch.float8_e4m3fn."""
    return quantize_fp8_payload(tensor, scale, quant_dtype="fp8_e4m3", axis=axis, rounding=rounding)


def dequantize_fp8_e4m3_payload(
    payload,
    scale,
    *,
    axis: str | int | None = "out_features",
):
    """Dequantize uint8 FP8 E4M3 bytes to FP32 values."""
    return dequantize_fp8_payload(payload, scale, quant_dtype="fp8_e4m3", axis=axis)


def quantize_tensor_fp8_e4m3(
    tensor,
    *,
    granularity: str = "per_channel",
    axis: str | int | None = "out_features",
    rounding: str = "nearest_even",
) -> FP8E4M3Tensor:
    """Solve scales and produce a reference FP8 E4M3 payload."""
    return quantize_tensor_fp8(tensor, quant_dtype="fp8_e4m3", granularity=granularity, axis=axis, rounding=rounding)  # type: ignore[return-value]


def solve_fp8_e5m2_scale(
    tensor,
    *,
    granularity: str = "per_channel",
    axis: str | int | None = "out_features",
    eps: float = 1.0e-12,
):
    """Solve FP32 scales for FP8 E5M2 quantization."""
    return solve_fp8_scale(tensor, quant_dtype="fp8_e5m2", granularity=granularity, axis=axis, eps=eps)


def quantize_fp8_e5m2_payload(
    tensor,
    scale,
    *,
    axis: str | int | None = "out_features",
    rounding: str = "nearest_even",
):
    """Quantize a tensor to uint8 bytes using torch.float8_e5m2."""
    return quantize_fp8_payload(tensor, scale, quant_dtype="fp8_e5m2", axis=axis, rounding=rounding)


def dequantize_fp8_e5m2_payload(
    payload,
    scale,
    *,
    axis: str | int | None = "out_features",
):
    """Dequantize uint8 FP8 E5M2 bytes to FP32 values."""
    return dequantize_fp8_payload(payload, scale, quant_dtype="fp8_e5m2", axis=axis)


def quantize_tensor_fp8_e5m2(
    tensor,
    *,
    granularity: str = "per_channel",
    axis: str | int | None = "out_features",
    rounding: str = "nearest_even",
) -> FP8E5M2Tensor:
    """Solve scales and produce a reference FP8 E5M2 payload."""
    return quantize_tensor_fp8(tensor, quant_dtype="fp8_e5m2", granularity=granularity, axis=axis, rounding=rounding)  # type: ignore[return-value]


class TorchReferenceBackend:
    backend_name = "torch_ref"
    version = "0.1.0"

    def check_compatibility(self, artifact: QuantArtifact) -> dict:
        return {"backend": self.backend_name, "level": "L1", "hardware_accelerated": False, "artifact_id": artifact.artifact_id}

    def export(self, artifact: QuantArtifact, output_dir: str) -> dict:
        return {"backend": self.backend_name, "output_dir": output_dir, "artifact_id": artifact.artifact_id}


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_backend(TorchReferenceBackend())
