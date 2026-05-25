"""Activation statistics used by INT4 SVDQuant solvers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("torch is required for INT4 SVDQuant activation statistics") from exc
    return torch


@dataclass(frozen=True)
class ActivationStats:
    """Per-input-channel activation statistics for one linear layer."""

    input_amax: Any
    input_rms: Any | None = None
    sample_count: int = 0
    element_count: int = 0

    def to(self, *, device: Any | None = None, dtype: Any | None = None) -> "ActivationStats":
        """Return a copy with tensor fields moved to the requested dtype/device."""
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        input_rms = None if self.input_rms is None else self.input_rms.to(**kwargs)
        return ActivationStats(
            input_amax=self.input_amax.to(**kwargs),
            input_rms=input_rms,
            sample_count=int(self.sample_count),
            element_count=int(self.element_count),
        )

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        torch = _require_torch()

        def _vector(value: Any | None) -> list[float] | None:
            if value is None:
                return None
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]

        data: dict[str, Any] = {
            "input_amax": _vector(self.input_amax),
            "sample_count": int(self.sample_count),
            "element_count": int(self.element_count),
        }
        rms = _vector(self.input_rms)
        if rms is not None:
            data["input_rms"] = rms
        return data


def _as_1d_float_tensor(value: Any, *, name: str, device: Any | None = None):
    torch = _require_torch()
    if value is None:
        raise ValueError(f"{name} is required")
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.detach().to(device=device, dtype=torch.float32).reshape(-1).contiguous()
    if int(tensor.numel()) == 0:
        raise ValueError(f"{name} must not be empty")
    finite = torch.isfinite(tensor)
    if not bool(finite.all().item()):
        raise ValueError(f"{name} contains NaN or Inf values")
    return tensor


def as_activation_vector(value: Any, *, name: str = "activation", device: Any | None = None):
    """Return a finite 1-D float tensor for a per-channel activation vector."""
    return _as_1d_float_tensor(value, name=name, device=device)


def activation_stats_from_jsonable(data: Mapping[str, Any], *, device: Any | None = None) -> ActivationStats:
    """Decode one layer statistics record from JSON-compatible data."""
    if "input_amax" not in data:
        raise ValueError("activation stats record is missing input_amax")
    input_amax = _as_1d_float_tensor(data["input_amax"], name="input_amax", device=device).abs()
    input_rms = None
    if data.get("input_rms") is not None:
        input_rms = _as_1d_float_tensor(data["input_rms"], name="input_rms", device=device).abs()
        if int(input_rms.numel()) != int(input_amax.numel()):
            raise ValueError(
                f"input_rms length {int(input_rms.numel())} does not match input_amax length {int(input_amax.numel())}"
            )
    return ActivationStats(
        input_amax=input_amax,
        input_rms=input_rms,
        sample_count=int(data.get("sample_count", 0) or 0),
        element_count=int(data.get("element_count", 0) or 0),
    )


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
    tensor = sample.detach().to(dtype=torch.float32)
    if dim != int(tensor.ndim) - 1:
        tensor = tensor.movedim(dim, -1)
    return tensor.reshape(-1, int(tensor.shape[-1]))


def activation_amax_from_samples(samples: Iterable[Any], *, channel_dim: int = -1) -> ActivationStats:
    """Reduce activation tensors to per-channel amax and RMS statistics."""
    torch = _require_torch()
    total_amax = None
    total_sumsq = None
    total_rows = 0
    sample_count = 0
    for sample in samples:
        flattened = _flatten_channel_last(sample, channel_dim=channel_dim)
        if int(flattened.shape[0]) == 0:
            continue
        sample_count += 1
        abs_values = flattened.abs()
        sample_amax = abs_values.amax(dim=0)
        sample_sumsq = flattened.square().sum(dim=0)
        if total_amax is None:
            total_amax = sample_amax
            total_sumsq = sample_sumsq
        else:
            if int(sample_amax.numel()) != int(total_amax.numel()):
                raise ValueError(
                    f"activation channel count changed from {int(total_amax.numel())} to {int(sample_amax.numel())}"
                )
            total_amax = torch.maximum(total_amax, sample_amax)
            total_sumsq = total_sumsq + sample_sumsq
        total_rows += int(flattened.shape[0])
    if total_amax is None or total_sumsq is None or total_rows <= 0:
        raise ValueError("at least one non-empty activation sample is required")
    rms = torch.sqrt(total_sumsq / float(total_rows))
    return ActivationStats(
        input_amax=total_amax.contiguous(),
        input_rms=rms.contiguous(),
        sample_count=sample_count,
        element_count=total_rows,
    )


def merge_activation_stats(stats: Iterable[ActivationStats]) -> ActivationStats:
    """Merge multiple per-layer activation statistics records."""
    torch = _require_torch()
    merged_amax = None
    merged_sumsq = None
    total_samples = 0
    total_elements = 0
    rms_available = True
    for item in stats:
        current_amax = _as_1d_float_tensor(item.input_amax, name="input_amax").abs()
        if merged_amax is None:
            merged_amax = current_amax
            merged_sumsq = torch.zeros_like(current_amax)
        else:
            if int(current_amax.numel()) != int(merged_amax.numel()):
                raise ValueError(
                    f"activation stats channel count changed from {int(merged_amax.numel())} to {int(current_amax.numel())}"
                )
            merged_amax = torch.maximum(merged_amax, current_amax)
        total_samples += int(item.sample_count)
        elements = int(item.element_count)
        total_elements += elements
        if item.input_rms is None or elements <= 0:
            rms_available = False
        elif merged_sumsq is not None:
            current_rms = _as_1d_float_tensor(item.input_rms, name="input_rms").abs()
            if int(current_rms.numel()) != int(merged_amax.numel()):
                raise ValueError(
                    f"activation RMS channel count {int(current_rms.numel())} does not match amax {int(merged_amax.numel())}"
                )
            merged_sumsq = merged_sumsq + current_rms.square() * float(elements)
    if merged_amax is None:
        raise ValueError("at least one ActivationStats record is required")
    input_rms = None
    if rms_available and merged_sumsq is not None and total_elements > 0:
        input_rms = torch.sqrt(merged_sumsq / float(total_elements)).contiguous()
    return ActivationStats(
        input_amax=merged_amax.contiguous(),
        input_rms=input_rms,
        sample_count=total_samples,
        element_count=total_elements,
    )


def _layers_payload(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if "layers" in data:
        layers = data["layers"]
        if not isinstance(layers, Mapping):
            raise ValueError("activation stats field 'layers' must be an object")
        return layers
    return data


def load_activation_stats_map(path: str | Path, *, device: Any | None = None) -> dict[str, ActivationStats]:
    """Load per-layer activation statistics from a JSON file."""
    stats_path = Path(path).expanduser()
    with stats_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"activation stats file must contain a JSON object: {stats_path}")
    layers = _layers_payload(data)
    stats: dict[str, ActivationStats] = {}
    for layer_name, record in layers.items():
        if not isinstance(layer_name, str):
            raise ValueError("activation stats layer names must be strings")
        if not isinstance(record, Mapping):
            raise ValueError(f"activation stats for {layer_name} must be an object")
        stats[layer_name] = activation_stats_from_jsonable(record, device=device)
    if not stats:
        raise ValueError(f"activation stats file contains no layer records: {stats_path}")
    return stats


def write_activation_stats_map(path: str | Path, stats: Mapping[str, ActivationStats], *, schema_version: str = "int4_activation_stats.v1") -> None:
    """Write per-layer activation statistics as JSON."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": schema_version,
        "layers": {name: value.to_jsonable() for name, value in sorted(stats.items())},
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
