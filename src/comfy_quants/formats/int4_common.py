"""Shared INT4 tensor packing primitives.

The helpers in this module describe byte-level storage only.  They do not own
model-layer selection, calibration, low-rank solving, or checkpoint file I/O.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("torch is required for tensor INT4 packing") from exc
    return torch


def encode_quant_config_tensor(config: Mapping[str, object]):
    """Encode a checkpoint quantization config as a uint8 JSON tensor."""
    torch = _require_torch()
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


def decode_quant_config_tensor(tensor: Any | None) -> dict[str, object] | None:
    """Decode a uint8 JSON tensor used for checkpoint quantization metadata."""
    if tensor is None:
        return None
    data = bytes(tensor.detach().cpu().tolist()).decode("utf-8")
    return json.loads(data)


def _validate_even_last_dim(tensor: Any, *, tensor_name: str) -> None:
    if int(tensor.shape[-1]) % 2 != 0:
        raise ValueError(f"{tensor_name} last dimension must be even, got shape {tuple(tensor.shape)}")


def _validate_signed_int4_range(values: Any) -> None:
    if int(values.numel()) == 0:
        return
    invalid = (values < -8) | (values > 7)
    if bool(invalid.any().item()):
        raise ValueError("signed INT4 tensor contains values outside [-8, 7]")


def _validate_unsigned_int4_range(values: Any) -> None:
    if int(values.numel()) == 0:
        return
    invalid = (values < 0) | (values > 15)
    if bool(invalid.any().item()):
        raise ValueError("unsigned INT4 tensor contains values outside [0, 15]")


def pack_signed_int4_pairs(values: Any, *, validate: bool = True):
    """Pack adjacent signed INT4 values from the last dimension into int8 bytes.

    The first value is stored in the low nibble and the second value is stored
    in the high nibble.  Values are interpreted as signed INT4 in ``[-8, 7]``.
    """
    torch = _require_torch()
    _validate_even_last_dim(values, tensor_name="values")
    if validate:
        _validate_signed_int4_range(values)
    lo = values[..., 0::2].to(torch.int32).bitwise_and(0x0F)
    hi = values[..., 1::2].to(torch.int32).bitwise_and(0x0F).bitwise_left_shift(4)
    return (lo | hi).to(torch.int8).contiguous()


def unpack_signed_int4_pairs(packed: Any):
    """Unpack int8-pair bytes into signed INT4 values stored in int8 tensors."""
    torch = _require_torch()
    x32 = packed.to(torch.int32)
    lo = x32.bitwise_and(0x0F)
    hi = x32.bitwise_right_shift(4).bitwise_and(0x0F)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    shape = (*packed.shape[:-1], int(packed.shape[-1]) * 2)
    return torch.stack((lo, hi), dim=-1).reshape(shape).to(torch.int8).contiguous()


def pack_uint4_pairs(values: Any, *, validate: bool = True):
    """Pack adjacent unsigned INT4 values from the last dimension into bytes.

    The first value is stored in the low nibble and the second value is stored
    in the high nibble.  The returned tensor uses ``torch.int8`` storage so it
    can be written by safetensors alongside the signed-INT4 checkpoint tensors.
    """
    torch = _require_torch()
    _validate_even_last_dim(values, tensor_name="values")
    if validate:
        _validate_unsigned_int4_range(values)
    lo = values[..., 0::2].to(torch.int32).bitwise_and(0x0F)
    hi = values[..., 1::2].to(torch.int32).bitwise_and(0x0F).bitwise_left_shift(4)
    return (lo | hi).to(torch.int8).contiguous()


def unpack_uint4_pairs(packed: Any):
    """Unpack byte-pair storage into unsigned INT4 values stored in int8 tensors."""
    torch = _require_torch()
    x32 = packed.to(torch.int32)
    lo = x32.bitwise_and(0x0F)
    hi = x32.bitwise_right_shift(4).bitwise_and(0x0F)
    shape = (*packed.shape[:-1], int(packed.shape[-1]) * 2)
    return torch.stack((lo, hi), dim=-1).reshape(shape).to(torch.int8).contiguous()
