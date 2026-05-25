"""Kitchen tile-packed SVDQuant W4A4 tensor layout.

This module implements the checkpoint storage layout identified by
``layout="kitchen_tile_packed_w4a4"``.  It is a layout transform only: signed
INT4 values, scales, smooth factors, low-rank tensors, and bias values are
preserved.  Model-specific choices such as QKV splitting belong in model
adapters, not in this format module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from comfy_quants.formats.int4_common import (
    decode_quant_config_tensor,
    encode_quant_config_tensor,
    pack_signed_int4_pairs,
    unpack_signed_int4_pairs,
)

KITCHEN_BLOCK_N = 128
KITCHEN_GROUP_SIZE = 64
KITCHEN_INTERLEAVE = 4
KITCHEN_TILEPACK_LAYOUT_NAME = "kitchen_tile_packed_w4a4"
SVDQUANT_W4A4_FORMAT_NAME = "svdquant_w4a4"
SVDQUANT_WEIGHT_SCALE_STORAGE_DTYPE_NAME = "bfloat16"

SVDQUANT_REQUIRED_PARAM_KEYS = (
    "weight",
    "weight_scale",
    "smooth_factor",
    "proj_down",
    "proj_up",
)
SVDQUANT_OPTIONAL_PARAM_KEYS = ("bias", "comfy_quant")

ProgressCallback = Callable[[int, int, str], None]


def is_svdquant_quant_config(config: Mapping[str, object] | None) -> bool:
    """Return whether a decoded checkpoint config declares SVDQuant W4A4."""
    return config is not None and config.get("format") == SVDQUANT_W4A4_FORMAT_NAME


def _validate_weight_tile_shape(weight: Any) -> None:
    expected_tail = (KITCHEN_BLOCK_N // KITCHEN_INTERLEAVE, KITCHEN_INTERLEAVE * KITCHEN_GROUP_SIZE // 2)
    if int(weight.ndim) != 4 or tuple(weight.shape[2:]) != expected_tail:
        raise ValueError(
            "expected tile-packed SVDQuant weight shape "
            f"(N/{KITCHEN_BLOCK_N}, K/{KITCHEN_GROUP_SIZE}, {expected_tail[0]}, {expected_tail[1]}), "
            f"got {tuple(weight.shape)}"
        )


def _validate_natural_weight_shape(weight: Any) -> tuple[int, int]:
    if int(weight.ndim) != 2:
        raise ValueError(f"expected natural SVDQuant weight shape (N, K/2), got {tuple(weight.shape)}")
    n, k_half = (int(weight.shape[0]), int(weight.shape[1]))
    k = k_half * 2
    if n % KITCHEN_BLOCK_N != 0:
        raise ValueError(f"N={n} is not divisible by {KITCHEN_BLOCK_N}")
    if k % KITCHEN_GROUP_SIZE != 0:
        raise ValueError(f"K={k} is not divisible by {KITCHEN_GROUP_SIZE}")
    return n, k


def pack_weight_tile(weight: Any):
    """Pack natural ``(N, K/2)`` signed-INT4 bytes to kitchen tile storage.

    The returned shape is ``(N/128, K/64, 32, 128)``.  Existing 4-D tile-packed
    weights are validated and returned as contiguous tensors.
    """
    if int(weight.ndim) == 4:
        _validate_weight_tile_shape(weight)
        return weight.contiguous()
    n, k = _validate_natural_weight_shape(weight)

    dense = unpack_signed_int4_pairs(weight)
    tiled = (
        dense.view(
            n // KITCHEN_BLOCK_N,
            KITCHEN_BLOCK_N // KITCHEN_INTERLEAVE,
            KITCHEN_INTERLEAVE,
            k // KITCHEN_GROUP_SIZE,
            KITCHEN_GROUP_SIZE,
        )
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    return pack_signed_int4_pairs(tiled, validate=False).view(
        n // KITCHEN_BLOCK_N,
        k // KITCHEN_GROUP_SIZE,
        KITCHEN_BLOCK_N // KITCHEN_INTERLEAVE,
        KITCHEN_INTERLEAVE * KITCHEN_GROUP_SIZE // 2,
    )


def unpack_weight_tile(weight: Any):
    """Unpack kitchen tile storage back to natural ``(N, K/2)`` signed-INT4 bytes."""
    _validate_weight_tile_shape(weight)
    n_blocks, k_groups = int(weight.shape[0]), int(weight.shape[1])
    tile = weight.view(
        n_blocks,
        k_groups,
        KITCHEN_BLOCK_N // KITCHEN_INTERLEAVE,
        KITCHEN_INTERLEAVE,
        KITCHEN_GROUP_SIZE // 2,
    )
    dense = unpack_signed_int4_pairs(tile)
    natural_dense = dense.permute(0, 2, 3, 1, 4).contiguous().view(
        n_blocks * KITCHEN_BLOCK_N,
        k_groups * KITCHEN_GROUP_SIZE,
    )
    return pack_signed_int4_pairs(natural_dense, validate=False)


def pack_n_axis(tensor: Any):
    """Tile-pack the N axis of a natural ``(N, *)`` tensor to ``(N/128, *, 128)``."""
    if int(tensor.ndim) >= 3:
        return tensor.contiguous()
    n = int(tensor.shape[0])
    if n % KITCHEN_BLOCK_N != 0:
        raise ValueError(f"N={n} is not divisible by {KITCHEN_BLOCK_N}")
    return tensor.view(n // KITCHEN_BLOCK_N, KITCHEN_BLOCK_N, *tensor.shape[1:]).movedim(1, -1).contiguous()


def unpack_n_axis(tensor: Any):
    """Unpack ``(N/128, *, 128)`` tile storage back to natural ``(N, *)`` layout."""
    if int(tensor.ndim) < 3:
        raise ValueError(f"expected tile-packed tensor rank >= 3, got {tuple(tensor.shape)}")
    n_blocks = int(tensor.shape[0])
    natural = tensor.movedim(-1, 1).contiguous()
    return natural.view(n_blocks * KITCHEN_BLOCK_N, *natural.shape[2:])


def normalize_svdquant_weight_scale_dtype(weight_scale: Any):
    """Return a runtime-compatible SVDQuant W4A4 weight-scale tensor.

    Kitchen tile-packed SVDQuant checkpoints store weight scales in the same
    low-precision floating domain used by the matrix-multiply runtime.  The
    runtime contract accepts fp16 or bf16 scales; fp32 scales are not valid
    checkpoint payloads because downstream kernels consume scale storage as
    low-precision elements.  Preserve existing fp16/bf16 tensors and cast fp32
    tensors to bf16 by default.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("torch is required for SVDQuant weight-scale dtype normalization") from exc

    dtype = getattr(weight_scale, "dtype", None)
    if dtype in (torch.float16, torch.bfloat16):
        return weight_scale
    if dtype == torch.float32:
        return weight_scale.to(torch.bfloat16)
    raise TypeError(f"SVDQuant weight_scale must be fp16, bf16, or fp32, got {dtype}")


def pack_weight_scale(weight_scale: Any):
    """Pack ``weight_scale`` from natural ``(K/64, N)`` to ``(N/128, K/64, 128)``.

    The packed tensor is guaranteed to use a runtime-compatible scale dtype:
    fp16/bf16 inputs are preserved and fp32 inputs are narrowed to bf16.
    """
    weight_scale = normalize_svdquant_weight_scale_dtype(weight_scale)
    if int(weight_scale.ndim) == 3:
        return weight_scale.contiguous()
    if int(weight_scale.ndim) != 2:
        raise ValueError(f"expected weight_scale rank 2 or 3, got {tuple(weight_scale.shape)}")
    return pack_n_axis(weight_scale.t().contiguous())


def unpack_weight_scale(weight_scale: Any):
    """Unpack ``weight_scale`` from tile storage back to natural ``(K/64, N)``."""
    if int(weight_scale.ndim) != 3:
        raise ValueError(f"expected tile-packed weight_scale rank 3, got {tuple(weight_scale.shape)}")
    return unpack_n_axis(weight_scale).t().contiguous()


def patch_svdquant_comfy_quant(comfy_quant: Any | None = None, **extra: object):
    """Return a quant-config tensor for kitchen tile-packed SVDQuant W4A4.

    ``comfy_quant`` is the checkpoint metadata tensor consumed by the target
    runtime.  The name is part of the artifact contract; this function only
    writes JSON metadata and does not import or call any runtime package.
    """
    config = decode_quant_config_tensor(comfy_quant) or {}
    config["format"] = SVDQUANT_W4A4_FORMAT_NAME
    config["layout"] = KITCHEN_TILEPACK_LAYOUT_NAME
    for key, value in extra.items():
        if value is None or value is False:
            config.pop(key, None)
        else:
            config[key] = value
    return encode_quant_config_tensor(config)


def _require_param_keys(params: Mapping[str, Any]) -> None:
    missing = [key for key in SVDQUANT_REQUIRED_PARAM_KEYS if key not in params]
    if missing:
        raise KeyError(f"missing SVDQuant parameter tensors: {', '.join(missing)}")


def to_kitchen_tile_packed_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one SVDQuant W4A4 parameter set to kitchen tile-packed layout.

    Expected natural input keys and shapes:

    - ``weight``: ``(N, K/2)`` int8 bytes containing signed INT4 pairs.
    - ``weight_scale``: ``(K/64, N)`` fp16/bf16, or fp32 to be stored as bf16.
    - ``smooth_factor``: ``(K,)`` fp16/bf16.
    - ``proj_down``: ``(K, R)`` fp16/bf16.
    - ``proj_up``: ``(N, R)`` fp16/bf16.
    - ``bias``: ``(N,)`` fp16/bf16, optional.
    - ``comfy_quant``: uint8 JSON tensor, optional.
    """
    _require_param_keys(params)
    out = {
        "weight": pack_weight_tile(params["weight"]),
        "weight_scale": pack_weight_scale(params["weight_scale"]),
        "smooth_factor": params["smooth_factor"].contiguous(),
        "proj_down": params["proj_down"].contiguous(),
        "proj_up": pack_n_axis(params["proj_up"]),
        "comfy_quant": patch_svdquant_comfy_quant(params.get("comfy_quant")),
    }
    if "bias" in params:
        out["bias"] = params["bias"].contiguous()
    return out


def svdquant_prefixes(keys: set[str], tensors: Mapping[str, Any]) -> list[str]:
    """Find state-dict prefixes whose quant metadata declares SVDQuant W4A4."""
    prefixes: list[str] = []
    for key in keys:
        if not key.endswith(".weight"):
            continue
        prefix = key[: -len(".weight")]
        config = decode_quant_config_tensor(tensors.get(f"{prefix}.comfy_quant"))
        if is_svdquant_quant_config(config):
            prefixes.append(prefix)
    return sorted(prefixes)


def repack_svdquant_state_dict(
    tensors: MutableMapping[str, Any],
    *,
    progress: ProgressCallback | None = None,
) -> list[str]:
    """Repack SVDQuant W4A4 layer tensors in a state dict in place."""
    prefixes = svdquant_prefixes(set(tensors), tensors)
    for index, prefix in enumerate(prefixes, 1):
        tensors[f"{prefix}.weight"] = pack_weight_tile(tensors[f"{prefix}.weight"])
        tensors[f"{prefix}.weight_scale"] = pack_weight_scale(tensors[f"{prefix}.weight_scale"])
        tensors[f"{prefix}.proj_up"] = pack_n_axis(tensors[f"{prefix}.proj_up"])
        tensors[f"{prefix}.comfy_quant"] = patch_svdquant_comfy_quant(tensors.get(f"{prefix}.comfy_quant"))
        if progress is not None:
            progress(index, len(prefixes), prefix)
    return prefixes
