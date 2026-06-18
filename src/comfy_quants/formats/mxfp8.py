"""MXFP8 (OCP microscaling FP8) reusable format declaration.

Storage format for the OFFLINE producer of **stock-ComfyUI-native** MXFP8
checkpoints: FP8-E4M3 weights with a per-32-element **E8M0** block scale stored in
the cuBLAS ``to_blocked`` swizzle layout, plus a per-layer ``comfy_quant`` marker.
Block math/swizzle live in :mod:`comfy_quants.formats.mxfp8_blocked`.

The consumer is **stock ComfyUI** via ``QUANT_ALGOS["mxfp8"]`` + ``TensorCoreMXFP8Layout``
(same per-layer ``comfy_quant`` handshake as the FP8 path) — NOT a downstream custom
node. The mxfp8 tensor-core matmul is gated at runtime on Blackwell (SM>=10) +
torch>=2.10 + comfy_kitchen; on other hardware ComfyUI silently dequantizes to the
compute dtype (loads & correct, no speedup). No ``input_scale`` is stored
(activations are quantized dynamically at runtime).
"""

from __future__ import annotations

from comfy_quants.formats.base import QuantFormatSpec
from comfy_quants.formats.mxfp8_blocked import BLOCK_SIZE
from comfy_quants.registry.global_registry import registry

__all__ = ["MXFP8_FORMAT_NAME", "MXFP8_FORMAT", "mxfp8_checkpoint_quant_config"]

MXFP8_FORMAT_NAME = "mxfp8"


def mxfp8_checkpoint_quant_config() -> dict[str, str]:
    """Return the per-layer ``comfy_quant`` marker payload.

    ComfyUI's loader only reads ``format`` (and optionally
    ``full_precision_matrix_mult``); ``"mxfp8"`` is the exact ``QUANT_ALGOS`` key.
    We do not set ``full_precision_matrix_mult`` — we want the mxfp8 kernel; on
    unsupported hardware ComfyUI auto-disables and dequantizes anyway.
    """
    return {"format": MXFP8_FORMAT_NAME}


MXFP8_FORMAT = QuantFormatSpec(
    name=MXFP8_FORMAT_NAME,
    storage_dtype="uint8",  # abstract bit-container (like fp8); on disk: weight is float8_e4m3fn, scale is uint8 E8M0
    bits=8,
    category="floating_point_block_scaled",
    scale_required=True,
    default_scale_granularity="block",
    compatible_families=("qwen_image", "qwen_image_edit", "qwen_image_layered", "anima", "anima_14b"),
    notes=(
        "MXFP8: float8_e4m3fn weights + per-32-element E8M0 block scale (OCP microscaling).",
        "weight_scale stored as uint8 in the cuBLAS to_blocked swizzle; loaded as float8_e8m0fnu.",
        "Loaded by stock ComfyUI QUANT_ALGOS[mxfp8] / TensorCoreMXFP8Layout (Blackwell SM>=10).",
        "No input_scale: activations quantized dynamically at runtime.",
    ),
    metadata={
        "block_size": BLOCK_SIZE,
        "weight_tensor": "weight",
        "weight_torch_dtype": "float8_e4m3fn",
        "scale_tensor": "weight_scale",
        "scale_dtype": "float8_e8m0fnu",
        "scale_storage_dtype": "uint8",
        "scale_layout": "to_blocked_swizzled",
        "marker_tensor": "comfy_quant",
        "marker_format": MXFP8_FORMAT_NAME,
        "no_input_scale": True,
        "downstream_loader": "stock ComfyUI QUANT_ALGOS[mxfp8] (Blackwell)",
    },
)


registry.register_format(MXFP8_FORMAT)
