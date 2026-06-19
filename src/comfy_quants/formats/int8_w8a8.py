"""INT8 W8A8 (+ optional ConvRot) reusable format declaration.

Storage format for the OFFLINE producer of ComfyUI-INT8-Fast prequantized
checkpoints: native ``torch.int8`` weights with a symmetric per-output-channel
``float32`` scale, plus a per-layer ``comfy_quant`` marker recording whether the
weights were ConvRot-rotated. Activations are quantized dynamically at runtime by
the downstream node (W8A8), so NO ``input_scale`` is stored.

The downstream consumer is the ComfyUI-INT8-Fast ``Int8TensorwiseOps`` loader
(not stock ComfyUI's native ``QUANT_ALGOS`` path) — same downstream-loader pattern
as the INT4 formats. Runtime (dynamic activation int8, online activation rotation,
Triton W8A8 kernel) is out of this library's scope.
"""

from __future__ import annotations

from comfy_quants.formats.base import QuantFormatSpec
from comfy_quants.formats.convrot import CONVROT_GROUP_SIZE
from comfy_quants.registry.global_registry import registry

__all__ = ["INT8_W8A8_FORMAT_NAME", "INT8_W8A8_FORMAT", "int8_w8a8_checkpoint_quant_config"]

INT8_W8A8_FORMAT_NAME = "int8_w8a8"


def int8_w8a8_checkpoint_quant_config(
    *, convrot: bool, convrot_groupsize: int = CONVROT_GROUP_SIZE, per_row: bool = True
) -> dict[str, bool | int]:
    """Return the per-layer ``comfy_quant`` marker payload.

    Keys/insertion-order match ComfyUI-INT8-Fast's save path
    (``{convrot[, convrot_groupsize], per_row}``); ``convrot_groupsize`` is only
    emitted when ``convrot`` is true.
    """
    conf: dict[str, bool | int] = {"convrot": bool(convrot)}
    if convrot:
        conf["convrot_groupsize"] = int(convrot_groupsize)
    conf["per_row"] = bool(per_row)
    return conf


INT8_W8A8_FORMAT = QuantFormatSpec(
    name=INT8_W8A8_FORMAT_NAME,
    storage_dtype="int8",
    bits=8,
    category="integer_weight_activation",
    scale_required=True,
    default_scale_granularity="per_channel",  # per output channel (axis = out_features)
    compatible_families=("qwen_image", "qwen_image_edit", "qwen_image_layered"),
    notes=(
        "INT8 W8A8: symmetric per-row int8 weights + dynamic int8 activations (runtime).",
        "Optional offline ConvRot (regular Hadamard) weight rotation; group size 256.",
        "Loaded by ComfyUI-INT8-Fast Int8TensorwiseOps; activations quantized/rotated at runtime.",
    ),
    metadata={
        "weight_tensor": "weight",
        "scale_tensor": "weight_scale",
        "marker_tensor": "comfy_quant",
        "weight_scale_shape": "per_row_2d_out_1",
        "convrot_group_size": CONVROT_GROUP_SIZE,
        "symmetric": True,
        "quant_min": -128,
        "quant_max": 127,
        "no_input_scale": True,
        "downstream_loader": "ComfyUI-INT8-Fast Int8TensorwiseOps",
    },
)


registry.register_format(INT8_W8A8_FORMAT)
