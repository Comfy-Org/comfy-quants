"""AWQ W4A16 reusable format declaration."""

from __future__ import annotations

from comfy_quants.formats.base import QuantFormatSpec
from comfy_quants.registry.global_registry import registry

AWQ_W4A16_FORMAT_NAME = "awq_w4a16"
AWQ_W4A16_GROUP_SIZE = 64


def awq_w4a16_checkpoint_quant_config(*, group_size: int = AWQ_W4A16_GROUP_SIZE) -> dict[str, int | str]:
    """Return checkpoint metadata for AWQ W4A16 tensors."""
    return {"format": AWQ_W4A16_FORMAT_NAME, "group_size": int(group_size)}


AWQ_W4A16_FORMAT = QuantFormatSpec(
    name=AWQ_W4A16_FORMAT_NAME,
    storage_dtype="int8",
    bits=4,
    category="integer_weight_only",
    scale_required=True,
    default_scale_granularity=f"group_size_{AWQ_W4A16_GROUP_SIZE}",
    compatible_families=("qwen_image_edit",),
    notes=(
        "AWQ W4A16 stores 4-bit weights with high-precision activations.",
        "Checkpoint dequantization is (uint4_weight - 8) * weight_scale + weight_zero.",
        "For Qwen-Image-Edit INT4 bundles this format is used by modulation linear layers when the target runtime supports it.",
    ),
    metadata={
        "checkpoint_format": AWQ_W4A16_FORMAT_NAME,
        "group_size": AWQ_W4A16_GROUP_SIZE,
        "weight_tensor": "weight",
        "scale_tensor": "weight_scale",
        "zero_tensor": "weight_zero",
        "optional_tensors": ["bias", "comfy_quant"],
    },
)


registry.register_format(AWQ_W4A16_FORMAT)
