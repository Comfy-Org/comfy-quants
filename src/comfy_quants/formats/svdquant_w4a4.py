"""SVDQuant W4A4 reusable format declaration."""

from __future__ import annotations

from comfy_quants.formats.base import QuantFormatSpec
from comfy_quants.formats.kitchen_tilepack import KITCHEN_GROUP_SIZE, KITCHEN_TILEPACK_LAYOUT_NAME, SVDQUANT_W4A4_FORMAT_NAME
from comfy_quants.registry.global_registry import registry

LOWRANK_BRANCH_INPUT_BASIS_RAW = "raw"
LOWRANK_BRANCH_INPUT_BASIS_POST_SMOOTHING = "post_smoothing"
DEFAULT_LOWRANK_BRANCH_INPUT_BASIS = LOWRANK_BRANCH_INPUT_BASIS_RAW
DEFAULT_PROJ_DOWN_SMOOTH_FOLDED = True


def svdquant_w4a4_checkpoint_quant_config(
    *,
    act_unsigned: bool = False,
    lowrank_branch_input_basis: str = DEFAULT_LOWRANK_BRANCH_INPUT_BASIS,
    proj_down_smooth_folded: bool = DEFAULT_PROJ_DOWN_SMOOTH_FOLDED,
) -> dict[str, bool | str]:
    """Return layer-local SVDQuant W4A4 checkpoint metadata."""
    if lowrank_branch_input_basis not in {LOWRANK_BRANCH_INPUT_BASIS_RAW, LOWRANK_BRANCH_INPUT_BASIS_POST_SMOOTHING}:
        raise ValueError(f"unsupported low-rank branch input basis: {lowrank_branch_input_basis}")
    config: dict[str, bool | str] = {
        "format": SVDQUANT_W4A4_FORMAT_NAME,
        "layout": KITCHEN_TILEPACK_LAYOUT_NAME,
        "lowrank_branch_input_basis": lowrank_branch_input_basis,
        "proj_down_smooth_folded": bool(proj_down_smooth_folded),
    }
    if act_unsigned:
        config["act_unsigned"] = True
    return config


SVDQUANT_W4A4_FORMAT = QuantFormatSpec(
    name=SVDQUANT_W4A4_FORMAT_NAME,
    storage_dtype="int8",
    bits=4,
    category="integer_weight_activation_low_rank",
    scale_required=True,
    default_scale_granularity=f"group_size_{KITCHEN_GROUP_SIZE}",
    compatible_families=("qwen_image_edit",),
    notes=(
        "SVDQuant W4A4 stores signed INT4 weights and activations with high-precision low-rank side tensors.",
        "The kitchen tile-packed layout is a checkpoint storage contract; model adapters choose which layers use it.",
    ),
    metadata={
        "checkpoint_format": SVDQUANT_W4A4_FORMAT_NAME,
        "layout": KITCHEN_TILEPACK_LAYOUT_NAME,
        "group_size": KITCHEN_GROUP_SIZE,
        "weight_tensor": "weight",
        "scale_tensor": "weight_scale",
        "side_tensors": ["smooth_factor", "proj_down", "proj_up"],
        "optional_tensors": ["bias", "comfy_quant"],
    },
)


registry.register_format(SVDQUANT_W4A4_FORMAT)
