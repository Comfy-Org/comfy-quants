"""Static Qwen-Image-Edit model contract."""

from __future__ import annotations

from comfy_quants.model_adapters.qwen_contracts.qwen_image import build_qwen_image_static_contract
from comfy_quants.model_adapters.qwen_contracts.types import QwenModelContract


CONTRACT_SCHEMA_VERSION = "qwen_image_edit_static_contract.v1"


def get_qwen_image_edit_static_contract() -> QwenModelContract:
    return build_qwen_image_static_contract(
        family="qwen_image_edit",
        schema_version=CONTRACT_SCHEMA_VERSION,
        export_name="Qwen-Image-Edit",
        include_visual_semantic_path=True,
        metadata={
            "supported_model_ids": ("Qwen/Qwen-Image-Edit", "Qwen/Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit-2511"),
            "calibration_profile": "image_edit",
            "reference_image_mode": "index_timestep_zero",
        },
    )
