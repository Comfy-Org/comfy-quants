"""Qwen-Image artifact contract metadata."""

from __future__ import annotations

from typing import Any

from comfy_quants.comfy.artifact_contracts import get_artifact_contract_index, get_qwen_image_adapter_contract


def qwen_image_artifact_contract_metadata(*, edit: bool) -> dict[str, Any]:
    contract_index = get_artifact_contract_index()
    contract = get_qwen_image_adapter_contract(edit=edit)
    metadata: dict[str, Any] = {
        "artifact_target": contract_index.artifact_target,
        "contract_source": contract_index.contract_source,
        "contract_mode": contract_index.contract_mode,
        "artifact_contract": contract,
        "adapter_scope": "qwen_image_edit" if edit else "qwen_image",
    }
    if edit:
        metadata["edit_contract"] = {
            "base_structure": "Qwen-Image transformer contract",
            "extra_inputs": ["prompt", "input_image", "edit_instruction", "reference_latents"],
            "calibration_requirement": "cover semantic edits and appearance-preserving edits",
            "reference_image_mode": "index_timestep_zero",
        }
    return metadata
