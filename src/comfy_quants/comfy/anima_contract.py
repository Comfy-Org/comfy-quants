"""Anima artifact contract metadata for ComfyUI exports."""

from __future__ import annotations

from typing import Any

from comfy_quants.comfy.artifact_contracts import get_anima_adapter_contract, get_artifact_contract_index


def anima_artifact_contract_metadata(model_channels: int = 2048) -> dict[str, Any]:
    contract_index = get_artifact_contract_index()
    contract = get_anima_adapter_contract(model_channels)
    return {
        "artifact_target": contract_index.artifact_target,
        "contract_source": contract_index.contract_source,
        "contract_mode": contract_index.contract_mode,
        "artifact_contract": contract,
        "adapter_scope": "anima" if model_channels == 2048 else "anima_14b",
    }
