"""Artifact contract metadata for stock-ComfyUI DiT exports (flux / ltxv / ideogram4)."""

from __future__ import annotations

from typing import Any

from comfy_quants.comfy.artifact_contracts import get_artifact_contract_index, get_stock_dit_adapter_contract


def stock_dit_artifact_contract_metadata(family: str) -> dict[str, Any]:
    contract_index = get_artifact_contract_index()
    contract = get_stock_dit_adapter_contract(family)
    return {
        "artifact_target": contract_index.artifact_target,
        "contract_source": contract_index.contract_source,
        "contract_mode": contract_index.contract_mode,
        "artifact_contract": contract,
        "adapter_scope": family,
    }
