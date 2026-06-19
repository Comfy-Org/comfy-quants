"""Static Anima model contracts."""

from __future__ import annotations

from comfy_quants.model_adapters.anima_contracts.anima import (
    CONTRACT_SCHEMA_VERSION,
    build_anima_static_contract,
    get_anima_static_contract,
)

__all__ = ["CONTRACT_SCHEMA_VERSION", "build_anima_static_contract", "get_anima_static_contract"]
