"""Artifact contract declarations for ComfyUI exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ArtifactContractIndex:
    """Registered artifact contracts for one consumer target."""

    schema_version: str
    artifact_target: str
    contract_source: str
    contract_mode: str
    contracts: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_QWEN_IMAGE_CONTRACTS: dict[str, dict[str, Any]] = {
    "qwen_image": {
        "schema_version": "qwen_image_contract.v1",
        "family": "qwen_image",
        "artifact_target": "comfyui",
        "export_name": "Qwen-Image",
        "consumer_layout": "ComfyUI QwenImage",
        "model_contract_schema": "qwen_image_static_contract.v1",
        "owner_module": "comfy_quants.model_adapters.qwen_image",
    },
    "qwen_image_edit": {
        "schema_version": "qwen_image_edit_contract.v1",
        "family": "qwen_image_edit",
        "artifact_target": "comfyui",
        "export_name": "Qwen-Image-Edit",
        "consumer_layout": "ComfyUI QwenImage edit",
        "model_contract_schema": "qwen_image_edit_static_contract.v1",
        "owner_module": "comfy_quants.model_adapters.qwen_image_edit",
    },
    "qwen_image_layered": {
        "schema_version": "qwen_image_layered_contract.v1",
        "family": "qwen_image_layered",
        "artifact_target": "comfyui",
        "export_name": "Qwen-Image-Layered",
        "consumer_layout": "ComfyUI QwenImage layered",
        "model_contract_schema": "qwen_image_layered_static_contract.v1",
        "owner_module": "comfy_quants.model_adapters.qwen_image_layered",
    },
}


def get_artifact_contract_index() -> ArtifactContractIndex:
    return ArtifactContractIndex(
        schema_version="artifact_contract_index.v1",
        artifact_target="comfyui",
        contract_source="comfy_quants",
        contract_mode="static_adapter_contract",
        contracts=_QWEN_IMAGE_CONTRACTS,
    )


def get_qwen_image_adapter_contract(*, edit: bool = False) -> dict[str, Any]:
    family = "qwen_image_edit" if edit else "qwen_image"
    return dict(_QWEN_IMAGE_CONTRACTS[family])


def get_qwen_image_layered_adapter_contract() -> dict[str, Any]:
    return dict(_QWEN_IMAGE_CONTRACTS["qwen_image_layered"])
