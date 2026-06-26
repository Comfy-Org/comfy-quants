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


_ANIMA_CONTRACTS: dict[str, dict[str, Any]] = {
    "anima": {
        "schema_version": "anima_contract.v1",
        "family": "anima",
        "artifact_target": "comfyui",
        "export_name": "Anima",
        "consumer_layout": "ComfyUI Anima",
        "model_contract_schema": "anima_static_contract.v1",
        "owner_module": "comfy_quants.model_adapters.anima",
    },
    "anima_14b": {
        "schema_version": "anima_14b_contract.v1",
        "family": "anima_14b",
        "artifact_target": "comfyui",
        "export_name": "Anima-14B",
        "consumer_layout": "ComfyUI Anima",
        "model_contract_schema": "anima_static_contract.v1",
        "owner_module": "comfy_quants.model_adapters.anima",
    },
}


_STOCK_DIT_CONTRACTS: dict[str, dict[str, Any]] = {
    "flux": {
        "schema_version": "flux_contract.v1",
        "family": "flux",
        "artifact_target": "comfyui",
        "export_name": "FLUX.1",
        "consumer_layout": "ComfyUI flux",
        "model_contract_schema": "flux_static_contract.v1",
        "owner_module": "comfy_quants.model_adapters.flux",
    },
    "flux2": {
        "schema_version": "flux2_contract.v1",
        "family": "flux2",
        "artifact_target": "comfyui",
        "export_name": "FLUX.2",
        "consumer_layout": "ComfyUI flux2",
        "model_contract_schema": "flux2_static_contract.v1",
        "owner_module": "comfy_quants.model_adapters.flux2",
    },
    "ltxv": {
        "schema_version": "ltxv_contract.v1",
        "family": "ltxv",
        "artifact_target": "comfyui",
        "export_name": "LTX-Video",
        "consumer_layout": "ComfyUI ltxv",
        "model_contract_schema": "ltxv_static_contract.v1",
        "owner_module": "comfy_quants.model_adapters.ltxv",
    },
    "ideogram4": {
        "schema_version": "ideogram4_contract.v1",
        "family": "ideogram4",
        "artifact_target": "comfyui",
        "export_name": "Ideogram-4",
        "consumer_layout": "ComfyUI ideogram4",
        "model_contract_schema": "ideogram4_static_contract.v1",
        "owner_module": "comfy_quants.model_adapters.ideogram4",
    },
}


def get_artifact_contract_index() -> ArtifactContractIndex:
    return ArtifactContractIndex(
        schema_version="artifact_contract_index.v1",
        artifact_target="comfyui",
        contract_source="comfy_quants",
        contract_mode="static_adapter_contract",
        contracts={**_QWEN_IMAGE_CONTRACTS, **_ANIMA_CONTRACTS, **_STOCK_DIT_CONTRACTS},
    )


def get_qwen_image_adapter_contract(*, edit: bool = False) -> dict[str, Any]:
    family = "qwen_image_edit" if edit else "qwen_image"
    return dict(_QWEN_IMAGE_CONTRACTS[family])


def get_qwen_image_layered_adapter_contract() -> dict[str, Any]:
    return dict(_QWEN_IMAGE_CONTRACTS["qwen_image_layered"])


def get_anima_adapter_contract(model_channels: int = 2048) -> dict[str, Any]:
    family = "anima" if model_channels == 2048 else "anima_14b"
    return dict(_ANIMA_CONTRACTS[family])


def get_stock_dit_adapter_contract(family: str) -> dict[str, Any]:
    return dict(_STOCK_DIT_CONTRACTS[family])
