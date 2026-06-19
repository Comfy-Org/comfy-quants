"""Static Qwen-Image-Layered model contract."""

from __future__ import annotations

from typing import Any

from comfy_quants.model_adapters.qwen_contracts.types import (
    ModuleContract,
    QwenModelContract,
    TensorContract,
    TransformerContract,
)

CONTRACT_SCHEMA_VERSION = "qwen_image_layered_static_contract.v1"


def _linear_tensors(name: str, out_dim: str, in_dim: str, *, bias: bool = True, scale_axis: str | None = None) -> tuple[TensorContract, ...]:
    tensors: list[TensorContract] = [
        TensorContract(
            name_template=f"{name}.weight",
            shape_template=(out_dim, in_dim),
            role="weight",
            scale_axis=scale_axis,
        )
    ]
    if bias:
        tensors.append(TensorContract(name_template=f"{name}.bias", shape_template=(out_dim,), role="bias"))
    return tuple(tensors)


def _linear_module(
    name: str,
    component: str,
    out_dim: str,
    in_dim: str,
    *,
    quantizable: bool,
    module_type: str = "Linear",
    notes: str = "",
) -> ModuleContract:
    return ModuleContract(
        name_template=name,
        module_type=module_type,
        component=component,
        quantizable=quantizable,
        default_action="quantize" if quantizable else "keep_bf16",
        tensors=_linear_tensors(name, out_dim, in_dim, scale_axis="out_features" if quantizable else None),
        notes=notes,
    )


def _norm_module(name: str, component: str, *, module_type: str, affine_dim: str | None = None, notes: str = "") -> ModuleContract:
    tensors = ()
    if affine_dim is not None:
        tensors = (TensorContract(name_template=f"{name}.weight", shape_template=(affine_dim,), role="weight"),)
    return ModuleContract(
        name_template=name,
        module_type=module_type,
        component=component,
        quantizable=False,
        default_action="keep_bf16",
        tensors=tensors,
        notes=notes,
    )


def _embedding_module(name: str, component: str, *, num_embeddings: int, embedding_dim: str, notes: str = "") -> ModuleContract:
    tensors = (
        TensorContract(
            name_template=f"{name}.weight",
            shape_template=(num_embeddings, embedding_dim),
            role="weight",
        ),
    )
    return ModuleContract(
        name_template=name,
        module_type="Embedding",
        component=component,
        quantizable=False,
        default_action="keep_bf16",
        tensors=tensors,
        notes=notes,
    )


def _extra_component(name: str, module_type: str, component: str, notes: str) -> ModuleContract:
    return ModuleContract(
        name_template=name,
        module_type=module_type,
        component=component,
        quantizable=False,
        default_action="keep_bf16",
        tensors=(),
        notes=notes,
    )


def _pre_modules(component: str) -> tuple[ModuleContract, ...]:
    return (
        _linear_module(
            "time_text_embed.timestep_embedder.linear_1",
            component,
            "hidden_size",
            "timestep_projection_size",
            quantizable=False,
            module_type="TimestepLinear",
            notes="timestep embedding projection kept high precision",
        ),
        _linear_module(
            "time_text_embed.timestep_embedder.linear_2",
            component,
            "hidden_size",
            "hidden_size",
            quantizable=False,
            module_type="TimestepLinear",
            notes="timestep embedding projection kept high precision",
        ),
        _embedding_module(
            "time_text_embed.addition_t_embedding",
            component,
            num_embeddings=2,
            embedding_dim="hidden_size",
            notes="addition timestep embedding for layered generation kept high precision",
        ),
        _norm_module(
            "txt_norm",
            component,
            module_type="RMSNorm",
            affine_dim="joint_attention_dim",
            notes="text normalization kept high precision",
        ),
        _linear_module(
            "img_in",
            component,
            "hidden_size",
            "in_channels",
            quantizable=False,
            notes="input projection kept high precision",
        ),
        _linear_module(
            "txt_in",
            component,
            "hidden_size",
            "joint_attention_dim",
            quantizable=False,
            notes="text input projection kept high precision",
        ),
    )


def _block_modules(component: str, block_prefix: str) -> tuple[ModuleContract, ...]:
    prefix = f"{block_prefix}.{{block}}"
    return (
        _linear_module(
            f"{prefix}.img_mod.1",
            component,
            "modulation_size",
            "hidden_size",
            quantizable=True,
            module_type="ModulationLinear",
            notes="image-stream modulation linear",
        ),
        _linear_module(f"{prefix}.img_mlp.net.0.proj", component, "intermediate_size", "hidden_size", quantizable=True, module_type="GELULinear"),
        _linear_module(f"{prefix}.img_mlp.net.2", component, "hidden_size", "intermediate_size", quantizable=True),
        _linear_module(
            f"{prefix}.txt_mod.1",
            component,
            "modulation_size",
            "hidden_size",
            quantizable=True,
            module_type="ModulationLinear",
            notes="text-stream modulation linear",
        ),
        _linear_module(f"{prefix}.txt_mlp.net.0.proj", component, "intermediate_size", "hidden_size", quantizable=True, module_type="GELULinear"),
        _linear_module(f"{prefix}.txt_mlp.net.2", component, "hidden_size", "intermediate_size", quantizable=True),
        _norm_module(f"{prefix}.attn.norm_q", component, module_type="RMSNorm", affine_dim="attention_head_dim", notes="attention Q normalization kept high precision"),
        _norm_module(f"{prefix}.attn.norm_k", component, module_type="RMSNorm", affine_dim="attention_head_dim", notes="attention K normalization kept high precision"),
        _norm_module(f"{prefix}.attn.norm_added_q", component, module_type="RMSNorm", affine_dim="attention_head_dim", notes="text attention Q normalization kept high precision"),
        _norm_module(f"{prefix}.attn.norm_added_k", component, module_type="RMSNorm", affine_dim="attention_head_dim", notes="text attention K normalization kept high precision"),
        _linear_module(f"{prefix}.attn.to_q", component, "hidden_size", "hidden_size", quantizable=True),
        _linear_module(f"{prefix}.attn.to_k", component, "hidden_size", "hidden_size", quantizable=True),
        _linear_module(f"{prefix}.attn.to_v", component, "hidden_size", "hidden_size", quantizable=True),
        _linear_module(f"{prefix}.attn.add_q_proj", component, "hidden_size", "hidden_size", quantizable=True),
        _linear_module(f"{prefix}.attn.add_k_proj", component, "hidden_size", "hidden_size", quantizable=True),
        _linear_module(f"{prefix}.attn.add_v_proj", component, "hidden_size", "hidden_size", quantizable=True),
        _linear_module(f"{prefix}.attn.to_out.0", component, "hidden_size", "hidden_size", quantizable=True),
        _linear_module(f"{prefix}.attn.to_add_out", component, "hidden_size", "hidden_size", quantizable=True),
    )


def _post_modules(component: str) -> tuple[ModuleContract, ...]:
    return (
        _linear_module(
            "norm_out.linear",
            component,
            "final_modulation_size",
            "hidden_size",
            quantizable=False,
            module_type="FinalModulationLinear",
            notes="final modulation kept high precision",
        ),
        _linear_module(
            "proj_out",
            component,
            "output_projection_size",
            "hidden_size",
            quantizable=False,
            module_type="OutputLinear",
            notes="final output projection kept high precision",
        ),
    )


def _extra_components() -> tuple[ModuleContract, ...]:
    return (
        _extra_component("vae", "VAE", "vae", "VAE component kept high precision"),
        _extra_component("text_encoders.qwen25_7b", "Qwen25TextEncoder", "text_encoder", "text encoder component kept high precision"),
    )


def get_qwen_image_layered_static_contract() -> QwenModelContract:
    component = "transformer"
    block_prefix = "transformer_blocks"
    transformer = TransformerContract(
        component=component,
        block_prefix=block_prefix,
        block_count=60,
        hidden_size=3072,
        intermediate_size=12288,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=3584,
        in_channels=64,
        out_channels=16,
        patch_size=2,
        timestep_projection_size=256,
        pre_modules=_pre_modules(component),
        block_modules=_block_modules(component, block_prefix),
        post_modules=_post_modules(component),
    )
    metadata: dict[str, Any] = {
        "export_name": "Qwen-Image-Layered",
        "architecture": "qwen_image_layered_transformer_2d",
        "transformer_prefix": block_prefix,
        "text_encoder": "qwen25_7b",
        "latent_format": "wan21",
        "supported_model_ids": ("Qwen/Qwen-Image-Layered",),
        "calibration_profile": "text_to_image_layered",
    }
    return QwenModelContract(
        family="qwen_image_layered",
        schema_version=CONTRACT_SCHEMA_VERSION,
        artifact_target="comfyui",
        contract_mode="static_adapter_contract",
        preferred_format="fp8_e4m3",
        transformer=transformer,
        extra_components=_extra_components(),
        metadata=metadata,
    )
