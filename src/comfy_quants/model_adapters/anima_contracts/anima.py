"""Static Anima (cosmos_predict2 + llm_adapter) model contract.

Authored from ComfyUI's ``comfy/ldm/cosmos/predict2.py`` (Block / Attention /
GPT2FeedForward) and ``comfy/ldm/anima/model.py``. Anima is one architecture at two
cosmos sizes selected by ``model_channels`` (= ``x_embedder.proj.1.weight.shape[0]``):
2048 → 2B (28 blocks, 16 heads); 5120 → 14B (36 blocks, 40 heads)
(``comfy/model_detection.py:669-674``). All quantizable Linears are ``bias=False``.

Keys use the released ``net.`` prefix (verified against
``circlestone-labs/Anima`` ``split_files/diffusion_models/anima-base-v1.0.safetensors``):
all tensors are under ``net.`` (e.g. ``net.blocks.{n}.self_attn.q_proj.weight``).
Pre/post/embedder modules and the whole ``net.llm_adapter`` are kept high precision
(declared as coarse non-quantizable components); the per-layer selection recipe
(convert_to_quant ``anima`` preset) lives in the adapter's ``default_policy``.
"""

from __future__ import annotations

from typing import Any

from comfy_quants.model_adapters.anima_contracts.types import (
    AnimaModelContract,
    AnimaTransformerContract,
    ModuleContract,
    TensorContract,
)

CONTRACT_SCHEMA_VERSION = "anima_static_contract.v1"

# model_channels -> (num_blocks, num_heads), per comfy/model_detection.py:669-674
_SIZE_TABLE: dict[int, tuple[int, int]] = {2048: (28, 16), 5120: (36, 40)}


def _dims(model_channels: int) -> dict[str, int]:
    x = model_channels
    return {"X": x, "H": 4 * x, "C": 1024, "A": 256, "M": 3 * x, "head_dim": 128}


def _linear(name: str, out_dim: str, in_dim: str, *, quantizable: bool, module_type: str = "Linear", notes: str = "") -> ModuleContract:
    return ModuleContract(
        name_template=name,
        module_type=module_type,
        component="transformer",
        quantizable=quantizable,
        default_action="quantize" if quantizable else "keep_bf16",
        tensors=(
            TensorContract(
                name_template=f"{name}.weight",
                shape_template=(out_dim, in_dim),
                role="weight",
                scale_axis="out_features" if quantizable else None,
            ),
        ),
        notes=notes,
    )


def _rmsnorm(name: str, affine_dim: str, *, notes: str = "") -> ModuleContract:
    return ModuleContract(
        name_template=name,
        module_type="RMSNorm",
        component="transformer",
        quantizable=False,
        default_action="keep_bf16",
        tensors=(TensorContract(name_template=f"{name}.weight", shape_template=(affine_dim,), role="weight"),),
        notes=notes,
    )


def _component(name: str, module_type: str, component: str, notes: str) -> ModuleContract:
    return ModuleContract(
        name_template=name,
        module_type=module_type,
        component=component,
        quantizable=False,
        default_action="keep_bf16",
        tensors=(),
        notes=notes,
    )


def _block_modules(block_prefix: str) -> tuple[ModuleContract, ...]:
    p = f"{block_prefix}.{{block}}"
    return (
        # self-attention (RMSNorm q/k kept; q/k/v/output projections quantized)
        _linear(f"{p}.self_attn.q_proj", "X", "X", quantizable=True),
        _linear(f"{p}.self_attn.k_proj", "X", "X", quantizable=True),
        _linear(f"{p}.self_attn.v_proj", "X", "X", quantizable=True),
        _linear(f"{p}.self_attn.output_proj", "X", "X", quantizable=True),
        _rmsnorm(f"{p}.self_attn.q_norm", "head_dim", notes="kept high precision"),
        _rmsnorm(f"{p}.self_attn.k_norm", "head_dim", notes="kept high precision"),
        # cross-attention (k/v project from context_dim C=1024)
        _linear(f"{p}.cross_attn.q_proj", "X", "X", quantizable=True),
        _linear(f"{p}.cross_attn.k_proj", "X", "C", quantizable=True),
        _linear(f"{p}.cross_attn.v_proj", "X", "C", quantizable=True),
        _linear(f"{p}.cross_attn.output_proj", "X", "X", quantizable=True),
        _rmsnorm(f"{p}.cross_attn.q_norm", "head_dim", notes="kept high precision"),
        _rmsnorm(f"{p}.cross_attn.k_norm", "head_dim", notes="kept high precision"),
        # GPT2FeedForward (layer1/layer2)
        _linear(f"{p}.mlp.layer1", "H", "X", quantizable=True, module_type="GELULinear"),
        _linear(f"{p}.mlp.layer2", "X", "H", quantizable=True),
        # AdaLN-LoRA modulation (SiLU at index 0 has no params; .1 = X->A, .2 = A->3X)
        _linear(f"{p}.adaln_modulation_self_attn.1", "A", "X", quantizable=True, module_type="ModulationLinear"),
        _linear(f"{p}.adaln_modulation_self_attn.2", "M", "A", quantizable=True, module_type="ModulationLinear"),
        _linear(f"{p}.adaln_modulation_cross_attn.1", "A", "X", quantizable=True, module_type="ModulationLinear"),
        _linear(f"{p}.adaln_modulation_cross_attn.2", "M", "A", quantizable=True, module_type="ModulationLinear"),
        _linear(f"{p}.adaln_modulation_mlp.1", "A", "X", quantizable=True, module_type="ModulationLinear"),
        _linear(f"{p}.adaln_modulation_mlp.2", "M", "A", quantizable=True, module_type="ModulationLinear"),
    )


def _extra_components() -> tuple[ModuleContract, ...]:
    # Kept high precision; their tensors are copied verbatim by the exporter (they are
    # outside the `net.blocks.*` selection scope). The text encoder and VAE ship as
    # separate files, so they are not part of this diffusion_models contract.
    return (
        _component("net.t_embedder", "TimestepEmbedder", "transformer", "timestep embedding kept high precision"),
        _component("net.x_embedder", "PatchEmbed", "transformer", "patch embedding kept high precision"),
        _component("net.final_layer", "FinalLayer", "transformer", "final layer kept high precision"),
        _component("net.t_embedding_norm", "RMSNorm", "transformer", "kept high precision"),
        _component("net.llm_adapter", "LLMAdapter", "llm_adapter", "LLM text adapter kept high precision"),
    )


def build_anima_static_contract(
    *,
    model_channels: int = 2048,
    family: str = "anima",
    schema_version: str = CONTRACT_SCHEMA_VERSION,
    export_name: str = "Anima",
    metadata: dict[str, Any] | None = None,
) -> AnimaModelContract:
    if model_channels not in _SIZE_TABLE:
        raise ValueError(f"unsupported anima model_channels {model_channels}; expected one of {sorted(_SIZE_TABLE)}")
    num_blocks, num_heads = _SIZE_TABLE[model_channels]
    block_prefix = "net.blocks"
    transformer = AnimaTransformerContract(
        component="transformer",
        block_prefix=block_prefix,
        block_count=num_blocks,
        num_heads=num_heads,
        dims=_dims(model_channels),
        pre_modules=(),
        block_modules=_block_modules(block_prefix),
        post_modules=(),
    )
    contract_metadata: dict[str, Any] = {
        "export_name": export_name,
        "architecture": "cosmos_predict2_anima",
        "transformer_prefix": block_prefix,
        "model_channels": model_channels,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
        "head_dim": 128,
        "context_dim": 1024,
        "adaln_lora_dim": 256,
        "mlp_ratio": 4.0,
        "latent_format": "wan21",
    }
    if metadata:
        contract_metadata.update(metadata)
    return AnimaModelContract(
        family=family,
        schema_version=schema_version,
        artifact_target="comfyui",
        contract_mode="static_adapter_contract",
        preferred_format="fp8_e4m3",
        transformer=transformer,
        extra_components=_extra_components(),
        metadata=contract_metadata,
    )


def get_anima_static_contract(model_channels: int = 2048) -> AnimaModelContract:
    family = "anima" if model_channels == 2048 else "anima_14b"
    return build_anima_static_contract(
        model_channels=model_channels,
        family=family,
        metadata={"supported_model_ids": ("Comfy-Org/anima",), "calibration_profile": "text_to_image"},
    )
