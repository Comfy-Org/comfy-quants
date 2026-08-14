"""MiniMax Music 3 adapters for stock-ComfyUI native quantized checkpoints.

MiniMax Music 3 has two separately loaded neural networks: a 36-block autoregressive
text/audio encoder and a 36-block diffusion transformer (DiT).  The published ComfyUI
artifacts keep these networks in separate safetensors files, so this module registers
one family per file.  Both use ComfyUI's bare repacked tensor names.

The default policy deliberately quantizes only transformer-block Linear weights.  It
keeps token/code embeddings, logits heads, norms, positional/timestep parameters,
convolutions, input/output projections, and the DAV high precision.  Every selected
input dimension is divisible by the NVFP4 block size (16).

Contracts were authored from ComfyUI's ``comfy/ldm/minimax_music/{ar,dit}.py`` and
verified against safetensors headers at Comfy-Org/MiniMax-Music-3 revision
``6444666eb6edfb2c7fcab5f8b81da8b84b4b17b6``.
"""

from __future__ import annotations

from typing import ClassVar

from comfy_quants.comfy.stock_dit_contract import stock_dit_artifact_contract_metadata
from comfy_quants.core.policy import QuantPolicy
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.stock_dit_contract import (
    BlockGroup,
    StockDitContract,
    build_stock_dit_graph,
    kept_component,
    linear,
    summarize_stock_dit_graph,
)

MODEL_REPOSITORY = "Comfy-Org/MiniMax-Music-3"
MODEL_REVISION = "6444666eb6edfb2c7fcab5f8b81da8b84b4b17b6"

DIT_CONTRACT_SCHEMA_VERSION = "minimax_music3_dit_static_contract.v1"
TEXT_ENCODER_CONTRACT_SCHEMA_VERSION = "minimax_music3_text_encoder_static_contract.v1"

_DIT_BLOCKS = 36
_DIT_HIDDEN = 2048
_DIT_INTERMEDIATE = 8192

_AR_BLOCKS = 36
_AR_DECODER_BLOCKS = 4
_AR_HIDDEN = 4096
_AR_INTERMEDIATE = 12288
_AR_DECODER_INTERMEDIATE = 6144


def _dit_dims() -> dict[str, int]:
    return {
        "D": _DIT_HIDDEN,
        "QKV": 3 * _DIT_HIDDEN,
        "FF": _DIT_INTERMEDIATE,
        "FF_GATE_VALUE": 2 * _DIT_INTERMEDIATE,
    }


def _dit_block_modules() -> tuple:
    prefix = "diffusion_transformer.transformer.layers.{block}"
    return (
        linear(f"{prefix}.self_attn.to_qkv", "QKV", "D"),
        linear(f"{prefix}.self_attn.to_out", "D", "D"),
        linear(
            f"{prefix}.ff.ff.0.proj", "FF_GATE_VALUE", "D", module_type="SwiGLULinear"
        ),
        linear(f"{prefix}.ff.ff.2", "D", "FF"),
    )


def build_minimax_music3_dit_static_contract() -> StockDitContract:
    return StockDitContract(
        family="minimax_music3_dit",
        schema_version=DIT_CONTRACT_SCHEMA_VERSION,
        preferred_format="nvfp4",
        dims=_dit_dims(),
        block_groups=(
            BlockGroup(
                prefix="diffusion_transformer.transformer.layers",
                count=_DIT_BLOCKS,
                modules=_dit_block_modules(),
            ),
        ),
        extra_components=(
            kept_component(
                "latent_conditioners",
                "Conv1d",
                "transformer",
                "conditioning convolution kept high precision",
            ),
            kept_component(
                "diffusion_transformer.preprocess_conv",
                "Conv1d",
                "transformer",
                "input convolution kept high precision",
            ),
            kept_component(
                "diffusion_transformer.postprocess_conv",
                "Conv1d",
                "transformer",
                "output convolution kept high precision",
            ),
            kept_component(
                "diffusion_transformer.to_timestep_embed",
                "Sequential",
                "transformer",
                "timestep embedding kept high precision",
            ),
            kept_component(
                "diffusion_transformer.transformer.project_in",
                "Linear",
                "transformer",
                "input projection kept high precision",
            ),
            kept_component(
                "diffusion_transformer.transformer.project_out",
                "Linear",
                "transformer",
                "output projection kept high precision",
            ),
        ),
        metadata={
            "export_name": "MiniMax-Music-3 DiT",
            "architecture": "minimax_music3_dit",
            "hidden_size": _DIT_HIDDEN,
            "intermediate_size": _DIT_INTERMEDIATE,
            "num_layers": _DIT_BLOCKS,
            "source_revision": MODEL_REVISION,
            "supported_model_ids": (MODEL_REPOSITORY,),
        },
    )


def _ar_dims() -> dict[str, int]:
    return {
        "D": _AR_HIDDEN,
        "QKV": 3 * _AR_HIDDEN // 2,
        "FF": _AR_INTERMEDIATE,
        "FF_GATE_UP": 2 * _AR_INTERMEDIATE,
        "DECODER_QKV": 3 * _AR_HIDDEN,
        "DECODER_FF": _AR_DECODER_INTERMEDIATE,
        "DECODER_FF_GATE_UP": 2 * _AR_DECODER_INTERMEDIATE,
    }


def _ar_block_modules() -> tuple:
    prefix = "model.layers.{block}"
    return (
        linear(f"{prefix}.self_attn.qkv_proj", "QKV", "D", component="text_encoder"),
        linear(f"{prefix}.self_attn.o_proj", "D", "D", component="text_encoder"),
        linear(
            f"{prefix}.mlp.gate_up_proj",
            "FF_GATE_UP",
            "D",
            module_type="SwiGLULinear",
            component="text_encoder",
        ),
        linear(f"{prefix}.mlp.down_proj", "D", "FF", component="text_encoder"),
    )


def _ar_decoder_block_modules() -> tuple:
    prefix = "model.audio_decoder.layers.{block}"
    return (
        linear(
            f"{prefix}.self_attn.qkv_proj", "DECODER_QKV", "D", component="text_encoder"
        ),
        linear(f"{prefix}.self_attn.o_proj", "D", "D", component="text_encoder"),
        linear(
            f"{prefix}.mlp.gate_up_proj",
            "DECODER_FF_GATE_UP",
            "D",
            module_type="SwiGLULinear",
            component="text_encoder",
        ),
        linear(f"{prefix}.mlp.down_proj", "D", "DECODER_FF", component="text_encoder"),
    )


def build_minimax_music3_text_encoder_static_contract() -> StockDitContract:
    return StockDitContract(
        family="minimax_music3_text_encoder",
        schema_version=TEXT_ENCODER_CONTRACT_SCHEMA_VERSION,
        preferred_format="nvfp4",
        dims=_ar_dims(),
        block_groups=(
            BlockGroup(
                prefix="model.layers", count=_AR_BLOCKS, modules=_ar_block_modules()
            ),
            BlockGroup(
                prefix="model.audio_decoder.layers",
                count=_AR_DECODER_BLOCKS,
                modules=_ar_decoder_block_modules(),
            ),
        ),
        extra_components=(
            kept_component(
                "model.embed_tokens_prefill",
                "Embedding",
                "text_encoder",
                "prefill token embedding kept high precision",
            ),
            kept_component(
                "model.embed_tokens_audio",
                "Embedding",
                "text_encoder",
                "audio token embedding kept high precision",
            ),
            kept_component(
                "model.audio_extra_embedding",
                "Embedding",
                "text_encoder",
                "RVQ code embedding kept high precision",
            ),
            kept_component(
                "model.lm_head_pruned",
                "Linear",
                "text_encoder",
                "audio logits head kept high precision",
            ),
            kept_component(
                "model.audio_decoder.audio_heads",
                "ModuleList",
                "text_encoder",
                "RVQ logits heads kept high precision",
            ),
            kept_component(
                "model.audio_decoder.projection",
                "Linear",
                "text_encoder",
                "RVQ input projection kept high precision",
            ),
        ),
        metadata={
            "export_name": "MiniMax-Music-3 text encoder",
            "architecture": "minimax_music3_autoregressive_audio",
            "hidden_size": _AR_HIDDEN,
            "intermediate_size": _AR_INTERMEDIATE,
            "num_layers": _AR_BLOCKS,
            "decoder_intermediate_size": _AR_DECODER_INTERMEDIATE,
            "decoder_num_layers": _AR_DECODER_BLOCKS,
            "source_revision": MODEL_REVISION,
            "supported_model_ids": (MODEL_REPOSITORY,),
        },
    )


class MiniMaxMusic3DitAdapter:
    """Adapter for the separately loaded MiniMax Music 3 diffusion transformer."""

    family = "minimax_music3_dit"
    supported_model_ids: ClassVar[list[str]] = [MODEL_REPOSITORY]

    def inspect(self, source: ModelSource):
        contract = build_minimax_music3_dit_static_contract()
        graph = build_stock_dit_graph(
            contract,
            source,
            artifact_metadata=stock_dit_artifact_contract_metadata(self.family),
        )
        return summarize_stock_dit_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "nvfp4") -> QuantPolicy:
        return QuantPolicy(
            name="minimax_music3_dit_default",
            algorithm="nvfp4",
            target_dtype=target_dtype,
            include=["diffusion_transformer.transformer.layers.*"],
            exclude=[],
            keep_components=["text_encoder", "vae"],
        )


class MiniMaxMusic3TextEncoderAdapter:
    """Adapter for the separately loaded MiniMax Music 3 AR text/audio encoder."""

    family = "minimax_music3_text_encoder"
    supported_model_ids: ClassVar[list[str]] = [MODEL_REPOSITORY]

    def inspect(self, source: ModelSource):
        contract = build_minimax_music3_text_encoder_static_contract()
        graph = build_stock_dit_graph(
            contract,
            source,
            artifact_metadata=stock_dit_artifact_contract_metadata(self.family),
        )
        return summarize_stock_dit_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "nvfp4") -> QuantPolicy:
        return QuantPolicy(
            name="minimax_music3_text_encoder_default",
            algorithm="nvfp4",
            target_dtype=target_dtype,
            include=["model.layers.*", "model.audio_decoder.layers.*"],
            exclude=[],
            keep_components=["vae"],
        )


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(MiniMaxMusic3DitAdapter())
registry.register_adapter(MiniMaxMusic3TextEncoderAdapter())
