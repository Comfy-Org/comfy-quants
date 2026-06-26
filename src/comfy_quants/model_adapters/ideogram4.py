"""Ideogram 4 model adapter (NextDiT/Lumina2-family single-stream DiT).

Authored from ComfyUI ``comfy/ldm/ideogram4/model.py`` (``image_model="ideogram4"``,
detection key ``embed_image_indicator.weight``). Ideogram 4 is an open-weight 9.3B
single-stream DiT that packs ``[text, image]`` tokens into one sequence: ``layers.N``
each with fused-qkv self-attention (``attention.qkv`` / ``attention.o``), a SwiGLU
feed-forward (``feed_forward.w1/w2/w3``, from ``comfy.ldm.lumina.model.FeedForward``),
and an ``adaln_modulation`` Linear. Config: emb_dim=4608 (18 heads x 256), 34 layers,
intermediate=12288, adaln_dim=512. Bare ``layers.`` keys. Stock-ComfyUI-native target
(fp8_e4m3/fp8_e5m2/mxfp8/nvfp4).

Text encoder is Qwen3-VL-8B and the VAE is the FLUX.2 VAE (separate files, kept high
precision). Default policy quantizes every block Linear and keeps the input/llm/time/
adaln projections, the image-indicator embedding, and the final layer high precision.
"""

from __future__ import annotations

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

CONTRACT_SCHEMA_VERSION = "ideogram4_static_contract.v1"

# Ideogram4 config (comfy/ldm/ideogram4/model.py defaults).
_NUM_HEADS = 18
_HEAD_DIM = 256
_EMB = _NUM_HEADS * _HEAD_DIM  # 4608
_INTERMEDIATE = 12288
_ADALN = 512
_LAYERS = 34


def _dims() -> dict[str, int]:
    return {
        "E": _EMB,  # 4608
        "QKV": 3 * _EMB,  # 13824
        "INTER": _INTERMEDIATE,  # 12288  (SwiGLU hidden)
        "ADALN": _ADALN,  # 512
        "MOD4": 4 * _EMB,  # 18432  (adaln_modulation -> scale/gate for msa+mlp)
    }


def _block_modules() -> tuple:
    p = "layers.{block}"
    return (
        # self-attention: fused qkv (bias-free) + output projection
        linear(f"{p}.attention.qkv", "QKV", "E"),
        linear(f"{p}.attention.o", "E", "E"),
        # SwiGLU feed-forward (lumina FeedForward: w2(silu(w1) * w3))
        linear(f"{p}.feed_forward.w1", "INTER", "E", module_type="SwiGLULinear"),
        linear(f"{p}.feed_forward.w2", "E", "INTER"),
        linear(f"{p}.feed_forward.w3", "INTER", "E", module_type="SwiGLULinear"),
        # adaln modulation (in_features = adaln_dim 512)
        linear(f"{p}.adaln_modulation", "MOD4", "ADALN", module_type="ModulationLinear"),
    )


def _extra_components() -> tuple:
    return (
        kept_component("input_proj", "Linear", "transformer", "latent input projection kept high precision"),
        kept_component("llm_cond_norm", "RMSNorm", "transformer", "kept high precision"),
        kept_component("llm_cond_proj", "Linear", "transformer", "Qwen3-VL feature projection kept high precision"),
        kept_component("t_embedding", "Ideogram4EmbedScalar", "transformer", "timestep embedding kept high precision"),
        kept_component("adaln_proj", "Linear", "transformer", "adaln projection kept high precision"),
        kept_component("embed_image_indicator", "Embedding", "transformer", "image-indicator embedding kept high precision"),
        kept_component("final_layer", "Ideogram4FinalLayer", "transformer", "final layer kept high precision"),
    )


def build_ideogram4_static_contract() -> StockDitContract:
    return StockDitContract(
        family="ideogram4",
        schema_version=CONTRACT_SCHEMA_VERSION,
        preferred_format="fp8_e4m3",
        dims=_dims(),
        block_groups=(BlockGroup(prefix="layers", count=_LAYERS, modules=_block_modules()),),
        extra_components=_extra_components(),
        metadata={
            "export_name": "Ideogram-4",
            "architecture": "ideogram4_nextdit",
            "emb_dim": _EMB,
            "num_heads": _NUM_HEADS,
            "head_dim": _HEAD_DIM,
            "intermediate_size": _INTERMEDIATE,
            "adaln_dim": _ADALN,
            "num_layers": _LAYERS,
            "llm_features_dim": 53248,
            "text_encoder": "qwen3vl_8b",
            "vae": "flux2-vae",
            "supported_model_ids": ("Comfy-Org/Ideogram-4",),
        },
    )


class Ideogram4Adapter:
    """Adapter for the open-weight Ideogram 4 (9.3B) single-stream DiT."""

    family = "ideogram4"
    supported_model_ids = ["Comfy-Org/Ideogram-4"]

    def inspect(self, source: ModelSource):
        contract = build_ideogram4_static_contract()
        graph = build_stock_dit_graph(
            contract,
            source,
            artifact_metadata=stock_dit_artifact_contract_metadata("ideogram4"),
        )
        return summarize_stock_dit_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "fp8_e4m3") -> QuantPolicy:
        return QuantPolicy(
            name="ideogram4_default",
            algorithm="fp8_static",
            target_dtype=target_dtype,
            include=["layers.*"],
            exclude=[],
            keep_components=["text_encoder", "vae"],
        )


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(Ideogram4Adapter())
