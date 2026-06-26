"""LTX-Video (LTXV) model adapter (Lightricks DiT, stock-ComfyUI ``image_model="ltxv"``).

Authored from ComfyUI ``comfy/ldm/lightricks/model.py``. LTXV is a single-stream video
DiT: ``transformer_blocks.N`` each with self-attention (``attn1``), cross-attention to
the text caption (``attn2``), and a GELU feed-forward (``ff.net.0.proj`` / ``ff.net.2``).
LTXV-2B (0.9.x): inner_dim=2048, 32 heads, head_dim=64, 28 blocks, cross_attention_dim=
2048, ff mult=4 (ff_inner=8192). Bare ``transformer_blocks.`` keys. Stock-ComfyUI-native
target (fp8_e4m3/fp8_e5m2/mxfp8/nvfp4).

Default policy quantizes every block Linear and keeps the patch / adaln / caption /
output projections and the ``scale_shift_table`` parameters high precision. The audio
variant (``ltxav``: 48 layers, inner_dim 4096) is a separate larger architecture and is
out of scope here.
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

CONTRACT_SCHEMA_VERSION = "ltxv_static_contract.v1"

# LTXV-2B config (comfy/model_detection.py: num_heads=32, head_dim=64 -> inner_dim=2048,
# cross_attention_dim=2048; num_layers counted from the checkpoint).
_INNER = 2048
_BLOCKS = 28
# The released ComfyUI single-file (Lightricks/LTX-Video ltx-video-2b-v0.9.safetensors)
# stores every diffusion tensor under a ``model.diffusion_model.`` prefix (verified by
# real-checkpoint key validation); flux/ideogram diffusion_models files are bare.
_PFX = "model.diffusion_model."


def _dims() -> dict[str, int]:
    return {"D": _INNER, "FF": 4 * _INNER}  # 2048, 8192


def _block_modules() -> tuple:
    p = _PFX + "transformer_blocks.{block}"
    return (
        # self-attention (attn1): q/k/v all from D, out projection in nn.Sequential[0]
        linear(f"{p}.attn1.to_q", "D", "D"),
        linear(f"{p}.attn1.to_k", "D", "D"),
        linear(f"{p}.attn1.to_v", "D", "D"),
        linear(f"{p}.attn1.to_out.0", "D", "D"),
        # cross-attention (attn2): k/v from the caption context (cross_attention_dim=D)
        linear(f"{p}.attn2.to_q", "D", "D"),
        linear(f"{p}.attn2.to_k", "D", "D"),
        linear(f"{p}.attn2.to_v", "D", "D"),
        linear(f"{p}.attn2.to_out.0", "D", "D"),
        # feed-forward (GELU_approx project_in -> Linear out)
        linear(f"{p}.ff.net.0.proj", "FF", "D", module_type="GELULinear"),
        linear(f"{p}.ff.net.2", "D", "FF"),
    )


def _extra_components() -> tuple:
    return (
        kept_component(_PFX + "patchify_proj", "Linear", "transformer", "latent patch projection kept high precision"),
        kept_component(_PFX + "adaln_single", "AdaLayerNormSingle", "transformer", "timestep adaln kept high precision"),
        kept_component(_PFX + "caption_projection", "PixArtAlphaTextProjection", "transformer", "text projection kept high precision"),
        kept_component(_PFX + "proj_out", "Linear", "transformer", "output projection kept high precision"),
        kept_component(_PFX + "scale_shift_table", "Parameter", "transformer", "per-block modulation table kept high precision"),
    )


def build_ltxv_static_contract() -> StockDitContract:
    return StockDitContract(
        family="ltxv",
        schema_version=CONTRACT_SCHEMA_VERSION,
        preferred_format="fp8_e4m3",
        dims=_dims(),
        block_groups=(BlockGroup(prefix=_PFX + "transformer_blocks", count=_BLOCKS, modules=_block_modules()),),
        extra_components=_extra_components(),
        metadata={
            "export_name": "LTX-Video",
            "architecture": "ltxv_dit",
            "inner_dim": _INNER,
            "num_heads": 32,
            "head_dim": 64,
            "cross_attention_dim": _INNER,
            "ff_mult": 4,
            "num_blocks": _BLOCKS,
            "caption_channels": 4096,
            "supported_model_ids": ("Lightricks/LTX-Video",),
        },
    )


class LtxvAdapter:
    """Adapter for LTXV-2B (0.9.x) video DiT."""

    family = "ltxv"
    supported_model_ids = ["Lightricks/LTX-Video"]

    def inspect(self, source: ModelSource):
        contract = build_ltxv_static_contract()
        graph = build_stock_dit_graph(
            contract,
            source,
            artifact_metadata=stock_dit_artifact_contract_metadata("ltxv"),
        )
        return summarize_stock_dit_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "fp8_e4m3") -> QuantPolicy:
        return QuantPolicy(
            name="ltxv_default",
            algorithm="fp8_static",
            target_dtype=target_dtype,
            include=[_PFX + "transformer_blocks.*"],
            exclude=[],
            keep_components=["text_encoder", "vae"],
        )


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(LtxvAdapter())
