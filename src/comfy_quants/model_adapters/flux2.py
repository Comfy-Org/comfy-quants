"""FLUX.2 model adapter (Black Forest Labs FLUX.2 MMDiT, stock-ComfyUI ``image_model="flux2"``).

Authored from ComfyUI ``comfy/model_detection.py`` (flux2 branch) and validated against the
real ``Comfy-Org/flux2-dev`` checkpoint header. FLUX.2 differs from FLUX.1 in several ways:

- hidden=6144 (48 heads x 128), mlp_ratio=3, **gated MLP** (``mlp_silu_act``), qkv_bias=False;
- **global modulation**: per-block ``img_mod``/``txt_mod`` Linears are gone; modulation lives in
  three top-level Linears (``double_stream_modulation_img``/``_txt``, ``single_stream_modulation``)
  and is kept high precision here;
- FLUX.2-dev: 8 ``double_blocks`` + 48 ``single_blocks``; out_channels=128 (patch_size 1).

Bare ``double_blocks.``/``single_blocks.`` keys. Stock-ComfyUI-native target
(fp8_e4m3/fp8_e5m2/mxfp8/nvfp4). The shipped ``flux2_dev_fp8mixed`` checkpoint quantizes only
the MLP/feedforward Linears (attention kept bf16); this adapter's default policy quantizes all
block Linears, and callers can ``exclude`` the attention to reproduce the mixed recipe.
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

CONTRACT_SCHEMA_VERSION = "flux2_static_contract.v1"

# Mixed-precision recipe (matches official flux2_dev_fp8mixed): keep double-stream
# attention projections in bf16; module names carry no ``.weight`` suffix.
MIXED_KEEP_ATTENTION = ("*_attn.qkv", "*_attn.proj")

# FLUX.2-dev config (comfy/model_detection.py flux2 branch + real checkpoint header).
_HIDDEN = 6144  # 48 heads x 128 head_dim
_DOUBLE_BLOCKS = 8
_SINGLE_BLOCKS = 48
_MLP_RATIO = 3


def _dims() -> dict[str, int]:
    h = _HIDDEN
    mlp = _MLP_RATIO * h  # 18432  (mlp_hidden)
    return {
        "H": h,  # 6144
        "QKV": 3 * h,  # 18432
        "MLP": mlp,  # 18432  (gated mlp hidden; linear2/mlp.2 in-features)
        "MLP2": 2 * mlp,  # 36864  (mlp.0 out = 2x hidden for SiLU gating)
        "L1OUT": 3 * h + 2 * mlp,  # 55296  (single linear1 = qkv + gated mlp_in)
        "L2IN": h + mlp,  # 24576   (single linear2 = proj + mlp_out)
    }


def _double_block_modules() -> tuple:
    p = "double_blocks.{block}"
    mods = []
    for stream in ("img", "txt"):
        mods += [
            linear(f"{p}.{stream}_attn.qkv", "QKV", "H"),
            linear(f"{p}.{stream}_attn.proj", "H", "H"),
            linear(f"{p}.{stream}_mlp.0", "MLP2", "H", module_type="SiLUGatedLinear"),
            linear(f"{p}.{stream}_mlp.2", "H", "MLP"),
        ]
    return tuple(mods)


def _single_block_modules() -> tuple:
    p = "single_blocks.{block}"
    return (
        linear(f"{p}.linear1", "L1OUT", "H"),  # fused qkv + gated mlp_in
        linear(f"{p}.linear2", "H", "L2IN"),  # fused proj + mlp_out
    )


def _extra_components() -> tuple:
    return (
        kept_component("img_in", "Linear", "transformer", "image input projection kept high precision"),
        kept_component("txt_in", "Linear", "transformer", "text input projection kept high precision"),
        kept_component("time_in", "MLPEmbedder", "transformer", "timestep embedding kept high precision"),
        kept_component("guidance_in", "MLPEmbedder", "transformer", "guidance embedding kept high precision"),
        kept_component("double_stream_modulation_img", "Modulation", "transformer", "global img modulation kept high precision"),
        kept_component("double_stream_modulation_txt", "Modulation", "transformer", "global txt modulation kept high precision"),
        kept_component("single_stream_modulation", "Modulation", "transformer", "global single-stream modulation kept high precision"),
        kept_component("final_layer", "LastLayer", "transformer", "final layer kept high precision"),
    )


def build_flux2_static_contract() -> StockDitContract:
    return StockDitContract(
        family="flux2",
        schema_version=CONTRACT_SCHEMA_VERSION,
        preferred_format="fp8_e4m3",
        dims=_dims(),
        block_groups=(
            BlockGroup(prefix="double_blocks", count=_DOUBLE_BLOCKS, modules=_double_block_modules()),
            BlockGroup(prefix="single_blocks", count=_SINGLE_BLOCKS, modules=_single_block_modules()),
        ),
        extra_components=_extra_components(),
        metadata={
            "export_name": "FLUX.2",
            "architecture": "flux2_mmdit",
            "hidden_size": _HIDDEN,
            "num_heads": 48,
            "head_dim": 128,
            "mlp_ratio": float(_MLP_RATIO),
            "mlp_silu_act": True,
            "global_modulation": True,
            "qkv_bias": False,
            "double_blocks": _DOUBLE_BLOCKS,
            "single_blocks": _SINGLE_BLOCKS,
            "out_channels": 128,
            "supported_model_ids": ("black-forest-labs/FLUX.2-dev",),
        },
    )


class Flux2Adapter:
    """Adapter for FLUX.2-dev MMDiT (global modulation, gated MLP)."""

    family = "flux2"
    supported_model_ids = ["black-forest-labs/FLUX.2-dev"]

    def inspect(self, source: ModelSource):
        contract = build_flux2_static_contract()
        graph = build_stock_dit_graph(
            contract,
            source,
            artifact_metadata=stock_dit_artifact_contract_metadata("flux2"),
        )
        return summarize_stock_dit_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "fp8_e4m3", *, mixed: bool = False) -> QuantPolicy:
        """Return the layer-selection policy.

        ``mixed=True`` reproduces the official ``flux2_dev_fp8mixed`` recipe: keep the
        double-stream **attention** projections (``*_attn.qkv`` / ``*_attn.proj``) in bf16
        and quantize only the MLP/feedforward — plus the single-stream fused
        ``linear1``/``linear2`` (which mix qkv+mlp and cannot be split). 32B attention is
        quant-sensitive, so this trades ~6% extra weight size for noticeably better images.
        Full quantization (``mixed=False``) stays the default for max compression/speed.
        """
        return QuantPolicy(
            name="flux2_mixed" if mixed else "flux2_default",
            algorithm="fp8_static",
            target_dtype=target_dtype,
            include=["double_blocks.*", "single_blocks.*"],
            exclude=list(MIXED_KEEP_ATTENTION) if mixed else [],
            keep_components=["text_encoder", "vae"],
        )


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(Flux2Adapter())
