"""FLUX.1 model adapter (Black Forest Labs MMDiT, stock-ComfyUI ``image_model="flux"``).

Authored from ComfyUI ``comfy/ldm/flux/{model,layers}.py``. FLUX is a two-stream MMDiT:
``double_blocks.N`` (img+txt streams, 10 Linears each) followed by ``single_blocks.N``
(fused qkv+mlp, 3 Linears each). FLUX.1-dev: hidden=3072, 24 heads, 19 double + 38
single, mlp_ratio=4 (mlp_hidden=12288). Bare ``double_blocks.``/``single_blocks.`` keys
(no wrapper prefix). Stock-ComfyUI-native target (fp8_e4m3/fp8_e5m2/mxfp8/nvfp4).

Default policy quantizes every block Linear and keeps the embedders / final layer high
precision. FLUX.2 is a different architecture (``image_model="flux2"``) and is out of
scope here.
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

CONTRACT_SCHEMA_VERSION = "flux_static_contract.v1"

# FLUX.1-dev config (comfy/model_detection.py: hidden_size=3072, num_heads=24,
# mlp_ratio=4; depth/depth_single_blocks counted from the checkpoint).
_HIDDEN = 3072
_DOUBLE_BLOCKS = 19
_SINGLE_BLOCKS = 38


def _dims() -> dict[str, int]:
    h = _HIDDEN
    mlp = 4 * h  # 12288
    return {
        "H": h,  # 3072
        "QKV": 3 * h,  # 9216
        "MLP": mlp,  # 12288
        "MOD6": 6 * h,  # 18432  (double-stream modulation: shift/scale/gate x2)
        "MOD3": 3 * h,  # 9216   (single-stream modulation)
        "L1OUT": 3 * h + mlp,  # 21504  (single linear1 = qkv + mlp_in)
        "L2IN": h + mlp,  # 15360   (single linear2 = proj + mlp_out)
    }


def _double_block_modules() -> tuple:
    p = "double_blocks.{block}"
    mods = []
    for stream in ("img", "txt"):
        mods += [
            linear(f"{p}.{stream}_attn.qkv", "QKV", "H"),
            linear(f"{p}.{stream}_attn.proj", "H", "H"),
            linear(f"{p}.{stream}_mlp.0", "MLP", "H", module_type="GELULinear"),
            linear(f"{p}.{stream}_mlp.2", "H", "MLP"),
            linear(f"{p}.{stream}_mod.lin", "MOD6", "H", module_type="ModulationLinear"),
        ]
    return tuple(mods)


def _single_block_modules() -> tuple:
    p = "single_blocks.{block}"
    return (
        linear(f"{p}.linear1", "L1OUT", "H"),  # fused qkv + mlp_in
        linear(f"{p}.linear2", "H", "L2IN"),  # fused proj + mlp_out
        linear(f"{p}.modulation.lin", "MOD3", "H", module_type="ModulationLinear"),
    )


def _extra_components() -> tuple:
    return (
        kept_component("img_in", "Linear", "transformer", "image input projection kept high precision"),
        kept_component("txt_in", "Linear", "transformer", "text input projection kept high precision"),
        kept_component("time_in", "MLPEmbedder", "transformer", "timestep embedding kept high precision"),
        kept_component("vector_in", "MLPEmbedder", "transformer", "pooled-vector embedding kept high precision"),
        kept_component("guidance_in", "MLPEmbedder", "transformer", "guidance embedding kept high precision"),
        kept_component("final_layer", "LastLayer", "transformer", "final layer kept high precision"),
    )


def build_flux_static_contract() -> StockDitContract:
    return StockDitContract(
        family="flux",
        schema_version=CONTRACT_SCHEMA_VERSION,
        preferred_format="fp8_e4m3",
        dims=_dims(),
        block_groups=(
            BlockGroup(prefix="double_blocks", count=_DOUBLE_BLOCKS, modules=_double_block_modules()),
            BlockGroup(prefix="single_blocks", count=_SINGLE_BLOCKS, modules=_single_block_modules()),
        ),
        extra_components=_extra_components(),
        metadata={
            "export_name": "FLUX.1",
            "architecture": "flux_mmdit",
            "hidden_size": _HIDDEN,
            "num_heads": 24,
            "head_dim": 128,
            "mlp_ratio": 4.0,
            "double_blocks": _DOUBLE_BLOCKS,
            "single_blocks": _SINGLE_BLOCKS,
            "context_in_dim": 4096,
            "supported_model_ids": ("black-forest-labs/FLUX.1-dev",),
        },
    )


class FluxAdapter:
    """Adapter for FLUX.1-dev (and architecturally-identical FLUX.1 variants)."""

    family = "flux"
    supported_model_ids = ["black-forest-labs/FLUX.1-dev"]

    def inspect(self, source: ModelSource):
        contract = build_flux_static_contract()
        graph = build_stock_dit_graph(
            contract,
            source,
            artifact_metadata=stock_dit_artifact_contract_metadata("flux"),
        )
        return summarize_stock_dit_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "fp8_e4m3") -> QuantPolicy:
        return QuantPolicy(
            name="flux_default",
            algorithm="fp8_static",
            target_dtype=target_dtype,
            include=["double_blocks.*", "single_blocks.*"],
            exclude=[],
            keep_components=["text_encoder", "vae"],
        )


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(FluxAdapter())
