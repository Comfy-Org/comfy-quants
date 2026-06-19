"""Anima model adapter (cosmos_predict2 DiT + llm_adapter).

One architecture at two cosmos sizes: ``anima`` (2B, model_channels=2048) and
``anima_14b`` (14B, model_channels=5120). Stock-ComfyUI-native target (fp8/mxfp8/nvfp4).
Layer selection follows the convert_to_quant ``anima`` preset: keep block 0 entirely,
keep block 1's adaln modulation, keep the embedders / final layer / llm_adapter high
precision; quantize the main-DiT blocks 2+ (and block 1's attn/mlp).
"""

from __future__ import annotations

from comfy_quants.comfy.anima_contract import anima_artifact_contract_metadata
from comfy_quants.core.policy import QuantPolicy
from comfy_quants.model_adapters.anima_contracts.anima import get_anima_static_contract
from comfy_quants.model_adapters.anima_graph_builder import build_anima_graph_from_contract, summarize_anima_graph
from comfy_quants.model_adapters.base import ModelSource


def _anima_policy(*, name: str, target_dtype: str) -> QuantPolicy:
    return QuantPolicy(
        name=name,
        algorithm="fp8_static",  # overridden per-config by the export command
        target_dtype=target_dtype,
        include=["net.blocks.*"],  # main-DiT blocks only (net.llm_adapter.blocks.* / embedders not matched)
        exclude=["net.blocks.0.*", "net.blocks.1.adaln_modulation*"],
        keep_components=["llm_adapter", "text_encoder", "vae"],
    )


class AnimaAdapter:
    """Adapter for the Anima 2B model (model_channels=2048)."""

    family = "anima"
    model_channels = 2048
    supported_model_ids = ["Comfy-Org/anima"]

    def inspect(self, source: ModelSource):
        contract = get_anima_static_contract(self.model_channels)
        graph = build_anima_graph_from_contract(
            contract,
            source,
            artifact_metadata=anima_artifact_contract_metadata(self.model_channels),
        )
        return summarize_anima_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "fp8_e4m3") -> QuantPolicy:
        return _anima_policy(name="anima_default", target_dtype=target_dtype)


class Anima14BAdapter:
    """Adapter for the Anima 14B model (model_channels=5120)."""

    family = "anima_14b"
    model_channels = 5120
    supported_model_ids = ["Comfy-Org/anima-14b"]

    def inspect(self, source: ModelSource):
        contract = get_anima_static_contract(self.model_channels)
        graph = build_anima_graph_from_contract(
            contract,
            source,
            artifact_metadata=anima_artifact_contract_metadata(self.model_channels),
        )
        return summarize_anima_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "fp8_e4m3") -> QuantPolicy:
        return _anima_policy(name="anima_14b_default", target_dtype=target_dtype)


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(AnimaAdapter())
registry.register_adapter(Anima14BAdapter())
