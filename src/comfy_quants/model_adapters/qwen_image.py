"""Qwen-Image model adapter."""

from __future__ import annotations

from comfy_quants.comfy.qwen_image_contract import qwen_image_artifact_contract_metadata
from comfy_quants.core.policy import QuantPolicy
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.qwen_contracts.qwen_image import get_qwen_image_static_contract
from comfy_quants.model_adapters.qwen_graph_builder import build_qwen_graph_from_contract, summarize_qwen_graph


class QwenImageAdapter:
    """Adapter for Qwen-Image text-to-image models."""

    family = "qwen_image"
    supported_model_ids = ["Qwen/Qwen-Image", "Qwen/Qwen-Image-2512"]

    def inspect(self, source: ModelSource):
        contract = get_qwen_image_static_contract()
        graph = build_qwen_graph_from_contract(
            contract,
            source,
            artifact_metadata=qwen_image_artifact_contract_metadata(edit=False),
        )
        return summarize_qwen_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "fp8_e4m3") -> QuantPolicy:
        return QuantPolicy(
            name="qwen_image_default_fp8_static",
            algorithm="fp8_static",
            target_dtype=target_dtype,
            include=[
                "transformer_blocks.*.attn.to_*",
                "transformer_blocks.*.attn.add_*_proj",
                "transformer_blocks.*.img_mlp.*",
                "transformer_blocks.*.txt_mlp.*",
                "transformer_blocks.*.img_mod.1",
                "transformer_blocks.*.txt_mod.1",
            ],
            exclude=["transformer_blocks.0.img_mod.1", "*.norm*", "*txt_norm*", "*norm_out*", "*proj_out*", "*vae*", "*text_encoder*"],
            keep_components=["vae", "text_encoder"],
        )


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(QwenImageAdapter())
