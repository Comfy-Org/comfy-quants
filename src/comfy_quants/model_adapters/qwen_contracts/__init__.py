"""Static Qwen model contracts."""

from comfy_quants.model_adapters.qwen_contracts.qwen_image import get_qwen_image_static_contract
from comfy_quants.model_adapters.qwen_contracts.qwen_image_edit import get_qwen_image_edit_static_contract
from comfy_quants.model_adapters.qwen_contracts.types import ModuleContract, QwenModelContract, TensorContract, TransformerContract

__all__ = [
    "ModuleContract",
    "QwenModelContract",
    "TensorContract",
    "TransformerContract",
    "get_qwen_image_static_contract",
    "get_qwen_image_edit_static_contract",
]
