"""Registry facade for model adapters."""

from __future__ import annotations

# Import built-in adapters for registration side effects.
from comfy_quants.model_adapters import qwen_image as _qwen_image_adapter  # noqa: F401
from comfy_quants.model_adapters import qwen_image_edit as _qwen_image_edit_adapter  # noqa: F401
from comfy_quants.model_adapters import qwen_image_layered as _qwen_image_layered_adapter  # noqa: F401
from comfy_quants.model_adapters import anima as _anima_adapter  # noqa: F401
from comfy_quants.model_adapters import flux as _flux_adapter  # noqa: F401
from comfy_quants.model_adapters import flux2 as _flux2_adapter  # noqa: F401
from comfy_quants.model_adapters import ltxv as _ltxv_adapter  # noqa: F401
from comfy_quants.model_adapters import ideogram4 as _ideogram4_adapter  # noqa: F401
from comfy_quants.registry.global_registry import registry


def list_adapters() -> list[str]:
    return registry.list_adapters()


def get_adapter(family: str):
    return registry.get_adapter(family)
