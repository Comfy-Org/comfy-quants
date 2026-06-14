"""Comfy Quants: ComfyUI-aligned offline quantization sub-library.

The package mirrors the Comfy Kitchen pattern: import built-in modules for
registration side effects, expose a central registry, and keep UI/custom-node
integration outside the core library.
"""

from __future__ import annotations

__version__ = "0.1.0"

from comfy_quants.registry.global_registry import registry  # noqa: E402

# Import built-ins to trigger auto-registration, Comfy Kitchen style.
from comfy_quants.backends import int4_kitchen_export as _int4_kitchen_export_backend  # noqa: E402,F401
from comfy_quants.backends import int4_full_pipeline_export as _int4_full_pipeline_export_backend  # noqa: E402,F401
from comfy_quants.backends import deepcompressor_import as _deepcompressor_int4_import_backend  # noqa: E402,F401
from comfy_quants.backends import torch_ref as _torch_ref_backend  # noqa: E402,F401
from comfy_quants.algorithms import fp8_static as _fp8_static_algorithm  # noqa: E402,F401
from comfy_quants.algorithms import int8_w8a8 as _int8_w8a8_algorithm  # noqa: E402,F401
from comfy_quants.algorithms import mxfp8 as _mxfp8_algorithm  # noqa: E402,F401
from comfy_quants.formats import awq_w4a16 as _awq_w4a16_format  # noqa: E402,F401
from comfy_quants.formats import fp8_e4m3 as _fp8_e4m3_format  # noqa: E402,F401
from comfy_quants.formats import fp8_e5m2 as _fp8_e5m2_format  # noqa: E402,F401
from comfy_quants.formats import int8_w8a8 as _int8_w8a8_format  # noqa: E402,F401
from comfy_quants.formats import mxfp8 as _mxfp8_format  # noqa: E402,F401
from comfy_quants.formats import svdquant_w4a4 as _svdquant_w4a4_format  # noqa: E402,F401
from comfy_quants.model_adapters import qwen_image as _qwen_image_adapter  # noqa: E402,F401
from comfy_quants.model_adapters import qwen_image_edit as _qwen_image_edit_adapter  # noqa: E402,F401
from comfy_quants.model_adapters import qwen_image_layered as _qwen_image_layered_adapter  # noqa: E402,F401


def list_model_adapters() -> list[str]:
    return registry.list_adapters()


def list_algorithms() -> list[str]:
    return registry.list_algorithms()


def list_quant_formats() -> list[str]:
    return registry.list_formats()


def list_backends() -> list[str]:
    return registry.list_backends()


__all__ = [
    "__version__",
    "registry",
    "list_model_adapters",
    "list_algorithms",
    "list_quant_formats",
    "list_backends",
]
