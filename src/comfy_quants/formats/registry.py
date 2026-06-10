"""Registry facade for reusable quantization formats."""

from __future__ import annotations

# Import built-in formats for registration side effects.
from comfy_quants.formats import awq_w4a16 as _awq_w4a16_format  # noqa: F401
from comfy_quants.formats import fp8_e4m3 as _fp8_e4m3_format  # noqa: F401
from comfy_quants.formats import fp8_e5m2 as _fp8_e5m2_format  # noqa: F401
from comfy_quants.formats import int8_w8a8 as _int8_w8a8_format  # noqa: F401
from comfy_quants.formats import svdquant_w4a4 as _svdquant_w4a4_format  # noqa: F401
from comfy_quants.registry.global_registry import registry


def list_formats() -> list[str]:
    return registry.list_formats()


def get_format(name: str):
    return registry.get_format(name)
