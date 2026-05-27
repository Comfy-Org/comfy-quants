"""Registry facade for quantization algorithms."""

from __future__ import annotations

# Import built-in algorithms for registration side effects.
from comfy_quants.algorithms import fp8_static as _fp8_static_algorithm  # noqa: F401
from comfy_quants.registry.global_registry import registry


def list_algorithms() -> list[str]:
    return registry.list_algorithms()


def get_algorithm(name: str):
    return registry.get_algorithm(name)
