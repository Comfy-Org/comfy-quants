"""Public Python SDK."""

from __future__ import annotations

from pathlib import Path

from comfy_quants.core.config import QuantConfig, load_quant_config
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter, list_adapters


def inspect_model(model_id: str, family: str, revision: str | None = None, dtype: str = "bf16"):
    """Inspect a model using the registered adapter and return inspection plus graph."""
    adapter = get_adapter(family)
    return adapter.inspect(ModelSource(family=family, model_id=model_id, revision=revision, dtype=dtype))


def load_config(path: str | Path) -> QuantConfig:
    """Load a quantization config file."""
    return load_quant_config(path)


__all__ = ["inspect_model", "load_config", "list_adapters", "QuantConfig"]
