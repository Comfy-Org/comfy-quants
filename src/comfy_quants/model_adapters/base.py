"""Model adapter protocol and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from comfy_quants.core.graph import ModelGraph, ModelInspection
from comfy_quants.core.policy import QuantPolicy


@dataclass
class ModelSource:
    """Source descriptor for local or Hugging Face model snapshots."""

    family: str
    model_id: str
    revision: str | None = None
    dtype: str = "bf16"
    source: str = "huggingface"

    @property
    def is_local_path(self) -> bool:
        return Path(self.model_id).exists()


class ModelAdapter(Protocol):
    """Protocol implemented by model-family adapters."""

    family: str
    supported_model_ids: list[str]

    def inspect(self, source: ModelSource) -> tuple[ModelInspection, ModelGraph]:
        """Inspect model structure and return framework-neutral metadata."""

    def default_policy(self, target_dtype: str = "fp8_e4m3") -> QuantPolicy:
        """Return a safe default policy for the model family."""
