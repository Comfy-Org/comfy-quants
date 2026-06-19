"""Quantization algorithm protocol."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlgorithmPlanStep:
    """A planned module-level algorithm step."""

    step_id: str
    module_name: str
    action: str
    algorithm: str
    target_dtype: str
