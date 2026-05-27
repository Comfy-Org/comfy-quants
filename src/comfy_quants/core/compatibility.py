"""Artifact compatibility levels used by Comfy Quants reports and manifests."""

from enum import Enum


class CompatibilityLevel(str, Enum):
    """Explicit compatibility levels for quantized artifacts."""

    L0_SCHEMA_VALID = "L0"
    L1_ROUNDTRIP_VALID = "L1"
    L2_MODEL_GRAPH_LOADABLE = "L2"
    L3_BACKEND_LOADABLE = "L3"
    L4_BACKEND_RUNNABLE = "L4"
    L5_HARDWARE_ACCELERATED = "L5"
    L6_QUALITY_GATED = "L6"


LEVEL_DESCRIPTIONS = {
    CompatibilityLevel.L0_SCHEMA_VALID: "schema-valid",
    CompatibilityLevel.L1_ROUNDTRIP_VALID: "roundtrip-valid",
    CompatibilityLevel.L2_MODEL_GRAPH_LOADABLE: "model-graph-loadable",
    CompatibilityLevel.L3_BACKEND_LOADABLE: "backend-loadable",
    CompatibilityLevel.L4_BACKEND_RUNNABLE: "backend-runnable",
    CompatibilityLevel.L5_HARDWARE_ACCELERATED: "hardware-accelerated",
    CompatibilityLevel.L6_QUALITY_GATED: "quality-gated",
}


def parse_compatibility_level(value: str | CompatibilityLevel) -> CompatibilityLevel:
    """Parse either an enum instance, L0-L6 token, or description string."""
    if isinstance(value, CompatibilityLevel):
        return value
    normalized = str(value).strip().lower()
    for level, desc in LEVEL_DESCRIPTIONS.items():
        if normalized in {level.value.lower(), desc.lower(), level.name.lower()}:
            return level
    raise ValueError(f"unknown compatibility level: {value!r}")
