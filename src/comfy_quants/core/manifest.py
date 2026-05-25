"""Artifact manifest domain object and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from comfy_quants.core.compatibility import CompatibilityLevel, parse_compatibility_level
from comfy_quants.core.errors import ManifestError
from comfy_quants.utils.jsonio import read_json, write_json


@dataclass
class ArtifactManifest:
    """Versioned manifest that makes a quantized artifact auditable."""

    artifact_id: str
    model: dict[str, Any]
    quantization: dict[str, Any]
    calibration: dict[str, Any]
    hardware: dict[str, Any]
    compatibility: dict[str, Any]
    schema_version: str = "0.1.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    files: list[dict[str, Any]] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=lambda: {"status": "pending", "report_path": None})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactManifest":
        cls.validate_dict(data)
        return cls(**data)

    @staticmethod
    def validate_dict(data: dict[str, Any]) -> None:
        required = {
            "schema_version",
            "artifact_id",
            "created_at",
            "model",
            "quantization",
            "calibration",
            "hardware",
            "compatibility",
            "files",
            "hashes",
            "validation",
        }
        missing = required - set(data)
        if missing:
            raise ManifestError(f"manifest missing required fields: {sorted(missing)}")
        if not isinstance(data["model"], dict) or not data["model"].get("family"):
            raise ManifestError("manifest.model.family is required")
        if not isinstance(data["quantization"], dict) or not data["quantization"].get("algorithm"):
            raise ManifestError("manifest.quantization.algorithm is required")
        compat = data.get("compatibility", {})
        if not isinstance(compat, dict) or "level" not in compat:
            raise ManifestError("manifest.compatibility.level is required")
        parse_compatibility_level(compat["level"])

    def save(self, path: str | Path) -> None:
        """Save manifest as JSON."""
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ArtifactManifest":
        """Load and validate a manifest JSON file."""
        return cls.from_dict(read_json(path))


def create_minimal_manifest(
    artifact_id: str,
    family: str,
    model_id: str,
    revision: str | None,
    algorithm: str,
    target_dtype: str,
    compatibility_level: str | CompatibilityLevel = "L0",
    hardware: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Create a minimal valid artifact manifest."""
    level = parse_compatibility_level(compatibility_level)
    return ArtifactManifest(
        artifact_id=artifact_id,
        model={"family": family, "model_id": model_id, "revision": revision, "config_hash": None, "weight_hashes": {}},
        quantization={"algorithm": algorithm, "target_dtype": target_dtype, "algorithm_version": "0.1.0"},
        calibration={"dataset_id": None, "dataset_hash": None, "prompt_count": 0, "edit_case_count": 0},
        hardware=hardware or {},
        compatibility={"level": level.value, "description": level.name, "backend": "torch_ref", "hardware_accelerated": False},
    )
