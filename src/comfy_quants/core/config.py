"""Configuration objects for quantization jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from comfy_quants.core.errors import ConfigurationError
from comfy_quants.utils.jsonio import read_yaml


@dataclass
class ProjectSection:
    name: str
    created_by: str = "comfy_quants"
    seed: int = 42


@dataclass
class ModelSection:
    family: str
    model_id: str
    revision: str | None = None
    dtype: str = "bf16"
    source: str = "huggingface"
    trust_remote_code: bool = False
    components: dict[str, str] = field(default_factory=dict)


@dataclass
class HardwareSection:
    device: str = "cuda:0"
    gpu_profile: str = "rtx_pro_6000_blackwell_96gb"
    max_vram_gb: int = 88
    cpu_offload: bool = True
    nvme_offload: bool = False
    scratch_dir: str | None = None


@dataclass
class CalibrationSection:
    source: str | None = None
    batch_size: int = 1
    capture_dtype: str = "bf16"
    activation_cache: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaleSection:
    granularity: str = "per_channel"
    axis: str | int = "out_features"
    method: str = "amax"
    percentile: float | None = None


@dataclass
class QuantSection:
    algorithm: str = "fp8_static"
    target_dtype: str = "fp8_e4m3"
    scale: ScaleSection = field(default_factory=ScaleSection)
    rounding: str = "nearest_even"
    modules: dict[str, list[str]] = field(default_factory=lambda: {"include": [], "exclude": []})
    fallback: dict[str, Any] = field(default_factory=lambda: {"on_error": "keep_bf16"})


@dataclass
class ArtifactSection:
    format: str = "safetensors_quant"
    compatibility_target: str = "L2"
    save_reference_dequant: bool = False
    write_module_reports: bool = True


@dataclass
class ValidationSection:
    smoke_set: str | None = None
    smoke_edit_set: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantConfig:
    """Top-level quantization config parsed from YAML/JSON."""

    project: ProjectSection
    model: ModelSection
    hardware: HardwareSection = field(default_factory=HardwareSection)
    calibration: CalibrationSection = field(default_factory=CalibrationSection)
    quant: QuantSection = field(default_factory=QuantSection)
    artifact: ArtifactSection = field(default_factory=ArtifactSection)
    validation: ValidationSection = field(default_factory=ValidationSection)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("raw", None)
        return data


def _unknown_keys(section: str, data: dict[str, Any], allowed: set[str]) -> None:
    extra = set(data) - allowed
    if extra:
        raise ConfigurationError(f"unknown keys in {section}: {sorted(extra)}")


def load_quant_config(path: str | Path) -> QuantConfig:
    """Load and validate a Comfy Quants quantization config."""
    raw = read_yaml(path)
    if not isinstance(raw, dict):
        raise ConfigurationError("top-level config must be a mapping")
    if "project" not in raw or "model" not in raw:
        raise ConfigurationError("config requires 'project' and 'model' sections")

    project_raw = dict(raw.get("project") or {})
    model_raw = dict(raw.get("model") or {})
    hardware_raw = dict(raw.get("hardware") or {})
    calibration_raw = dict(raw.get("calibration") or {})
    quant_raw = dict(raw.get("quant") or {})
    artifact_raw = dict(raw.get("artifact") or {})
    validation_raw = dict(raw.get("validation") or {})

    _unknown_keys("project", project_raw, {"name", "created_by", "seed"})
    _unknown_keys("model", model_raw, {"family", "model_id", "revision", "dtype", "source", "trust_remote_code", "components"})
    _unknown_keys("hardware", hardware_raw, {"device", "gpu_profile", "max_vram_gb", "cpu_offload", "nvme_offload", "scratch_dir"})
    _unknown_keys("quant", quant_raw, {"algorithm", "target_dtype", "scale", "rounding", "modules", "fallback"})
    _unknown_keys("artifact", artifact_raw, {"format", "compatibility_target", "save_reference_dequant", "write_module_reports"})

    scale_raw = dict(quant_raw.pop("scale", {}) or {})
    _unknown_keys("quant.scale", scale_raw, {"granularity", "axis", "method", "percentile"})

    calib_known = {"source", "batch_size", "capture_dtype", "activation_cache"}
    calib_extra = {k: v for k, v in calibration_raw.items() if k not in calib_known}
    calibration_args = {k: v for k, v in calibration_raw.items() if k in calib_known}

    validation_known = {"smoke_set", "smoke_edit_set"}
    validation_extra = {k: v for k, v in validation_raw.items() if k not in validation_known}
    validation_args = {k: v for k, v in validation_raw.items() if k in validation_known}

    try:
        cfg = QuantConfig(
            project=ProjectSection(**project_raw),
            model=ModelSection(**model_raw),
            hardware=HardwareSection(**hardware_raw),
            calibration=CalibrationSection(**calibration_args, extra=calib_extra),
            quant=QuantSection(scale=ScaleSection(**scale_raw), **quant_raw),
            artifact=ArtifactSection(**artifact_raw),
            validation=ValidationSection(**validation_args, extra=validation_extra),
            raw=raw,
        )
    except TypeError as exc:
        raise ConfigurationError(str(exc)) from exc
    return cfg
