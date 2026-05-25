"""Shared metadata for Comfy Quants FP8 checkpoint formats."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FP8RuntimeSpec:
    """Runtime contract needed by FP8 tensor writers."""

    name: str
    torch_dtype_name: str
    checkpoint_format: str
    max_finite: float
    safetensors_dtype: str
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int


_FP8_SPECS: dict[str, FP8RuntimeSpec] = {
    "fp8_e4m3": FP8RuntimeSpec(
        name="fp8_e4m3",
        torch_dtype_name="float8_e4m3fn",
        checkpoint_format="float8_e4m3fn",
        max_finite=448.0,
        safetensors_dtype="F8_E4M3",
        exponent_bits=4,
        mantissa_bits=3,
        exponent_bias=7,
    ),
    "fp8_e5m2": FP8RuntimeSpec(
        name="fp8_e5m2",
        torch_dtype_name="float8_e5m2",
        checkpoint_format="float8_e5m2",
        max_finite=57344.0,
        safetensors_dtype="F8_E5M2",
        exponent_bits=5,
        mantissa_bits=2,
        exponent_bias=15,
    ),
}

FP8_FORMAT_NAMES = tuple(_FP8_SPECS)


def get_fp8_runtime_spec(name: str) -> FP8RuntimeSpec:
    """Return the FP8 runtime spec for a supported format name."""
    key = str(name).strip().lower()
    try:
        return _FP8_SPECS[key]
    except KeyError as exc:
        supported = ", ".join(FP8_FORMAT_NAMES)
        raise KeyError(f"unsupported FP8 format: {name}; supported: {supported}") from exc


def is_fp8_format_name(name: str | None) -> bool:
    """Return whether *name* is a supported Comfy Quants FP8 format."""
    return str(name or "").strip().lower() in _FP8_SPECS


def fp8_checkpoint_quant_config(name: str) -> dict[str, bool | str]:
    """Return the checkpoint-side quantization config tensor payload."""
    spec = get_fp8_runtime_spec(name)
    return {
        "format": spec.checkpoint_format,
        "full_precision_matrix_mult": True,
    }


def fp8_inference_checkpoint_kind(name: str) -> str:
    """Return a stable report kind for an exported FP8 checkpoint."""
    spec = get_fp8_runtime_spec(name)
    return f"{spec.name}_inference_checkpoint"


def fp8_inference_artifact_contract(name: str) -> str:
    """Return the artifact contract identifier for a Qwen FP8 checkpoint."""
    spec = get_fp8_runtime_spec(name)
    return f"qwen_image_{spec.name}_inference_checkpoint.v1"
