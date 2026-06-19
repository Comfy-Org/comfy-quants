"""FP8 E5M2 reusable format declaration."""

from __future__ import annotations

from comfy_quants.formats.base import QuantFormatSpec
from comfy_quants.formats.fp8_common import fp8_checkpoint_quant_config, get_fp8_runtime_spec
from comfy_quants.registry.global_registry import registry


_SPEC = get_fp8_runtime_spec("fp8_e5m2")
FP8_E5M2_EXPONENT_BITS = _SPEC.exponent_bits
FP8_E5M2_MANTISSA_BITS = _SPEC.mantissa_bits
FP8_E5M2_EXPONENT_BIAS = _SPEC.exponent_bias
FP8_E5M2_MAX_FINITE = _SPEC.max_finite
FP8_E5M2_TORCH_DTYPE = _SPEC.torch_dtype_name


def fp8_e5m2_checkpoint_quant_config() -> dict[str, bool | str]:
    return fp8_checkpoint_quant_config("fp8_e5m2")


FP8_E5M2_FORMAT = QuantFormatSpec(
    name="fp8_e5m2",
    storage_dtype="uint8",
    bits=8,
    category="floating_point",
    scale_required=True,
    default_scale_granularity="per_tensor",
    compatible_families=("qwen_image", "qwen_image_edit", "anima", "anima_14b"),
    notes=(
        "FP8 E5M2 checkpoint format for Qwen image exports.",
        "Reusable across model adapters; not owned by any single Qwen adapter.",
    ),
    metadata={
        "exponent_bits": FP8_E5M2_EXPONENT_BITS,
        "mantissa_bits": FP8_E5M2_MANTISSA_BITS,
        "exponent_bias": FP8_E5M2_EXPONENT_BIAS,
        "max_finite": FP8_E5M2_MAX_FINITE,
        "torch_dtype": FP8_E5M2_TORCH_DTYPE,
        "safetensors_dtype": _SPEC.safetensors_dtype,
        "default_axis": None,
        "default_scale_method": "amax",
    },
)


registry.register_format(FP8_E5M2_FORMAT)
