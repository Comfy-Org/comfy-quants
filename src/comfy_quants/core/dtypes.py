"""Quantization dtype metadata without importing heavy tensor frameworks."""

from dataclasses import dataclass
from enum import Enum


class QuantDType(str, Enum):
    """Dtypes that Comfy Quants can describe in metadata."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"
    UINT4 = "uint4"
    FP4_E2M1 = "fp4_e2m1"
    NVFP4 = "nvfp4"
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"


@dataclass(frozen=True)
class DTypeSpec:
    """Static dtype description for schemas and reports."""

    name: str
    bits: int
    storage_dtype: str
    subbyte: bool = False
    floating: bool = False
    block_scaled: bool = False
    notes: str = ""


KNOWN_DTYPES: dict[str, DTypeSpec] = {
    QuantDType.BF16.value: DTypeSpec("bf16", 16, "uint16", floating=True),
    QuantDType.FP16.value: DTypeSpec("fp16", 16, "uint16", floating=True),
    QuantDType.FP32.value: DTypeSpec("fp32", 32, "uint32", floating=True),
    QuantDType.FP8_E4M3.value: DTypeSpec("fp8_e4m3", 8, "uint8", floating=True),
    QuantDType.FP8_E5M2.value: DTypeSpec("fp8_e5m2", 8, "uint8", floating=True),
    QuantDType.INT8.value: DTypeSpec("int8", 8, "int8"),
    QuantDType.UINT8.value: DTypeSpec("uint8", 8, "uint8"),
    QuantDType.INT4.value: DTypeSpec("int4", 4, "uint8", subbyte=True),
    QuantDType.UINT4.value: DTypeSpec("uint4", 4, "uint8", subbyte=True),
    QuantDType.FP4_E2M1.value: DTypeSpec("fp4_e2m1", 4, "uint8", subbyte=True, floating=True),
    QuantDType.NVFP4.value: DTypeSpec("nvfp4", 4, "uint8", subbyte=True, floating=True, block_scaled=True),
    QuantDType.MXFP8.value: DTypeSpec("mxfp8", 8, "uint8", floating=True, block_scaled=True),
    QuantDType.MXFP4.value: DTypeSpec("mxfp4", 4, "uint8", subbyte=True, floating=True, block_scaled=True),
}


def get_dtype_spec(name: str) -> DTypeSpec:
    """Return dtype metadata for a known quantization dtype."""
    key = str(name).lower()
    if key not in KNOWN_DTYPES:
        raise KeyError(f"unknown dtype: {name}")
    return KNOWN_DTYPES[key]
