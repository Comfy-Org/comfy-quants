"""AWQ W4A16 algorithm helpers."""

from __future__ import annotations

from comfy_quants.algorithms.awq_w4a16.config import AWQ_W4A16_ALGORITHM_STATE, AWQ_W4A16_RUNTIME_UNVERIFIED_STATE
from comfy_quants.algorithms.awq_w4a16.qwen_modulation import reorder_qwen_modulation_awq_tensors
from comfy_quants.algorithms.awq_w4a16.reference import AWQ_W4A16_REFERENCE_STATE, reference_awq_w4a16_linear
from comfy_quants.algorithms.awq_w4a16.runtime_fixture import (
    AWQ_W4A16_RUNTIME_FIXTURE_ARTIFACT_STATE,
    AWQ_W4A16_RUNTIME_FIXTURE_SCHEMA_VERSION,
    DEFAULT_AWQ_RUNTIME_FIXTURE_FILENAME,
    DEFAULT_AWQ_RUNTIME_FIXTURE_REPORT_FILENAME,
    AwqW4A16RuntimeFixture,
    AwqW4A16RuntimeFixtureConfig,
    WrittenAwqW4A16RuntimeFixture,
    build_awq_w4a16_runtime_fixture,
    write_awq_w4a16_runtime_fixture,
)
from comfy_quants.algorithms.awq_w4a16.weight_quant import (
    AwqW4A16LinearTensors,
    dequantize_awq_w4a16_weight,
    quantize_linear_weight_to_awq_w4a16,
)

__all__ = [
    "AWQ_W4A16_ALGORITHM_STATE",
    "AWQ_W4A16_REFERENCE_STATE",
    "AWQ_W4A16_RUNTIME_FIXTURE_ARTIFACT_STATE",
    "AWQ_W4A16_RUNTIME_FIXTURE_SCHEMA_VERSION",
    "AWQ_W4A16_RUNTIME_UNVERIFIED_STATE",
    "AwqW4A16LinearTensors",
    "AwqW4A16RuntimeFixture",
    "AwqW4A16RuntimeFixtureConfig",
    "DEFAULT_AWQ_RUNTIME_FIXTURE_FILENAME",
    "DEFAULT_AWQ_RUNTIME_FIXTURE_REPORT_FILENAME",
    "WrittenAwqW4A16RuntimeFixture",
    "build_awq_w4a16_runtime_fixture",
    "dequantize_awq_w4a16_weight",
    "quantize_linear_weight_to_awq_w4a16",
    "reference_awq_w4a16_linear",
    "reorder_qwen_modulation_awq_tensors",
    "write_awq_w4a16_runtime_fixture",
]
