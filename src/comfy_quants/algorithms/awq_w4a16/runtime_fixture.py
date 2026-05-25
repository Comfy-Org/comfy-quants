"""Deterministic AWQ W4A16 layer fixtures for runtime parity work.

This module writes small safetensors fixtures that pair one kitchen-native AWQ
W4A16 layer with oracle inputs and outputs.  The fixtures are intended for
external fused-runtime comparison.  They are not proof that an external runtime
has accepted the format, and they do not import or depend on any model runtime
package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comfy_quants.algorithms.awq_w4a16.reference import AWQ_W4A16_REFERENCE_STATE, reference_awq_w4a16_linear
from comfy_quants.algorithms.awq_w4a16.weight_quant import (
    dequantize_awq_w4a16_weight,
    quantize_linear_weight_to_awq_w4a16_debug,
)
from comfy_quants.formats.awq_w4a16 import (
    AWQ_W4A16_FORMAT_NAME,
    AWQ_W4A16_GROUP_SIZE,
    awq_w4a16_checkpoint_quant_config,
)
from comfy_quants.formats.int4_common import encode_quant_config_tensor
from comfy_quants.utils.hashing import hash_file
from comfy_quants.utils.jsonio import write_json

AWQ_W4A16_RUNTIME_FIXTURE_SCHEMA_VERSION = "awq_w4a16_runtime_fixture.v1"
AWQ_W4A16_RUNTIME_FIXTURE_ARTIFACT_STATE = "local_awq_runtime_fixture_external_runtime_unverified"
AWQ_W4A16_KITCHEN_NATIVE_LAYOUT_NAME = "kitchen_native_awq_w4a16"
DEFAULT_AWQ_RUNTIME_FIXTURE_FILENAME = "awq_w4a16_runtime_fixture.safetensors"
DEFAULT_AWQ_RUNTIME_FIXTURE_REPORT_FILENAME = "runtime_fixture_report.json"
DEFAULT_AWQ_RUNTIME_FIXTURE_LAYER_PREFIX = "fixture_layer"
DEFAULT_AWQ_RUNTIME_FIXTURE_SEED = 2234
DEFAULT_AWQ_RUNTIME_FIXTURE_N = 12
DEFAULT_AWQ_RUNTIME_FIXTURE_K = AWQ_W4A16_GROUP_SIZE * 2
DEFAULT_AWQ_RUNTIME_FIXTURE_BATCH = 3
DEFAULT_AWQ_RUNTIME_FIXTURE_SCALE_DTYPE = "float32"


@dataclass(frozen=True)
class AwqW4A16RuntimeFixtureConfig:
    """Configuration for one deterministic AWQ W4A16 fixture."""

    seed: int = DEFAULT_AWQ_RUNTIME_FIXTURE_SEED
    n: int = DEFAULT_AWQ_RUNTIME_FIXTURE_N
    k: int = DEFAULT_AWQ_RUNTIME_FIXTURE_K
    batch: int = DEFAULT_AWQ_RUNTIME_FIXTURE_BATCH
    group_size: int = AWQ_W4A16_GROUP_SIZE
    include_bias: bool = True
    scale_dtype: str = DEFAULT_AWQ_RUNTIME_FIXTURE_SCALE_DTYPE
    layer_prefix: str = DEFAULT_AWQ_RUNTIME_FIXTURE_LAYER_PREFIX


@dataclass(frozen=True)
class AwqW4A16RuntimeFixture:
    """In-memory fixture tensors and the JSON-serializable report."""

    tensors: dict[str, Any]
    metadata: dict[str, str]
    report: dict[str, Any]


@dataclass(frozen=True)
class WrittenAwqW4A16RuntimeFixture:
    """Paths and report for a written AWQ runtime fixture."""

    fixture_path: Path
    report_path: Path
    report: dict[str, Any]


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImportError("torch is required to build AWQ W4A16 runtime fixtures") from exc
    return torch


def _require_safetensors_save_file():
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - dependency should be installed by package metadata
        raise ImportError("safetensors is required to write AWQ W4A16 runtime fixtures") from exc
    return save_file


def _validate_config(config: AwqW4A16RuntimeFixtureConfig) -> None:
    if int(config.group_size) != AWQ_W4A16_GROUP_SIZE:
        raise ValueError(f"AWQ runtime fixture requires group size {AWQ_W4A16_GROUP_SIZE}, got {config.group_size}")
    if int(config.n) <= 0:
        raise ValueError(f"n must be positive, got {config.n}")
    if int(config.k) <= 0 or int(config.k) % int(config.group_size) != 0:
        raise ValueError(f"k must be a positive multiple of group size {config.group_size}, got {config.k}")
    if int(config.k) % 2 != 0:
        raise ValueError(f"k must be even for INT4 pair packing, got {config.k}")
    if int(config.batch) <= 0:
        raise ValueError(f"batch must be positive, got {config.batch}")
    if not str(config.layer_prefix):
        raise ValueError("layer_prefix must be non-empty")
    if config.scale_dtype not in {"source", "float16", "bfloat16", "float32"}:
        raise ValueError(f"unsupported scale dtype: {config.scale_dtype}")


def _random_fixture_tensors(config: AwqW4A16RuntimeFixtureConfig) -> dict[str, Any]:
    torch = _require_torch()
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    n = int(config.n)
    k = int(config.k)
    weight = (torch.randn((n, k), generator=generator, dtype=torch.float32) * 0.18).contiguous()
    inputs = (torch.randn((int(config.batch), k), generator=generator, dtype=torch.float32) * 1.1).contiguous()
    tensors: dict[str, Any] = {"dense_weight": weight, "inputs": inputs}
    if config.include_bias:
        tensors["bias"] = (torch.randn((n,), generator=generator, dtype=torch.float32) * 0.02).contiguous()
    return tensors


def _owned_tensor(tensor: Any):
    return tensor.detach().clone().contiguous()


def _max_abs_error(actual: Any, expected: Any) -> float:
    return float((actual.detach().float() - expected.detach().float()).abs().max().item()) if int(actual.numel()) else 0.0


def _mean_abs_error(actual: Any, expected: Any) -> float:
    diff = (actual.detach().float() - expected.detach().float()).abs()
    return float(diff.mean().item()) if int(diff.numel()) else 0.0


def _string_metadata(config: AwqW4A16RuntimeFixtureConfig) -> dict[str, str]:
    return {
        "artifact_contract": AWQ_W4A16_RUNTIME_FIXTURE_SCHEMA_VERSION,
        "artifact_state": AWQ_W4A16_RUNTIME_FIXTURE_ARTIFACT_STATE,
        "format": AWQ_W4A16_FORMAT_NAME,
        "storage_layout": AWQ_W4A16_KITCHEN_NATIVE_LAYOUT_NAME,
        "runtime_reference_state": AWQ_W4A16_REFERENCE_STATE,
        "publishable_svdquant_gptq": "false",
        "group_size": str(int(config.group_size)),
    }


def build_awq_w4a16_runtime_fixture(
    config: AwqW4A16RuntimeFixtureConfig | None = None,
) -> AwqW4A16RuntimeFixture:
    """Build a deterministic AWQ W4A16 layer fixture in memory.

    The returned tensors include a single kitchen-native AWQ W4A16 layer under
    ``<layer_prefix>.*`` and oracle tensors under ``fixture.*``.  The expected
    output is produced by this repository's PyTorch reference, so the report
    deliberately keeps ``external_runtime_validation`` at ``not_run``.
    """

    cfg = config or AwqW4A16RuntimeFixtureConfig()
    _validate_config(cfg)
    base = _random_fixture_tensors(cfg)
    debug = quantize_linear_weight_to_awq_w4a16_debug(
        base["dense_weight"],
        group_size=int(cfg.group_size),
        scale_dtype=cfg.scale_dtype,
    )
    quant_config = awq_w4a16_checkpoint_quant_config(group_size=int(cfg.group_size))
    natural_params: dict[str, Any] = {
        "weight": debug.packed_weight,
        "weight_scale": debug.weight_scale,
        "weight_zero": debug.weight_zero,
        "comfy_quant": encode_quant_config_tensor(quant_config),
    }
    if "bias" in base:
        natural_params["bias"] = base["bias"]

    inputs = base["inputs"]
    expected = reference_awq_w4a16_linear(
        inputs,
        natural_params["weight"],
        natural_params["weight_scale"],
        natural_params["weight_zero"],
        bias=natural_params.get("bias"),
        group_size=int(cfg.group_size),
    )
    stored_dequantized_weight = dequantize_awq_w4a16_weight(
        natural_params["weight"],
        natural_params["weight_scale"],
        natural_params["weight_zero"],
        group_size=int(cfg.group_size),
    )
    manual = inputs.float().matmul(stored_dequantized_weight.float().t())
    if "bias" in natural_params:
        manual = manual + natural_params["bias"].float().reshape(1, -1)
    manual = manual.contiguous()

    local_max = _max_abs_error(expected, manual)
    local_mean = _mean_abs_error(expected, manual)
    local_self_check_status = "passed" if local_max <= 1e-5 else "failed"

    prefix = str(cfg.layer_prefix)
    tensors: dict[str, Any] = {
        f"{prefix}.weight": _owned_tensor(natural_params["weight"]),
        f"{prefix}.weight_scale": _owned_tensor(natural_params["weight_scale"]),
        f"{prefix}.weight_zero": _owned_tensor(natural_params["weight_zero"]),
        f"{prefix}.comfy_quant": _owned_tensor(natural_params["comfy_quant"]),
        "fixture.input": _owned_tensor(inputs),
        "fixture.expected_output": _owned_tensor(expected),
        "fixture.source_dense_weight": _owned_tensor(base["dense_weight"]),
        "fixture.dequantized_weight": _owned_tensor(stored_dequantized_weight),
        "fixture.quantized_weight_uint4": _owned_tensor(debug.quantized_weight),
        "fixture.expected_output_from_dequantized_weight": _owned_tensor(manual),
    }
    if "bias" in natural_params:
        tensors[f"{prefix}.bias"] = _owned_tensor(natural_params["bias"])

    required_layer_tensors = [
        f"{prefix}.weight",
        f"{prefix}.weight_scale",
        f"{prefix}.weight_zero",
        f"{prefix}.comfy_quant",
    ]
    optional_layer_tensors = [f"{prefix}.bias"] if "bias" in natural_params else []
    report: dict[str, Any] = {
        "schema_version": AWQ_W4A16_RUNTIME_FIXTURE_SCHEMA_VERSION,
        "status": "fixture_built" if local_self_check_status == "passed" else "fixture_failed_self_check",
        "artifact_state": AWQ_W4A16_RUNTIME_FIXTURE_ARTIFACT_STATE,
        "format": AWQ_W4A16_FORMAT_NAME,
        "storage_layout": AWQ_W4A16_KITCHEN_NATIVE_LAYOUT_NAME,
        "runtime_reference_state": AWQ_W4A16_REFERENCE_STATE,
        "publishable_svdquant_gptq": False,
        "external_runtime_validation": "not_run",
        "note": (
            "Local PyTorch AWQ W4A16 layer fixture only; this is not external fused-runtime "
            "validation and not a publishable mixed SVDQuant+AWQ checkpoint claim."
        ),
        "layer_prefix": prefix,
        "seed": int(cfg.seed),
        "n": int(cfg.n),
        "k": int(cfg.k),
        "batch": int(cfg.batch),
        "group_size": int(cfg.group_size),
        "scale_dtype": cfg.scale_dtype,
        "tensor_keys": sorted(tensors),
        "external_harness_contract": {
            "scope": "single_layer_awq_w4a16_linear_forward",
            "validation_command": "validate-runtime-fixture-output",
            "forward_input_tensor": "fixture.input",
            "expected_output_tensor": "fixture.expected_output",
            "external_output_tensor": "runtime.output",
            "layer_prefix": prefix,
            "required_layer_tensors": required_layer_tensors,
            "optional_layer_tensors": optional_layer_tensors,
            "group_size": int(cfg.group_size),
            "scale_dtype": cfg.scale_dtype,
            "does_not_validate": [
                "external fused-runtime correctness by itself",
                "mixed SVDQuant W4A4 plus AWQ W4A16 dispatch",
                "full Qwen-Image/Edit model load",
                "full image inference PNG quality",
                "publishable SVDQuant+GPTQ checkpoint status",
            ],
        },
        "local_self_check": {
            "status": local_self_check_status,
            "reference_vs_dequantized_weight_max_abs_error": local_max,
            "reference_vs_dequantized_weight_mean_abs_error": local_mean,
        },
        "checkpoint_quant_config": quant_config,
    }
    metadata = _string_metadata(cfg)
    return AwqW4A16RuntimeFixture(tensors=tensors, metadata=metadata, report=report)


def write_awq_w4a16_runtime_fixture(
    out_dir: str | Path,
    *,
    config: AwqW4A16RuntimeFixtureConfig | None = None,
    fixture_filename: str = DEFAULT_AWQ_RUNTIME_FIXTURE_FILENAME,
    report_filename: str = DEFAULT_AWQ_RUNTIME_FIXTURE_REPORT_FILENAME,
    hash_fixture: bool = True,
) -> WrittenAwqW4A16RuntimeFixture:
    """Write a deterministic AWQ W4A16 runtime fixture and JSON report."""

    save_file = _require_safetensors_save_file()
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    fixture_path = out_path / fixture_filename
    report_path = out_path / report_filename

    fixture = build_awq_w4a16_runtime_fixture(config)
    save_file(fixture.tensors, str(fixture_path), metadata=fixture.metadata)
    report = dict(fixture.report)
    report["fixture_path"] = str(fixture_path)
    report["report_path"] = str(report_path)
    report["fixture_hash_sha256"] = hash_file(fixture_path) if hash_fixture else None
    report["status"] = "fixture_written" if report["local_self_check"]["status"] == "passed" else "fixture_failed_self_check"
    write_json(report_path, report)
    return WrittenAwqW4A16RuntimeFixture(fixture_path=fixture_path, report_path=report_path, report=report)
