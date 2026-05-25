"""Deterministic SVDQuant W4A4 layer fixtures for runtime parity work.

This module writes small safetensors fixtures that pair a kitchen tile-packed
SVDQuant W4A4 layer with oracle inputs and outputs.  The fixtures are intended
for external fused-runtime comparison.  They are not proof that an external
runtime has accepted the format, and they do not import or depend on any model
runtime package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from comfy_quants.algorithms.int4_svdquant.branch_basis import fold_proj_down_for_raw_branch
from comfy_quants.algorithms.int4_svdquant.runtime_reference import (
    GELU_UNSIGNED_SHIFT,
    SVDQUANT_W4A4_RUNTIME_REFERENCE_STATE,
    quantize_activation_w4_signed,
    quantize_activation_w4_unsigned,
    reference_svdquant_w4a4_linear_runtime,
)
from comfy_quants.formats.int4_common import encode_quant_config_tensor, pack_signed_int4_pairs
from comfy_quants.formats.kitchen_tilepack import (
    KITCHEN_BLOCK_N,
    KITCHEN_GROUP_SIZE,
    KITCHEN_TILEPACK_LAYOUT_NAME,
    SVDQUANT_W4A4_FORMAT_NAME,
    to_kitchen_tile_packed_params,
)
from comfy_quants.formats.svdquant_w4a4 import (
    LOWRANK_BRANCH_INPUT_BASIS_POST_SMOOTHING,
    LOWRANK_BRANCH_INPUT_BASIS_RAW,
    svdquant_w4a4_checkpoint_quant_config,
)
from comfy_quants.utils.hashing import hash_file
from comfy_quants.utils.jsonio import write_json

SVDQUANT_W4A4_RUNTIME_FIXTURE_SCHEMA_VERSION = "svdquant_w4a4_runtime_fixture.v1"
DEFAULT_RUNTIME_FIXTURE_FILENAME = "svdquant_w4a4_runtime_fixture.safetensors"
DEFAULT_RUNTIME_FIXTURE_REPORT_FILENAME = "runtime_fixture_report.json"
DEFAULT_RUNTIME_FIXTURE_LAYER_PREFIX = "fixture_layer"
DEFAULT_RUNTIME_FIXTURE_SEED = 1234
DEFAULT_RUNTIME_FIXTURE_N = KITCHEN_BLOCK_N
DEFAULT_RUNTIME_FIXTURE_K = KITCHEN_GROUP_SIZE * 2
DEFAULT_RUNTIME_FIXTURE_RANK = 4
DEFAULT_RUNTIME_FIXTURE_BATCH = 3
RUNTIME_FIXTURE_ARTIFACT_STATE = "local_runtime_fixture_external_runtime_unverified"

ActivationSignedness = Literal["signed", "unsigned"]
LowRankBranchInputBasis = Literal["post_smoothing", "raw"]


@dataclass(frozen=True)
class SVDQuantW4A4RuntimeFixtureConfig:
    """Configuration for one deterministic SVDQuant W4A4 fixture."""

    seed: int = DEFAULT_RUNTIME_FIXTURE_SEED
    n: int = DEFAULT_RUNTIME_FIXTURE_N
    k: int = DEFAULT_RUNTIME_FIXTURE_K
    rank: int = DEFAULT_RUNTIME_FIXTURE_RANK
    batch: int = DEFAULT_RUNTIME_FIXTURE_BATCH
    group_size: int = KITCHEN_GROUP_SIZE
    activation_signedness: ActivationSignedness = "signed"
    lowrank_branch_input_basis: LowRankBranchInputBasis = LOWRANK_BRANCH_INPUT_BASIS_RAW
    include_bias: bool = True
    layer_prefix: str = DEFAULT_RUNTIME_FIXTURE_LAYER_PREFIX


@dataclass(frozen=True)
class SVDQuantW4A4RuntimeFixture:
    """In-memory fixture tensors and the JSON-serializable report."""

    tensors: dict[str, Any]
    metadata: dict[str, str]
    report: dict[str, Any]


@dataclass(frozen=True)
class WrittenSVDQuantW4A4RuntimeFixture:
    """Paths and report for a written runtime fixture."""

    fixture_path: Path
    report_path: Path
    report: dict[str, Any]


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImportError("torch is required to build SVDQuant W4A4 runtime fixtures") from exc
    return torch


def _require_safetensors_save_file():
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - dependency should be installed by package metadata
        raise ImportError("safetensors is required to write SVDQuant W4A4 runtime fixtures") from exc
    return save_file


def _validate_config(config: SVDQuantW4A4RuntimeFixtureConfig) -> None:
    if config.activation_signedness not in {"signed", "unsigned"}:
        raise ValueError(f"unsupported activation signedness: {config.activation_signedness}")
    if config.lowrank_branch_input_basis not in {LOWRANK_BRANCH_INPUT_BASIS_POST_SMOOTHING, LOWRANK_BRANCH_INPUT_BASIS_RAW}:
        raise ValueError(f"unsupported low-rank branch input basis: {config.lowrank_branch_input_basis}")
    if int(config.group_size) != KITCHEN_GROUP_SIZE:
        raise ValueError(f"runtime fixture requires group size {KITCHEN_GROUP_SIZE}, got {config.group_size}")
    if int(config.n) <= 0 or int(config.n) % KITCHEN_BLOCK_N != 0:
        raise ValueError(f"n must be a positive multiple of {KITCHEN_BLOCK_N}, got {config.n}")
    if int(config.k) <= 0 or int(config.k) % int(config.group_size) != 0:
        raise ValueError(f"k must be a positive multiple of group size {config.group_size}, got {config.k}")
    if int(config.k) % 2 != 0:
        raise ValueError(f"k must be even for INT4 pair packing, got {config.k}")
    if int(config.rank) <= 0:
        raise ValueError(f"rank must be positive, got {config.rank}")
    if int(config.batch) <= 0:
        raise ValueError(f"batch must be positive, got {config.batch}")
    if not str(config.layer_prefix):
        raise ValueError("layer_prefix must be non-empty")


def _quantize_activation(inputs: Any, *, group_size: int, activation_signedness: ActivationSignedness):
    if activation_signedness == "signed":
        return quantize_activation_w4_signed(inputs, group_size=int(group_size))
    if activation_signedness == "unsigned":
        return quantize_activation_w4_unsigned(inputs, group_size=int(group_size))
    raise ValueError(f"unsupported activation signedness: {activation_signedness}")


def _random_fixture_tensors(config: SVDQuantW4A4RuntimeFixtureConfig) -> dict[str, Any]:
    torch = _require_torch()
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    n = int(config.n)
    k = int(config.k)
    rank = int(config.rank)
    group_size = int(config.group_size)

    codes = torch.randint(-7, 8, (n, k), generator=generator, dtype=torch.int16).to(dtype=torch.int8)
    weight = pack_signed_int4_pairs(codes)
    weight_scale = (torch.rand((k // group_size, n), generator=generator, dtype=torch.float32) * 0.06 + 0.02).to(
        torch.bfloat16
    ).contiguous()
    smooth_factor = (torch.rand((k,), generator=generator, dtype=torch.float32) * 0.8 + 0.75).contiguous()
    proj_down_post_smoothing = (torch.randn((k, rank), generator=generator, dtype=torch.float32) * 0.025).contiguous()
    proj_up = (torch.randn((n, rank), generator=generator, dtype=torch.float32) * 0.03).contiguous()
    inputs = (torch.randn((int(config.batch), k), generator=generator, dtype=torch.float32) * 1.2).contiguous()
    tensors: dict[str, Any] = {
        "weight": weight,
        "weight_scale": weight_scale,
        "smooth_factor": smooth_factor,
        "proj_down_post_smoothing": proj_down_post_smoothing,
        "proj_up": proj_up,
        "inputs": inputs,
    }
    if config.include_bias:
        tensors["bias"] = (torch.randn((n,), generator=generator, dtype=torch.float32) * 0.01).contiguous()
    return tensors


def _max_abs_error(actual: Any, expected: Any) -> float:
    return float((actual.detach().float() - expected.detach().float()).abs().max().item()) if int(actual.numel()) else 0.0


def _mean_abs_error(actual: Any, expected: Any) -> float:
    diff = (actual.detach().float() - expected.detach().float()).abs()
    return float(diff.mean().item()) if int(diff.numel()) else 0.0


def _owned_tensor(tensor: Any):
    return tensor.detach().clone().contiguous()


def _fixture_main_and_lowrank_inputs(inputs: Any, smooth: Any, cfg: SVDQuantW4A4RuntimeFixtureConfig):
    main_inputs = inputs + GELU_UNSIGNED_SHIFT if cfg.activation_signedness == "unsigned" else inputs
    smooth_view = smooth.reshape(*([1] * (int(inputs.ndim) - 1)), int(cfg.k))
    main_post_smoothing_inputs = (main_inputs / smooth_view).contiguous()
    lowrank_inputs = inputs
    return main_inputs.contiguous(), main_post_smoothing_inputs, lowrank_inputs.contiguous()


def _string_metadata(config: SVDQuantW4A4RuntimeFixtureConfig) -> dict[str, str]:
    return {
        "artifact_contract": SVDQUANT_W4A4_RUNTIME_FIXTURE_SCHEMA_VERSION,
        "artifact_state": RUNTIME_FIXTURE_ARTIFACT_STATE,
        "format": SVDQUANT_W4A4_FORMAT_NAME,
        "storage_layout": KITCHEN_TILEPACK_LAYOUT_NAME,
        "runtime_reference_state": SVDQUANT_W4A4_RUNTIME_REFERENCE_STATE,
        "publishable_svdquant_gptq": "false",
        "activation_signedness": config.activation_signedness,
        "lowrank_branch_input_basis": config.lowrank_branch_input_basis,
    }


def build_svdquant_w4a4_runtime_fixture(
    config: SVDQuantW4A4RuntimeFixtureConfig | None = None,
) -> SVDQuantW4A4RuntimeFixture:
    """Build a deterministic SVDQuant W4A4 layer fixture in memory.

    The returned tensors include a single kitchen tile-packed layer under
    ``<layer_prefix>.*`` and oracle tensors under ``fixture.*``.  The oracle
    output is produced by this repository's runtime-like PyTorch reference, so
    the report deliberately keeps ``publishable_svdquant_gptq`` false.
    """

    cfg = config or SVDQuantW4A4RuntimeFixtureConfig()
    _validate_config(cfg)

    base = _random_fixture_tensors(cfg)
    proj_down_post = base["proj_down_post_smoothing"]
    if cfg.lowrank_branch_input_basis == LOWRANK_BRANCH_INPUT_BASIS_RAW:
        stored_proj_down = fold_proj_down_for_raw_branch(proj_down_post, base["smooth_factor"])
        proj_down_smooth_folded = True
    else:
        stored_proj_down = proj_down_post
        proj_down_smooth_folded = False

    quant_config = svdquant_w4a4_checkpoint_quant_config(
        act_unsigned=cfg.activation_signedness == "unsigned",
        lowrank_branch_input_basis=cfg.lowrank_branch_input_basis,
        proj_down_smooth_folded=proj_down_smooth_folded,
    )
    natural_params = {
        "weight": base["weight"],
        "weight_scale": base["weight_scale"],
        "smooth_factor": base["smooth_factor"],
        "proj_down": stored_proj_down,
        "proj_up": base["proj_up"],
        "comfy_quant": encode_quant_config_tensor(quant_config),
    }
    if "bias" in base:
        natural_params["bias"] = base["bias"]

    packed_params = to_kitchen_tile_packed_params(natural_params)
    inputs = base["inputs"]
    smooth = base["smooth_factor"]
    main_inputs, main_post_smoothing_inputs, lowrank_inputs = _fixture_main_and_lowrank_inputs(inputs, smooth, cfg)
    activation = _quantize_activation(
        main_post_smoothing_inputs,
        group_size=int(cfg.group_size),
        activation_signedness=cfg.activation_signedness,
    )

    expected_natural = reference_svdquant_w4a4_linear_runtime(
        inputs,
        natural_params["weight"],
        natural_params["weight_scale"],
        natural_params["smooth_factor"],
        natural_params["proj_down"],
        natural_params["proj_up"],
        bias=natural_params.get("bias"),
        group_size=int(cfg.group_size),
        activation_signedness=cfg.activation_signedness,
        branch_input_basis=cfg.lowrank_branch_input_basis,
    )
    expected_packed = reference_svdquant_w4a4_linear_runtime(
        inputs,
        packed_params["weight"],
        packed_params["weight_scale"],
        packed_params["smooth_factor"],
        packed_params["proj_down"],
        packed_params["proj_up"],
        bias=packed_params.get("bias"),
        group_size=int(cfg.group_size),
        activation_signedness=cfg.activation_signedness,
        branch_input_basis=cfg.lowrank_branch_input_basis,
    )
    post_basis_equivalent = reference_svdquant_w4a4_linear_runtime(
        inputs,
        natural_params["weight"],
        natural_params["weight_scale"],
        natural_params["smooth_factor"],
        proj_down_post,
        natural_params["proj_up"],
        bias=natural_params.get("bias"),
        group_size=int(cfg.group_size),
        activation_signedness=cfg.activation_signedness,
        branch_input_basis=LOWRANK_BRANCH_INPUT_BASIS_POST_SMOOTHING,
    )

    packed_vs_natural_max = _max_abs_error(expected_packed, expected_natural)
    branch_equivalence_max = _max_abs_error(expected_natural, post_basis_equivalent)
    branch_equivalence_mean = _mean_abs_error(expected_natural, post_basis_equivalent)
    local_self_check_status = "passed" if packed_vs_natural_max <= 1e-5 and branch_equivalence_max <= 1e-5 else "failed"

    prefix = str(cfg.layer_prefix)
    tensors: dict[str, Any] = {
        f"{prefix}.weight": _owned_tensor(packed_params["weight"]),
        f"{prefix}.weight_scale": _owned_tensor(packed_params["weight_scale"]),
        f"{prefix}.smooth_factor": _owned_tensor(packed_params["smooth_factor"]),
        f"{prefix}.proj_down": _owned_tensor(packed_params["proj_down"]),
        f"{prefix}.proj_up": _owned_tensor(packed_params["proj_up"]),
        f"{prefix}.comfy_quant": _owned_tensor(packed_params["comfy_quant"]),
        "fixture.input": _owned_tensor(inputs),
        "fixture.main_input": _owned_tensor(main_inputs),
        "fixture.main_post_smoothing_input": _owned_tensor(main_post_smoothing_inputs),
        "fixture.lowrank_input": _owned_tensor(lowrank_inputs),
        "fixture.post_smoothing_input": _owned_tensor(main_post_smoothing_inputs),
        "fixture.activation_q_values": _owned_tensor(activation.q_values),
        "fixture.activation_packed": _owned_tensor(activation.packed),
        "fixture.activation_scale": _owned_tensor(activation.scale),
        "fixture.activation_dequantized": _owned_tensor(activation.dequantized),
        "fixture.expected_output": _owned_tensor(expected_packed),
        "fixture.expected_output_post_smoothing_basis": _owned_tensor(post_basis_equivalent),
        "fixture.proj_down_post_smoothing_reference": _owned_tensor(proj_down_post),
    }
    if "bias" in packed_params:
        tensors[f"{prefix}.bias"] = _owned_tensor(packed_params["bias"])

    required_layer_tensors = [
        f"{prefix}.weight",
        f"{prefix}.weight_scale",
        f"{prefix}.smooth_factor",
        f"{prefix}.proj_down",
        f"{prefix}.proj_up",
        f"{prefix}.comfy_quant",
    ]
    optional_layer_tensors = [f"{prefix}.bias"] if "bias" in packed_params else []
    report: dict[str, Any] = {
        "schema_version": SVDQUANT_W4A4_RUNTIME_FIXTURE_SCHEMA_VERSION,
        "status": "fixture_built" if local_self_check_status == "passed" else "fixture_failed_self_check",
        "artifact_state": RUNTIME_FIXTURE_ARTIFACT_STATE,
        "format": SVDQUANT_W4A4_FORMAT_NAME,
        "storage_layout": KITCHEN_TILEPACK_LAYOUT_NAME,
        "runtime_reference_state": SVDQUANT_W4A4_RUNTIME_REFERENCE_STATE,
        "publishable_svdquant_gptq": False,
        "external_runtime_validation": "not_run",
        "note": (
            "Local PyTorch runtime-like layer fixture only; this is not external fused-runtime "
            "validation and not a publishable SVDQuant+GPTQ checkpoint claim."
        ),
        "layer_prefix": prefix,
        "seed": int(cfg.seed),
        "n": int(cfg.n),
        "k": int(cfg.k),
        "rank": int(cfg.rank),
        "batch": int(cfg.batch),
        "group_size": int(cfg.group_size),
        "activation_signedness": cfg.activation_signedness,
        "act_unsigned": cfg.activation_signedness == "unsigned",
        "lowrank_branch_input_basis": cfg.lowrank_branch_input_basis,
        "proj_down_smooth_folded": proj_down_smooth_folded,
        "tensor_keys": sorted(tensors),
        "external_harness_contract": {
            "scope": "single_layer_svdquant_w4a4_linear_forward",
            "validation_command": "validate-runtime-fixture-output",
            "forward_input_tensor": "fixture.input",
            "main_input_tensor": "fixture.main_input",
            "main_post_smoothing_input_tensor": "fixture.main_post_smoothing_input",
            "lowrank_input_tensor": "fixture.lowrank_input",
            "expected_output_tensor": "fixture.expected_output",
            "external_output_tensor": "runtime.output",
            "layer_prefix": prefix,
            "required_layer_tensors": required_layer_tensors,
            "optional_layer_tensors": optional_layer_tensors,
            "activation_signedness": cfg.activation_signedness,
            "act_unsigned": cfg.activation_signedness == "unsigned",
            "unsigned_activation_shift": GELU_UNSIGNED_SHIFT,
            "post_smoothing_input_compat_alias": "fixture.post_smoothing_input",
            "lowrank_branch_input_basis": cfg.lowrank_branch_input_basis,
            "proj_down_smooth_folded": proj_down_smooth_folded,
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
            "packed_vs_natural_max_abs_error": packed_vs_natural_max,
            "packed_vs_natural_mean_abs_error": _mean_abs_error(expected_packed, expected_natural),
            "branch_basis_equivalence_max_abs_error": branch_equivalence_max,
            "branch_basis_equivalence_mean_abs_error": branch_equivalence_mean,
        },
        "checkpoint_quant_config": quant_config,
    }
    metadata = _string_metadata(cfg)
    return SVDQuantW4A4RuntimeFixture(tensors=tensors, metadata=metadata, report=report)


def write_svdquant_w4a4_runtime_fixture(
    out_dir: str | Path,
    *,
    config: SVDQuantW4A4RuntimeFixtureConfig | None = None,
    fixture_filename: str = DEFAULT_RUNTIME_FIXTURE_FILENAME,
    report_filename: str = DEFAULT_RUNTIME_FIXTURE_REPORT_FILENAME,
    hash_fixture: bool = True,
) -> WrittenSVDQuantW4A4RuntimeFixture:
    """Write a deterministic SVDQuant W4A4 runtime fixture and JSON report."""

    save_file = _require_safetensors_save_file()
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    fixture_path = out_path / fixture_filename
    report_path = out_path / report_filename

    fixture = build_svdquant_w4a4_runtime_fixture(config)
    save_file(fixture.tensors, str(fixture_path), metadata=fixture.metadata)
    report = dict(fixture.report)
    report["fixture_path"] = str(fixture_path)
    report["report_path"] = str(report_path)
    report["fixture_hash_sha256"] = hash_file(fixture_path) if hash_fixture else None
    report["status"] = "fixture_written" if report["local_self_check"]["status"] == "passed" else "fixture_failed_self_check"
    write_json(report_path, report)
    return WrittenSVDQuantW4A4RuntimeFixture(fixture_path=fixture_path, report_path=report_path, report=report)
