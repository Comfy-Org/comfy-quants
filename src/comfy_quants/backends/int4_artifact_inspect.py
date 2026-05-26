"""Structural inspection for exported INT4 checkpoint artifacts.

The inspector reads safetensors metadata and tensor shapes directly.  It does
not import or execute a model runtime; full image-generation validation remains
a separate external acceptance step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from comfy_quants.core.errors import PayloadWriteError
from comfy_quants.formats.int4_common import decode_quant_config_tensor
from comfy_quants.formats.kitchen_tilepack import (
    KITCHEN_BLOCK_N,
    KITCHEN_GROUP_SIZE,
    KITCHEN_INTERLEAVE,
    KITCHEN_TILEPACK_LAYOUT_NAME,
    SVDQUANT_REQUIRED_PARAM_KEYS,
    SVDQUANT_W4A4_FORMAT_NAME,
)
from comfy_quants.model_adapters.qwen_image_edit_int4 import GROUPED_QKV_BRANCH_SPECS, transformer_block_prefixes


INT4_ARTIFACT_INSPECTION_SCHEMA_VERSION = "int4_artifact_inspection.v1"


@dataclass(frozen=True)
class TensorInfo:
    """Shape and dtype for one tensor without loading its payload."""

    shape: tuple[int, ...]
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype}


@dataclass
class Int4ArtifactInspectionReport:
    """Result of a structural INT4 safetensors inspection."""

    artifact: str
    artifact_size_bytes: int
    family: str
    requested_format: str
    schema_version: str = INT4_ARTIFACT_INSPECTION_SCHEMA_VERSION
    status: str = "ok"
    tensor_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
    comfy_quant_count: int = 0
    format_counts: dict[str, int] = field(default_factory=dict)
    layout_counts: dict[str, int] = field(default_factory=dict)
    svdquant_w4a4_count: int = 0
    svdquant_lowrank_count: int = 0
    svdquant_no_lowrank_rank0_count: int = 0
    missing_required_tensor_count: int = 0
    bad_layout_count: int = 0
    bad_shape_count: int = 0
    qkv_group_count: int = 0
    qkv_split_prefix_count: int = 0
    qkv_missing_split_count: int = 0
    qkv_full_proj_up_shape_count: int = 0
    qkv_rank0_count: int = 0
    qkv_bad_shared_count: int = 0
    expected_svdquant_layers: int | None = None
    expected_qkv_group_count: int | None = None
    expected_qkv_split_prefix_count: int | None = None
    ok_expected_counts: bool = True
    prefix_examples: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_safetensors():
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - exercised only without package deps
        raise PayloadWriteError("safetensors is required for INT4 artifact inspection") from exc
    return safe_open


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without package deps
        raise PayloadWriteError("torch is required for INT4 artifact inspection") from exc
    return torch


def _add_example(report: Int4ArtifactInspectionReport, *, limit: int, **item: Any) -> None:
    if len(report.examples) < limit:
        report.examples.append(item)


def _tensor_info(handle: Any, key: str) -> TensorInfo:
    tensor_slice = handle.get_slice(key)
    return TensorInfo(
        shape=tuple(int(dim) for dim in tensor_slice.get_shape()),
        dtype=str(tensor_slice.get_dtype()),
    )


def _has_zero_dim(info: TensorInfo | None) -> bool:
    return info is None or any(int(dim) == 0 for dim in info.shape)


def _prefix_param_key(prefix: str, param_key: str) -> str:
    return f"{prefix}.{param_key}"


def _shape_issue_for_svdquant_layer(infos: dict[str, TensorInfo]) -> str | None:
    weight = infos.get("weight")
    weight_scale = infos.get("weight_scale")
    smooth = infos.get("smooth_factor")
    proj_down = infos.get("proj_down")
    proj_up = infos.get("proj_up")
    if weight is None or weight_scale is None or smooth is None or proj_down is None or proj_up is None:
        return "missing_required_tensor"

    expected_weight_tail = (KITCHEN_BLOCK_N // KITCHEN_INTERLEAVE, KITCHEN_INTERLEAVE * KITCHEN_GROUP_SIZE // 2)
    if len(weight.shape) != 4 or tuple(weight.shape[2:]) != expected_weight_tail:
        return f"bad_weight_shape:{weight.shape}"
    if any(dim <= 0 for dim in weight.shape):
        return f"bad_weight_zero_dim:{weight.shape}"

    n_blocks, k_groups = int(weight.shape[0]), int(weight.shape[1])
    n_features = n_blocks * KITCHEN_BLOCK_N
    k_features = k_groups * KITCHEN_GROUP_SIZE

    if weight_scale.shape != (n_blocks, k_groups, KITCHEN_BLOCK_N):
        return f"bad_weight_scale_shape:{weight_scale.shape}"
    if smooth.shape != (k_features,):
        return f"bad_smooth_factor_shape:{smooth.shape}"
    if len(proj_down.shape) != 2 or int(proj_down.shape[0]) != k_features:
        return f"bad_proj_down_shape:{proj_down.shape}"
    if len(proj_up.shape) < 3 or int(proj_up.shape[0]) != n_blocks or int(proj_up.shape[-1]) != KITCHEN_BLOCK_N:
        return f"bad_proj_up_shape:{proj_up.shape}"
    if len(proj_up.shape) == 3 and len(proj_down.shape) == 2 and int(proj_up.shape[1]) != int(proj_down.shape[1]):
        return f"bad_lowrank_rank_mismatch:proj_down={proj_down.shape},proj_up={proj_up.shape}"

    bias = infos.get("bias")
    if bias is not None and bias.shape != (n_features,):
        return f"bad_bias_shape:{bias.shape}"
    return None


def _count_by_key(counts: dict[str, int], key: object | None) -> None:
    name = "missing" if key is None else str(key)
    counts[name] = counts.get(name, 0) + 1


def _decode_quant_config(handle: Any, key: str) -> dict[str, object] | None:
    try:
        return decode_quant_config_tensor(handle.get_tensor(key))
    except Exception as exc:  # noqa: BLE001 - config decode errors are reported structurally
        return {"__decode_error__": str(exc)}


def _collect_svdquant_prefixes(
    *,
    handle: Any,
    keys: set[str],
    requested_format: str,
    report: Int4ArtifactInspectionReport,
    example_limit: int,
) -> tuple[list[str], dict[str, dict[str, TensorInfo]], set[str]]:
    prefixes: list[str] = []
    per_prefix_infos: dict[str, dict[str, TensorInfo]] = {}
    invalid_config_prefixes: set[str] = set()

    for quant_key in sorted(key for key in keys if key.endswith(".comfy_quant")):
        prefix = quant_key[: -len(".comfy_quant")]
        report.comfy_quant_count += 1
        config = _decode_quant_config(handle, quant_key)
        if config is None:
            invalid_config_prefixes.add(prefix)
            _add_example(report, limit=example_limit, prefix=prefix, issue="missing_comfy_quant_config")
            continue
        if "__decode_error__" in config:
            invalid_config_prefixes.add(prefix)
            _add_example(report, limit=example_limit, prefix=prefix, issue="bad_comfy_quant_config", error=config["__decode_error__"])
            continue

        fmt = config.get("format")
        layout = config.get("layout")
        _count_by_key(report.format_counts, fmt)
        _count_by_key(report.layout_counts, layout)
        if fmt != requested_format:
            continue
        prefixes.append(prefix)
        if layout != KITCHEN_TILEPACK_LAYOUT_NAME:
            report.bad_layout_count += 1
            _add_example(report, limit=example_limit, prefix=prefix, issue="bad_layout", layout=layout)

        infos: dict[str, TensorInfo] = {"comfy_quant": _tensor_info(handle, quant_key)}
        missing = []
        for param_key in SVDQUANT_REQUIRED_PARAM_KEYS:
            tensor_key = _prefix_param_key(prefix, param_key)
            if tensor_key not in keys:
                missing.append(param_key)
                continue
            infos[param_key] = _tensor_info(handle, tensor_key)
        optional_bias = _prefix_param_key(prefix, "bias")
        if optional_bias in keys:
            infos["bias"] = _tensor_info(handle, optional_bias)
        per_prefix_infos[prefix] = infos

        if missing:
            report.missing_required_tensor_count += len(missing)
            _add_example(report, limit=example_limit, prefix=prefix, issue="missing_required_tensor", missing=missing)

        shape_issue = _shape_issue_for_svdquant_layer(infos)
        if shape_issue is not None and shape_issue != "missing_required_tensor":
            report.bad_shape_count += 1
            _add_example(report, limit=example_limit, prefix=prefix, issue=shape_issue)

        rank_zero = _has_zero_dim(infos.get("proj_down")) or _has_zero_dim(infos.get("proj_up"))
        if rank_zero:
            report.svdquant_no_lowrank_rank0_count += 1
        elif "proj_down" in infos and "proj_up" in infos:
            report.svdquant_lowrank_count += 1

    return prefixes, per_prefix_infos, invalid_config_prefixes


def _tensor_equal(handle: Any, left_key: str, right_key: str) -> bool:
    torch = _require_torch()
    left = handle.get_tensor(left_key)
    right = handle.get_tensor(right_key)
    return bool(torch.equal(left, right))


def _check_qwen_grouped_qkv(
    *,
    handle: Any,
    prefixes: set[str],
    infos: dict[str, dict[str, TensorInfo]],
    report: Int4ArtifactInspectionReport,
    example_limit: int,
) -> None:
    for block_prefix in transformer_block_prefixes(prefixes):
        for spec in GROUPED_QKV_BRANCH_SPECS:
            group = tuple(f"{block_prefix}.{suffix}" for suffix in spec.target_suffixes)
            missing = [prefix for prefix in group if prefix not in prefixes]
            if missing:
                report.qkv_missing_split_count += len(missing)
                _add_example(report, limit=example_limit, prefix=f"{block_prefix}.{spec.anchor_suffix}", issue="qkv_missing_split", missing=missing)
                continue

            report.qkv_group_count += 1
            report.qkv_split_prefix_count += len(group)
            weight_n_blocks = []
            rank_zero_prefixes = []
            for prefix in group:
                prefix_infos = infos.get(prefix, {})
                weight = prefix_infos.get("weight")
                proj_up = prefix_infos.get("proj_up")
                proj_down = prefix_infos.get("proj_down")
                if weight is not None:
                    weight_n_blocks.append(int(weight.shape[0]))
                if _has_zero_dim(proj_up) or _has_zero_dim(proj_down):
                    rank_zero_prefixes.append(prefix)

                if weight is not None and proj_up is not None and len(proj_up.shape) >= 3:
                    group_n_blocks = sum(
                        int(infos[item]["weight"].shape[0])
                        for item in group
                        if item in infos and "weight" in infos[item]
                    )
                    if group_n_blocks and int(proj_up.shape[0]) == group_n_blocks and int(proj_up.shape[0]) != int(weight.shape[0]):
                        report.qkv_full_proj_up_shape_count += 1
                        _add_example(
                            report,
                            limit=example_limit,
                            prefix=prefix,
                            issue="qkv_full_grouped_proj_up_shape",
                            proj_up_shape=list(proj_up.shape),
                            expected_n_blocks=int(weight.shape[0]),
                            grouped_n_blocks=group_n_blocks,
                        )

            if rank_zero_prefixes:
                report.qkv_rank0_count += len(rank_zero_prefixes)
                _add_example(report, limit=example_limit, prefix=group[0], issue="qkv_rank0_lowrank_branch", prefixes=rank_zero_prefixes)
                continue

            if len(weight_n_blocks) == len(group) and len(set(weight_n_blocks)) != 1:
                report.qkv_bad_shared_count += 1
                _add_example(report, limit=example_limit, prefix=group[0], issue="qkv_unequal_output_block_counts", counts=weight_n_blocks)
                continue

            anchor = group[0]
            shared_bad = False
            for target in group[1:]:
                if not _tensor_equal(handle, _prefix_param_key(anchor, "smooth_factor"), _prefix_param_key(target, "smooth_factor")):
                    shared_bad = True
                    _add_example(report, limit=example_limit, prefix=target, issue="qkv_smooth_factor_not_shared", anchor=anchor)
                    break
                if not _tensor_equal(handle, _prefix_param_key(anchor, "proj_down"), _prefix_param_key(target, "proj_down")):
                    shared_bad = True
                    _add_example(report, limit=example_limit, prefix=target, issue="qkv_proj_down_not_shared", anchor=anchor)
                    break
            if shared_bad:
                report.qkv_bad_shared_count += 1


def _finish_expected_checks(
    *,
    report: Int4ArtifactInspectionReport,
    require_all_lowrank: bool,
    check_qkv_splits: bool,
) -> None:
    errors: list[dict[str, Any]] = []
    if report.expected_svdquant_layers is not None and report.svdquant_w4a4_count != report.expected_svdquant_layers:
        errors.append(
            {
                "check": "expected_svdquant_layers",
                "expected": report.expected_svdquant_layers,
                "actual": report.svdquant_w4a4_count,
            }
        )
    if report.expected_qkv_group_count is not None and report.qkv_group_count != report.expected_qkv_group_count:
        errors.append(
            {
                "check": "expected_qkv_group_count",
                "expected": report.expected_qkv_group_count,
                "actual": report.qkv_group_count,
            }
        )
    if report.expected_qkv_split_prefix_count is not None and report.qkv_split_prefix_count != report.expected_qkv_split_prefix_count:
        errors.append(
            {
                "check": "expected_qkv_split_prefix_count",
                "expected": report.expected_qkv_split_prefix_count,
                "actual": report.qkv_split_prefix_count,
            }
        )
    if require_all_lowrank and report.svdquant_lowrank_count != report.svdquant_w4a4_count:
        errors.append(
            {
                "check": "all_svdquant_layers_have_lowrank",
                "expected": report.svdquant_w4a4_count,
                "actual": report.svdquant_lowrank_count,
            }
        )
    structural_bad_counts = {
        "missing_required_tensor_count": report.missing_required_tensor_count,
        "bad_layout_count": report.bad_layout_count,
        "bad_shape_count": report.bad_shape_count,
    }
    if check_qkv_splits:
        structural_bad_counts.update(
            {
                "qkv_missing_split_count": report.qkv_missing_split_count,
                "qkv_full_proj_up_shape_count": report.qkv_full_proj_up_shape_count,
                "qkv_rank0_count": report.qkv_rank0_count,
                "qkv_bad_shared_count": report.qkv_bad_shared_count,
            }
        )
    for check, count in structural_bad_counts.items():
        if count:
            errors.append({"check": check, "expected": 0, "actual": count})

    report.errors = errors
    report.ok_expected_counts = not errors
    report.status = "ok" if not errors else "failed"


def inspect_svdquant_w4a4_artifact(
    artifact: str | Path,
    *,
    family: str = "qwen_image_edit",
    requested_format: str = SVDQUANT_W4A4_FORMAT_NAME,
    expected_svdquant_layers: int | None = None,
    require_all_lowrank: bool = False,
    check_qkv_splits: bool = False,
    strict_qwen_image_edit_2511: bool = False,
    example_limit: int = 20,
) -> Int4ArtifactInspectionReport:
    """Inspect a single exported INT4 safetensors artifact.

    ``strict_qwen_image_edit_2511`` applies the known production contract for a
    60-block Qwen-Image-Edit-2511 SVDQuant W4A4 tile-packed checkpoint:
    720 SVDQuant linears and 120 split QKV low-rank branch groups.
    """

    safe_open = _require_safetensors()
    path = Path(artifact).expanduser()
    if not path.exists():
        raise PayloadWriteError(f"INT4 artifact does not exist: {path}")
    if requested_format != SVDQUANT_W4A4_FORMAT_NAME:
        raise PayloadWriteError(f"unsupported INT4 artifact inspection format: {requested_format}")

    if strict_qwen_image_edit_2511:
        expected_svdquant_layers = 720 if expected_svdquant_layers is None else expected_svdquant_layers
        require_all_lowrank = True
        check_qkv_splits = True
        expected_qkv_group_count = 120
        expected_qkv_split_prefix_count = 360
    else:
        expected_qkv_group_count = None
        expected_qkv_split_prefix_count = None

    report = Int4ArtifactInspectionReport(
        artifact=str(path),
        artifact_size_bytes=int(path.stat().st_size),
        family=family,
        requested_format=requested_format,
        expected_svdquant_layers=expected_svdquant_layers,
        expected_qkv_group_count=expected_qkv_group_count,
        expected_qkv_split_prefix_count=expected_qkv_split_prefix_count,
    )

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        report.tensor_count = len(keys)
        report.metadata = dict(handle.metadata() or {})
        prefixes, per_prefix_infos, _invalid_config_prefixes = _collect_svdquant_prefixes(
            handle=handle,
            keys=keys,
            requested_format=requested_format,
            report=report,
            example_limit=example_limit,
        )
        report.svdquant_w4a4_count = len(prefixes)
        report.prefix_examples = prefixes[: min(example_limit, len(prefixes))]

        if family == "qwen_image_edit" and check_qkv_splits:
            _check_qwen_grouped_qkv(
                handle=handle,
                prefixes=set(prefixes),
                infos=per_prefix_infos,
                report=report,
                example_limit=example_limit,
            )

    report.format_counts = dict(sorted(report.format_counts.items()))
    report.layout_counts = dict(sorted(report.layout_counts.items()))
    _finish_expected_checks(report=report, require_all_lowrank=require_all_lowrank, check_qkv_splits=check_qkv_splits)
    return report
