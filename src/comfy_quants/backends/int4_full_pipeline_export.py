"""End-to-end SVDQuant W4A4 checkpoint writer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from comfy_quants.algorithms.int4_svdquant.config import (
    CALIBRATED_SVDQUANT_MODE,
    EXPERIMENTAL_SVDQUANT_GPTQ_AWQ_RUNTIME_UNVERIFIED_STATE,
    GPTQ_STATE_LAYER_CORE_INTEGRATED,
    GPTQ_STATE_NOT_IMPLEMENTED,
    LOWRANK_CALIBRATION_OUTPUT_ERROR,
    MIXED_QUANTIZATION_SVD_AWQ_EXPERIMENTAL_STATE,
    MIXED_QUANTIZATION_SVD_ONLY_STATE,
    RUNTIME_CONTRACT_STATIC_ARTIFACT_ONLY,
    SVDQUANT_GPTQ_EXPERIMENTAL_MODE,
    Int4SvdquantPipelineConfig,
    algorithm_notes_for_quantization_mode,
    algorithm_state_for_quantization_mode,
    is_publishable_svdquant_gptq_state,
)
from comfy_quants.algorithms.int4_svdquant.branch_basis import fold_proj_down_for_raw_branch
from comfy_quants.algorithms.int4_svdquant.calibration import ActivationSampleRef, load_activation_sample_refs
from comfy_quants.algorithms.int4_svdquant.gptq import GptqConfig
from comfy_quants.algorithms.int4_svdquant.hessian import (
    GptqHessianLayerRecord,
    load_gptq_hessian_manifest,
    load_gptq_hessian_tensor,
    resolve_gptq_hessian_tensor_path,
)
from comfy_quants.algorithms.int4_svdquant.layer_selection import (
    Int4LinearSelection,
    activation_stats_lookup_candidates,
    select_qwen_image_edit_svdquant_linears,
)
from comfy_quants.algorithms.int4_svdquant.runtime_reference import SVDQUANT_W4A4_RUNTIME_REFERENCE_STATE
from comfy_quants.algorithms.int4_svdquant.stats import ActivationStats, load_activation_stats_map
from comfy_quants.algorithms.int4_svdquant.weight_quant import (
    quantize_linear_weight_to_calibrated_natural_svdquant,
    quantize_linear_weight_to_gptq_natural_svdquant,
    quantize_linear_weight_to_natural_svdquant,
)
from comfy_quants.algorithms.awq_w4a16.weight_quant import quantize_linear_weight_to_awq_w4a16
from comfy_quants.backends.int4_kitchen_export import write_svdquant_w4a4_kitchen_checkpoint
from comfy_quants.backends.safetensors_source import SafetensorsTensorSource
from comfy_quants.core.artifact import QuantArtifact
from comfy_quants.core.errors import PayloadWriteError
from comfy_quants.formats.int4_common import encode_quant_config_tensor
from comfy_quants.formats.awq_w4a16 import awq_w4a16_checkpoint_quant_config
from comfy_quants.formats.kitchen_tilepack import KITCHEN_GROUP_SIZE, KITCHEN_TILEPACK_LAYOUT_NAME, SVDQUANT_W4A4_FORMAT_NAME
from comfy_quants.formats.svdquant_w4a4 import (
    DEFAULT_LOWRANK_BRANCH_INPUT_BASIS,
    LOWRANK_BRANCH_INPUT_BASIS_POST_SMOOTHING,
    LOWRANK_BRANCH_INPUT_BASIS_RAW,
    svdquant_w4a4_checkpoint_quant_config,
)
from comfy_quants.model_adapters.qwen_image_edit_int4 import iter_awq_modulation_prefixes
from comfy_quants.utils.jsonio import write_json


def _require_safetensors():
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise PayloadWriteError("safetensors is required for INT4 full-pipeline export") from exc
    return safe_open


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise PayloadWriteError("torch is required for INT4 full-pipeline export") from exc
    return torch


@dataclass
class Int4FullPipelineReport:
    """Summary of a direct SVDQuant W4A4 checkpoint quantization job."""

    source_checkpoint: str
    output_checkpoint: str
    status: str
    model_family: str
    target_format: str
    storage_layout: str
    quantization_mode: str
    algorithm_state: str
    publishable_svdquant_gptq: bool
    gptq_state: str
    runtime_contract_state: str
    runtime_reference_state: str
    lowrank_branch_input_basis: str
    proj_down_smooth_folded: bool
    mixed_quantization_state: str
    group_size: int
    rank: int
    requested_device: str
    execution_device: str
    selected_layer_count: int
    quantized_layer_count: int
    awq_modulation_layer_count: int
    skipped_tensor_count: int
    output_tensor_count: int
    copied_tensor_count: int
    schema_version: str = "int4_full_pipeline_report.v1"
    pipeline_kind: str = "direct_quantize_to_kitchen_tilepack"
    source_format: str = "safetensors"
    source_layout: str = "single_file"
    source_tensor_count: int = 0
    source_file_count: int = 0
    output_bytes: int = 0
    output_hash: str = ""
    output_hash_state: str = "not_requested"
    calibration_path: str | None = None
    calibration_state: str = "not_provided"
    activation_stats_path: str | None = None
    activation_stats_state: str = "not_provided"
    activation_stats_layer_count: int = 0
    activation_stats_coverage: dict[str, Any] = field(default_factory=dict)
    gptq_hessian_stats_path: str | None = None
    gptq_hessian_stats_state: str = "not_provided"
    gptq_hessian_layer_count: int = 0
    gptq_hessian_coverage: dict[str, Any] = field(default_factory=dict)
    activation_samples_path: str | None = None
    activation_samples_input_root: str | None = None
    activation_samples_state: str = "not_provided"
    activation_samples_layer_count: int = 0
    activation_sample_ref_count: int = 0
    activation_samples_coverage: dict[str, Any] = field(default_factory=dict)
    lowrank_calibration: str = "weight_residual"
    lowrank_ridge: float = 1.0e-6
    gptq_config: dict[str, Any] = field(default_factory=dict)
    algorithm_notes: list[str] = field(default_factory=list)
    selected_layers: list[dict[str, Any]] = field(default_factory=list)
    awq_modulation_layers: list[dict[str, Any]] = field(default_factory=list)
    skipped_tensors: list[str] = field(default_factory=list)
    kitchen_export: dict[str, Any] = field(default_factory=dict)
    cuda_max_memory_allocated_bytes: int | None = None
    cuda_max_memory_reserved_bytes: int | None = None
    written_files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActivationStatsCoverageReport:
    """Coverage and shape validation for a calibrated INT4 activation-stats file."""

    activation_stats_path: str | None
    state: str
    selected_layer_count: int
    loaded_layer_count: int = 0
    matched_layer_count: int = 0
    missing_layer_count: int = 0
    shape_checked_layer_count: int = 0
    shape_mismatch_count: int = 0
    matched_layers: list[dict[str, Any]] = field(default_factory=list)
    missing_layers: list[dict[str, Any]] = field(default_factory=list)
    shape_mismatches: list[dict[str, Any]] = field(default_factory=list)
    load_error: str = ""
    schema_version: str = "int4_activation_stats_coverage.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GptqHessianCoverageReport:
    """Coverage and shape validation for a GPTQ Hessian manifest."""

    gptq_hessian_stats_path: str | None
    state: str
    selected_layer_count: int
    loaded_layer_count: int = 0
    matched_layer_count: int = 0
    missing_layer_count: int = 0
    shape_checked_layer_count: int = 0
    shape_mismatch_count: int = 0
    file_error_count: int = 0
    matched_layers: list[dict[str, Any]] = field(default_factory=list)
    missing_layers: list[dict[str, Any]] = field(default_factory=list)
    shape_mismatches: list[dict[str, Any]] = field(default_factory=list)
    file_errors: list[dict[str, Any]] = field(default_factory=list)
    load_error: str = ""
    schema_version: str = "int4_gptq_hessian_coverage.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActivationSampleCoverageReport:
    """Coverage and shape validation for activation sample tensor references."""

    activation_samples_path: str | None
    state: str
    selected_layer_count: int
    loaded_sample_ref_count: int = 0
    matched_layer_count: int = 0
    missing_layer_count: int = 0
    shape_checked_layer_count: int = 0
    shape_mismatch_count: int = 0
    file_error_count: int = 0
    matched_layers: list[dict[str, Any]] = field(default_factory=list)
    missing_layers: list[dict[str, Any]] = field(default_factory=list)
    shape_mismatches: list[dict[str, Any]] = field(default_factory=list)
    file_errors: list[dict[str, Any]] = field(default_factory=list)
    load_error: str = ""
    schema_version: str = "int4_activation_sample_coverage.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AwqModulationSelection:
    """One Qwen modulation linear selected for AWQ W4A16 conversion."""

    output_prefix: str
    source_prefix: str
    has_bias: bool
    shape: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.shape is not None:
            data["shape"] = list(self.shape)
        return data


def _resolve_torch_device(device: str):
    torch = _require_torch()
    requested = str(device or "auto")
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(requested)
    if device_obj.type == "cuda":
        if not torch.cuda.is_available():
            raise PayloadWriteError(f"CUDA device requested but torch.cuda is not available: {device}")
        index = torch.cuda.current_device() if device_obj.index is None else int(device_obj.index)
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return device_obj


def _iter_source_files(source: SafetensorsTensorSource) -> list[tuple[Path, list[str]]]:
    by_ref: dict[str, list[str]] = {}
    for tensor_name, file_ref in source.file_map.items():
        by_ref.setdefault(file_ref, []).append(tensor_name)
    return [(source.base_dir.joinpath(*PurePosixPath(file_ref).parts), sorted(names)) for file_ref, names in sorted(by_ref.items())]


def _emit_progress(progress: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if progress is not None:
        progress(event)


def _algorithm_contract_fields(quantization_mode: str) -> dict[str, Any]:
    """Return explicit algorithm/runtime state fields for reports and metadata."""
    algorithm_state = algorithm_state_for_quantization_mode(quantization_mode)
    gptq_state = GPTQ_STATE_LAYER_CORE_INTEGRATED if quantization_mode == SVDQUANT_GPTQ_EXPERIMENTAL_MODE else GPTQ_STATE_NOT_IMPLEMENTED
    return {
        "algorithm_state": algorithm_state,
        "publishable_svdquant_gptq": is_publishable_svdquant_gptq_state(algorithm_state),
        "gptq_state": gptq_state,
        "runtime_contract_state": RUNTIME_CONTRACT_STATIC_ARTIFACT_ONLY,
        "mixed_quantization_state": MIXED_QUANTIZATION_SVD_ONLY_STATE,
        "algorithm_notes": algorithm_notes_for_quantization_mode(quantization_mode),
    }


def _proj_down_smooth_folded_for_basis(lowrank_branch_input_basis: str) -> bool:
    if lowrank_branch_input_basis == LOWRANK_BRANCH_INPUT_BASIS_RAW:
        return True
    if lowrank_branch_input_basis == LOWRANK_BRANCH_INPUT_BASIS_POST_SMOOTHING:
        return False
    raise ValueError(f"unsupported low-rank branch input basis: {lowrank_branch_input_basis}")


def _svdquant_runtime_layout_fields(
    lowrank_branch_input_basis: str = DEFAULT_LOWRANK_BRANCH_INPUT_BASIS,
) -> dict[str, Any]:
    proj_down_smooth_folded = _proj_down_smooth_folded_for_basis(lowrank_branch_input_basis)
    return {
        "runtime_reference_state": SVDQUANT_W4A4_RUNTIME_REFERENCE_STATE,
        "lowrank_branch_input_basis": lowrank_branch_input_basis,
        "proj_down_smooth_folded": proj_down_smooth_folded,
    }


def _mixed_quantization_state_for_awq_count(awq_layer_count: int) -> str:
    return MIXED_QUANTIZATION_SVD_AWQ_EXPERIMENTAL_STATE if int(awq_layer_count) > 0 else MIXED_QUANTIZATION_SVD_ONLY_STATE


def _algorithm_state_for_pipeline(quantization_mode: str, base_state: str, awq_layer_count: int) -> str:
    if quantization_mode == SVDQUANT_GPTQ_EXPERIMENTAL_MODE and int(awq_layer_count) > 0:
        return EXPERIMENTAL_SVDQUANT_GPTQ_AWQ_RUNTIME_UNVERIFIED_STATE
    return base_state


def _read_safetensors_state_dict(
    source: SafetensorsTensorSource,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    safe_open = _require_safetensors()
    tensors: dict[str, Any] = {}
    source_files = _iter_source_files(source)
    _emit_progress(
        progress,
        stage="read_prepare",
        source_checkpoint=str(source.source_path),
        source_file_count=len(source_files),
        source_tensor_count=len(source.file_map),
    )
    for source_file_index, (source_file, tensor_names) in enumerate(source_files, start=1):
        if not source_file.is_file():
            raise PayloadWriteError(f"source tensor file is missing: {source_file}")
        _emit_progress(
            progress,
            stage="read_source_file",
            source_file=str(source_file),
            source_file_index=source_file_index,
            source_file_count=len(source_files),
            tensor_count=len(tensor_names),
        )
        with safe_open(str(source_file), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for tensor_name in tensor_names:
                if tensor_name not in available:
                    raise PayloadWriteError(f"safetensors index maps {tensor_name} to {source_file}, but the tensor is absent from that file")
                tensors[tensor_name] = handle.get_tensor(tensor_name).detach().contiguous()
    return tensors


def _selection_with_shapes(tensors: dict[str, Any], selection: list[Int4LinearSelection]) -> list[Int4LinearSelection]:
    shaped: list[Int4LinearSelection] = []
    for item in selection:
        weight = tensors[f"{item.source_prefix}.weight"]
        if int(weight.ndim) != 2:
            raise PayloadWriteError(f"selected linear weight must be rank 2: {item.source_prefix}.weight has shape {tuple(weight.shape)}")
        shaped.append(
            Int4LinearSelection(
                output_prefix=item.output_prefix,
                source_prefix=item.source_prefix,
                smooth_lookup_suffix=item.smooth_lookup_suffix,
                branch_lookup_suffix=item.branch_lookup_suffix,
                act_unsigned=item.act_unsigned,
                has_bias=item.has_bias,
                shape=(int(weight.shape[0]), int(weight.shape[1])),
            )
        )
    return shaped


def _selection_with_source_shapes(source: SafetensorsTensorSource, selection: list[Int4LinearSelection]) -> list[Int4LinearSelection]:
    safe_open = _require_safetensors()
    grouped_weight_names: dict[Path, list[tuple[Int4LinearSelection, str]]] = {}
    for item in selection:
        weight_name = f"{item.source_prefix}.weight"
        if weight_name not in source.file_map:
            raise PayloadWriteError(f"selected linear weight is absent from safetensors source: {weight_name}")
        grouped_weight_names.setdefault(source.file_path_for(weight_name), []).append((item, weight_name))

    shapes_by_prefix: dict[str, tuple[int, int]] = {}
    for file_path, rows in sorted(grouped_weight_names.items(), key=lambda value: str(value[0])):
        if not file_path.is_file():
            raise PayloadWriteError(f"source tensor file is missing: {file_path}")
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for item, weight_name in rows:
                if weight_name not in available:
                    raise PayloadWriteError(f"safetensors index maps {weight_name} to {file_path}, but the tensor is absent from that file")
                shape = [int(dim) for dim in handle.get_slice(weight_name).get_shape()]
                if len(shape) != 2:
                    raise PayloadWriteError(f"selected linear weight must be rank 2: {weight_name} has shape {tuple(shape)}")
                shapes_by_prefix[item.output_prefix] = (shape[0], shape[1])

    shaped: list[Int4LinearSelection] = []
    for item in selection:
        shaped.append(
            Int4LinearSelection(
                output_prefix=item.output_prefix,
                source_prefix=item.source_prefix,
                smooth_lookup_suffix=item.smooth_lookup_suffix,
                branch_lookup_suffix=item.branch_lookup_suffix,
                act_unsigned=item.act_unsigned,
                has_bias=item.has_bias,
                shape=shapes_by_prefix[item.output_prefix],
            )
        )
    return shaped


def _select_qwen_image_edit_awq_modulation(keys: Any) -> list[AwqModulationSelection]:
    keyset = set(keys)
    return [
        AwqModulationSelection(
            output_prefix=prefix,
            source_prefix=prefix,
            has_bias=f"{prefix}.bias" in keyset,
        )
        for prefix in iter_awq_modulation_prefixes(keyset)
    ]


def _awq_selection_with_shapes(tensors: dict[str, Any], selection: list[AwqModulationSelection]) -> list[AwqModulationSelection]:
    shaped: list[AwqModulationSelection] = []
    for item in selection:
        weight = tensors[f"{item.source_prefix}.weight"]
        if int(weight.ndim) != 2:
            raise PayloadWriteError(f"selected AWQ modulation weight must be rank 2: {item.source_prefix}.weight has shape {tuple(weight.shape)}")
        shaped.append(
            AwqModulationSelection(
                output_prefix=item.output_prefix,
                source_prefix=item.source_prefix,
                has_bias=item.has_bias,
                shape=(int(weight.shape[0]), int(weight.shape[1])),
            )
        )
    return shaped


def _awq_selection_with_source_shapes(
    source: SafetensorsTensorSource,
    selection: list[AwqModulationSelection],
) -> list[AwqModulationSelection]:
    safe_open = _require_safetensors()
    grouped_weight_names: dict[Path, list[tuple[AwqModulationSelection, str]]] = {}
    for item in selection:
        weight_name = f"{item.source_prefix}.weight"
        if weight_name not in source.file_map:
            raise PayloadWriteError(f"selected AWQ modulation weight is absent from safetensors source: {weight_name}")
        grouped_weight_names.setdefault(source.file_path_for(weight_name), []).append((item, weight_name))

    shapes_by_prefix: dict[str, tuple[int, int]] = {}
    for file_path, rows in sorted(grouped_weight_names.items(), key=lambda value: str(value[0])):
        if not file_path.is_file():
            raise PayloadWriteError(f"source tensor file is missing: {file_path}")
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for item, weight_name in rows:
                if weight_name not in available:
                    raise PayloadWriteError(f"safetensors index maps {weight_name} to {file_path}, but the tensor is absent from that file")
                shape = [int(dim) for dim in handle.get_slice(weight_name).get_shape()]
                if len(shape) != 2:
                    raise PayloadWriteError(f"selected AWQ modulation weight must be rank 2: {weight_name} has shape {tuple(shape)}")
                shapes_by_prefix[item.output_prefix] = (shape[0], shape[1])

    return [
        AwqModulationSelection(
            output_prefix=item.output_prefix,
            source_prefix=item.source_prefix,
            has_bias=item.has_bias,
            shape=shapes_by_prefix[item.output_prefix],
        )
        for item in selection
    ]


def _validate_output_not_source(output_checkpoint: Path, source: SafetensorsTensorSource) -> None:
    output_resolved = output_checkpoint.resolve(strict=False)
    source_paths = {path.resolve(strict=False) for path, _names in _iter_source_files(source)}
    if output_resolved in source_paths:
        raise PayloadWriteError(f"output checkpoint must not overwrite a source tensor file: {output_checkpoint}")


def _activation_stats_for_selection(stats: dict[str, ActivationStats], item: Int4LinearSelection) -> tuple[ActivationStats, str]:
    for candidate in activation_stats_lookup_candidates(item):
        if candidate in stats:
            return stats[candidate], candidate
    raise PayloadWriteError(
        f"missing activation stats for {item.output_prefix}; tried: {', '.join(activation_stats_lookup_candidates(item))}"
    )


def _build_activation_stats_coverage_report(
    *,
    activation_stats_path: str | None,
    selection: list[Int4LinearSelection],
    stats: dict[str, ActivationStats] | None = None,
    load_error: str = "",
) -> ActivationStatsCoverageReport:
    if load_error:
        return ActivationStatsCoverageReport(
            activation_stats_path=activation_stats_path,
            state="invalid",
            selected_layer_count=len(selection),
            load_error=load_error,
        )

    stats = stats or {}
    matched_layers: list[dict[str, Any]] = []
    missing_layers: list[dict[str, Any]] = []
    shape_mismatches: list[dict[str, Any]] = []
    shape_checked = 0

    for item in selection:
        expected_input_channels = int(item.shape[1]) if item.shape is not None else None
        try:
            layer_stats, stats_key = _activation_stats_for_selection(stats, item)
        except PayloadWriteError:
            missing_layers.append(
                {
                    "output_prefix": item.output_prefix,
                    "source_prefix": item.source_prefix,
                    "expected_input_channels": expected_input_channels,
                    "candidates": activation_stats_lookup_candidates(item),
                }
            )
            continue

        actual_input_channels = int(layer_stats.input_amax.numel())
        matched_layers.append(
            {
                "output_prefix": item.output_prefix,
                "source_prefix": item.source_prefix,
                "activation_stats_key": stats_key,
                "expected_input_channels": expected_input_channels,
                "actual_input_channels": actual_input_channels,
                "sample_count": int(layer_stats.sample_count),
                "element_count": int(layer_stats.element_count),
            }
        )
        if expected_input_channels is not None:
            shape_checked += 1
            if actual_input_channels != expected_input_channels:
                shape_mismatches.append(
                    {
                        "output_prefix": item.output_prefix,
                        "source_prefix": item.source_prefix,
                        "activation_stats_key": stats_key,
                        "expected_input_channels": expected_input_channels,
                        "actual_input_channels": actual_input_channels,
                    }
                )

    state = "valid" if not missing_layers and not shape_mismatches else "invalid"
    return ActivationStatsCoverageReport(
        activation_stats_path=activation_stats_path,
        state=state,
        selected_layer_count=len(selection),
        loaded_layer_count=len(stats),
        matched_layer_count=len(matched_layers),
        missing_layer_count=len(missing_layers),
        shape_checked_layer_count=shape_checked,
        shape_mismatch_count=len(shape_mismatches),
        matched_layers=matched_layers,
        missing_layers=missing_layers,
        shape_mismatches=shape_mismatches,
    )


def _activation_stats_coverage_error_message(report: ActivationStatsCoverageReport) -> str:
    details: list[str] = [
        "activation stats coverage is invalid",
        f"missing_layers={report.missing_layer_count}",
        f"shape_mismatches={report.shape_mismatch_count}",
    ]
    if report.load_error:
        details.append(f"load_error={report.load_error}")
    if report.missing_layers:
        first = report.missing_layers[0]
        details.append(
            "first_missing="
            f"{first['output_prefix']} "
            f"(expected_input_channels={first.get('expected_input_channels')}, "
            f"candidates={', '.join(first.get('candidates') or [])})"
        )
    if report.shape_mismatches:
        first = report.shape_mismatches[0]
        details.append(
            "first_shape_mismatch="
            f"{first['output_prefix']} "
            f"(stats_key={first['activation_stats_key']}, "
            f"expected_input_channels={first['expected_input_channels']}, "
            f"actual_input_channels={first['actual_input_channels']})"
        )
    return "; ".join(details)


def _activation_sample_refs_by_layer(refs: list[ActivationSampleRef]) -> dict[str, list[ActivationSampleRef]]:
    refs_by_layer: dict[str, list[ActivationSampleRef]] = {}
    for ref in refs:
        refs_by_layer.setdefault(ref.layer_name, []).append(ref)
    return refs_by_layer


def _activation_sample_refs_for_selection(
    refs_by_layer: dict[str, list[ActivationSampleRef]],
    item: Int4LinearSelection,
) -> tuple[list[ActivationSampleRef], str]:
    for candidate in activation_stats_lookup_candidates(item):
        if candidate in refs_by_layer:
            return refs_by_layer[candidate], candidate
    raise PayloadWriteError(
        f"missing activation samples for {item.output_prefix}; tried: {', '.join(activation_stats_lookup_candidates(item))}"
    )


def _activation_sample_channel_count(shape: list[int], channel_dim: int) -> tuple[int | None, str]:
    if not shape:
        return None, "activation sample tensor must have at least one dimension"
    dim = int(channel_dim)
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape):
        return None, f"activation sample channel_dim {channel_dim} is out of range for shape {shape}"
    return int(shape[dim]), ""


def _inspect_activation_sample_ref(ref: ActivationSampleRef) -> tuple[list[int], int | None, str]:
    safe_open = _require_safetensors()
    file_path = Path(ref.file_path).expanduser()
    if not file_path.is_file():
        return [], None, f"activation sample file is missing: {file_path}"
    try:
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            if ref.tensor_name not in handle.keys():
                return [], None, f"activation sample tensor {ref.tensor_name!r} is missing from {file_path}"
            shape = [int(dim) for dim in handle.get_slice(ref.tensor_name).get_shape()]
    except Exception as exc:  # pragma: no cover - exact safetensors exceptions vary by version
        return [], None, f"failed to inspect activation sample tensor {file_path}: {exc}"
    channel_count, error = _activation_sample_channel_count(shape, ref.channel_dim)
    if error:
        return shape, None, error
    return shape, channel_count, ""


def _build_activation_sample_coverage_report(
    *,
    activation_samples_path: str | None,
    selection: list[Int4LinearSelection],
    refs: list[ActivationSampleRef] | None = None,
    load_error: str = "",
) -> ActivationSampleCoverageReport:
    if load_error:
        return ActivationSampleCoverageReport(
            activation_samples_path=activation_samples_path,
            state="invalid",
            selected_layer_count=len(selection),
            load_error=load_error,
        )

    refs = refs or []
    refs_by_layer = _activation_sample_refs_by_layer(refs)
    matched_layers: list[dict[str, Any]] = []
    missing_layers: list[dict[str, Any]] = []
    shape_mismatches: list[dict[str, Any]] = []
    file_errors: list[dict[str, Any]] = []
    shape_checked = 0

    for item in selection:
        expected_input_channels = int(item.shape[1]) if item.shape is not None else None
        try:
            layer_refs, samples_key = _activation_sample_refs_for_selection(refs_by_layer, item)
        except PayloadWriteError:
            missing_layers.append(
                {
                    "output_prefix": item.output_prefix,
                    "source_prefix": item.source_prefix,
                    "expected_input_channels": expected_input_channels,
                    "candidates": activation_stats_lookup_candidates(item),
                }
            )
            continue

        inspected_ref_count = 0
        ref_summaries: list[dict[str, Any]] = []
        channel_dims: set[int] = set()
        for ref in layer_refs:
            shape, actual_input_channels, error = _inspect_activation_sample_ref(ref)
            ref_summary = {
                "file_path": ref.file_path,
                "tensor_name": ref.tensor_name,
                "channel_dim": int(ref.channel_dim),
                "sample_id": ref.sample_id,
                "shape": shape,
                "actual_input_channels": actual_input_channels,
            }
            ref_summaries.append(ref_summary)
            channel_dims.add(int(ref.channel_dim))
            if error:
                file_errors.append(
                    {
                        "output_prefix": item.output_prefix,
                        "source_prefix": item.source_prefix,
                        "activation_samples_key": samples_key,
                        "file_path": ref.file_path,
                        "tensor_name": ref.tensor_name,
                        "channel_dim": int(ref.channel_dim),
                        "error": error,
                    }
                )
                continue
            inspected_ref_count += 1
            if expected_input_channels is not None and actual_input_channels != expected_input_channels:
                shape_mismatches.append(
                    {
                        "output_prefix": item.output_prefix,
                        "source_prefix": item.source_prefix,
                        "activation_samples_key": samples_key,
                        "expected_input_channels": expected_input_channels,
                        "actual_input_channels": actual_input_channels,
                        "shape": shape,
                        "file_path": ref.file_path,
                        "tensor_name": ref.tensor_name,
                        "channel_dim": int(ref.channel_dim),
                    }
                )

        if inspected_ref_count > 0:
            shape_checked += 1
        if len(channel_dims) > 1:
            file_errors.append(
                {
                    "output_prefix": item.output_prefix,
                    "source_prefix": item.source_prefix,
                    "activation_samples_key": samples_key,
                    "error": f"inconsistent activation sample channel_dim values: {sorted(channel_dims)}",
                }
            )
        matched_layers.append(
            {
                "output_prefix": item.output_prefix,
                "source_prefix": item.source_prefix,
                "activation_samples_key": samples_key,
                "expected_input_channels": expected_input_channels,
                "sample_ref_count": len(layer_refs),
                "inspected_sample_ref_count": inspected_ref_count,
                "channel_dim": int(layer_refs[0].channel_dim) if layer_refs else -1,
                "samples": ref_summaries,
            }
        )

    state = "valid" if not missing_layers and not shape_mismatches and not file_errors else "invalid"
    return ActivationSampleCoverageReport(
        activation_samples_path=activation_samples_path,
        state=state,
        selected_layer_count=len(selection),
        loaded_sample_ref_count=len(refs),
        matched_layer_count=len(matched_layers),
        missing_layer_count=len(missing_layers),
        shape_checked_layer_count=shape_checked,
        shape_mismatch_count=len(shape_mismatches),
        file_error_count=len(file_errors),
        matched_layers=matched_layers,
        missing_layers=missing_layers,
        shape_mismatches=shape_mismatches,
        file_errors=file_errors,
    )


def _activation_sample_coverage_error_message(report: ActivationSampleCoverageReport) -> str:
    details: list[str] = [
        "activation sample coverage is invalid",
        f"missing_layers={report.missing_layer_count}",
        f"shape_mismatches={report.shape_mismatch_count}",
        f"file_errors={report.file_error_count}",
    ]
    if report.load_error:
        details.append(f"load_error={report.load_error}")
    if report.missing_layers:
        first = report.missing_layers[0]
        details.append(
            "first_missing="
            f"{first['output_prefix']} "
            f"(expected_input_channels={first.get('expected_input_channels')}, "
            f"candidates={', '.join(first.get('candidates') or [])})"
        )
    if report.shape_mismatches:
        first = report.shape_mismatches[0]
        details.append(
            "first_shape_mismatch="
            f"{first['output_prefix']} "
            f"(activation_samples_key={first['activation_samples_key']}, "
            f"expected_input_channels={first['expected_input_channels']}, "
            f"actual_input_channels={first['actual_input_channels']}, "
            f"shape={first['shape']})"
        )
    if report.file_errors:
        first = report.file_errors[0]
        details.append(
            "first_file_error="
            f"{first.get('output_prefix', '')} "
            f"(activation_samples_key={first.get('activation_samples_key', '')}, "
            f"error={first['error']})"
        )
    return "; ".join(details)


def _load_activation_sample_tensors(
    refs: list[ActivationSampleRef],
    *,
    device: Any,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[Any]:
    safe_open = _require_safetensors()
    tensors: list[Any] = []
    for index, ref in enumerate(refs, start=1):
        _emit_progress(
            progress,
            stage="load_activation_sample",
            sample_index=index,
            sample_count=len(refs),
            layer_name=ref.layer_name,
            file_path=ref.file_path,
            tensor_name=ref.tensor_name,
            execution_device=str(device),
        )
        file_path = Path(ref.file_path).expanduser()
        if not file_path.is_file():
            raise PayloadWriteError(f"activation sample file is missing: {file_path}")
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            if ref.tensor_name not in handle.keys():
                raise PayloadWriteError(f"activation sample tensor {ref.tensor_name!r} is missing from {file_path}")
            tensor = handle.get_tensor(ref.tensor_name).detach().contiguous()
        tensors.append(tensor.to(device=device, non_blocking=getattr(device, "type", "") == "cuda").contiguous())
    return tensors


def _mode_requires_activation_stats(mode: str) -> bool:
    return mode in {CALIBRATED_SVDQUANT_MODE, SVDQUANT_GPTQ_EXPERIMENTAL_MODE}


def _mode_requires_gptq_hessians(mode: str) -> bool:
    return mode == SVDQUANT_GPTQ_EXPERIMENTAL_MODE


def _config_uses_activation_samples(config: Int4SvdquantPipelineConfig) -> bool:
    return bool(config.activation_samples_path)


def _config_requires_activation_samples(config: Int4SvdquantPipelineConfig) -> bool:
    return config.lowrank_calibration == LOWRANK_CALIBRATION_OUTPUT_ERROR


def _gptq_hessian_for_selection(
    records: dict[str, GptqHessianLayerRecord],
    item: Int4LinearSelection,
) -> tuple[GptqHessianLayerRecord, str]:
    for candidate in activation_stats_lookup_candidates(item):
        if candidate in records:
            return records[candidate], candidate
    raise PayloadWriteError(
        f"missing GPTQ Hessian for {item.output_prefix}; tried: {', '.join(activation_stats_lookup_candidates(item))}"
    )


def _validate_hessian_record_file(
    *,
    manifest_path: str | Path,
    record: GptqHessianLayerRecord,
    expected_input_channels: int | None,
) -> str:
    safe_open = _require_safetensors()
    if expected_input_channels is None:
        return ""
    try:
        tensor_path = resolve_gptq_hessian_tensor_path(record, manifest_path=manifest_path)
    except ValueError as exc:
        return str(exc)
    if not tensor_path.is_file():
        return f"GPTQ Hessian tensor file is missing: {tensor_path}"
    try:
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            if record.tensor_name not in handle.keys():
                return f"GPTQ Hessian tensor {record.tensor_name!r} is missing from {tensor_path}"
            tensor_shape = [int(dim) for dim in handle.get_slice(record.tensor_name).get_shape()]
    except Exception as exc:  # pragma: no cover - exact safetensors exceptions vary by version
        return f"failed to inspect GPTQ Hessian tensor {tensor_path}: {exc}"
    expected_shape = [int(expected_input_channels), int(expected_input_channels)]
    if tensor_shape != expected_shape:
        return f"GPTQ Hessian tensor shape {tensor_shape} does not match expected {expected_shape}"
    return ""


def _build_gptq_hessian_coverage_report(
    *,
    gptq_hessian_stats_path: str | None,
    selection: list[Int4LinearSelection],
    records: dict[str, GptqHessianLayerRecord] | None = None,
    load_error: str = "",
) -> GptqHessianCoverageReport:
    if load_error:
        return GptqHessianCoverageReport(
            gptq_hessian_stats_path=gptq_hessian_stats_path,
            state="invalid",
            selected_layer_count=len(selection),
            load_error=load_error,
        )

    records = records or {}
    matched_layers: list[dict[str, Any]] = []
    missing_layers: list[dict[str, Any]] = []
    shape_mismatches: list[dict[str, Any]] = []
    file_errors: list[dict[str, Any]] = []
    shape_checked = 0

    for item in selection:
        expected_input_channels = int(item.shape[1]) if item.shape is not None else None
        try:
            record, hessian_key = _gptq_hessian_for_selection(records, item)
        except PayloadWriteError:
            missing_layers.append(
                {
                    "output_prefix": item.output_prefix,
                    "source_prefix": item.source_prefix,
                    "expected_input_channels": expected_input_channels,
                    "candidates": activation_stats_lookup_candidates(item),
                }
            )
            continue

        actual_input_channels = int(record.channel_count)
        matched_layers.append(
            {
                "output_prefix": item.output_prefix,
                "source_prefix": item.source_prefix,
                "gptq_hessian_key": hessian_key,
                "expected_input_channels": expected_input_channels,
                "actual_input_channels": actual_input_channels,
                "sample_count": int(record.sample_count),
                "row_count": int(record.row_count),
                "normalization_count": int(record.normalization_count),
                "file_path": record.file_path,
                "tensor_name": record.tensor_name,
            }
        )
        if expected_input_channels is not None:
            shape_checked += 1
            expected_shape = [expected_input_channels, expected_input_channels]
            record_shape = [int(dim) for dim in record.shape] if record.shape else []
            if actual_input_channels != expected_input_channels or record_shape != expected_shape:
                shape_mismatches.append(
                    {
                        "output_prefix": item.output_prefix,
                        "source_prefix": item.source_prefix,
                        "gptq_hessian_key": hessian_key,
                        "expected_input_channels": expected_input_channels,
                        "actual_input_channels": actual_input_channels,
                        "expected_shape": expected_shape,
                        "actual_shape": record_shape,
                    }
                )
            file_error = _validate_hessian_record_file(
                manifest_path=gptq_hessian_stats_path or "",
                record=record,
                expected_input_channels=expected_input_channels,
            )
            if file_error:
                file_errors.append(
                    {
                        "output_prefix": item.output_prefix,
                        "source_prefix": item.source_prefix,
                        "gptq_hessian_key": hessian_key,
                        "file_path": record.file_path,
                        "tensor_name": record.tensor_name,
                        "error": file_error,
                    }
                )

    state = "valid" if not missing_layers and not shape_mismatches and not file_errors else "invalid"
    return GptqHessianCoverageReport(
        gptq_hessian_stats_path=gptq_hessian_stats_path,
        state=state,
        selected_layer_count=len(selection),
        loaded_layer_count=len(records),
        matched_layer_count=len(matched_layers),
        missing_layer_count=len(missing_layers),
        shape_checked_layer_count=shape_checked,
        shape_mismatch_count=len(shape_mismatches),
        file_error_count=len(file_errors),
        matched_layers=matched_layers,
        missing_layers=missing_layers,
        shape_mismatches=shape_mismatches,
        file_errors=file_errors,
    )


def _gptq_hessian_coverage_error_message(report: GptqHessianCoverageReport) -> str:
    details: list[str] = [
        "GPTQ Hessian coverage is invalid",
        f"missing_layers={report.missing_layer_count}",
        f"shape_mismatches={report.shape_mismatch_count}",
        f"file_errors={report.file_error_count}",
    ]
    if report.load_error:
        details.append(f"load_error={report.load_error}")
    if report.missing_layers:
        first = report.missing_layers[0]
        details.append(
            "first_missing="
            f"{first['output_prefix']} "
            f"(expected_input_channels={first.get('expected_input_channels')}, "
            f"candidates={', '.join(first.get('candidates') or [])})"
        )
    if report.shape_mismatches:
        first = report.shape_mismatches[0]
        details.append(
            "first_shape_mismatch="
            f"{first['output_prefix']} "
            f"(hessian_key={first['gptq_hessian_key']}, "
            f"expected_input_channels={first['expected_input_channels']}, "
            f"actual_input_channels={first['actual_input_channels']}, "
            f"actual_shape={first['actual_shape']})"
        )
    if report.file_errors:
        first = report.file_errors[0]
        details.append(
            "first_file_error="
            f"{first['output_prefix']} "
            f"(hessian_key={first['gptq_hessian_key']}, "
            f"error={first['error']})"
        )
    return "; ".join(details)


def plan_qwen_image_edit_svdquant_w4a4_pipeline(
    *,
    source_checkpoint: str | Path,
    config: Int4SvdquantPipelineConfig,
) -> dict[str, Any]:
    """Return selected layers for a direct INT4 pipeline run without writing output."""
    config.validate()
    source = SafetensorsTensorSource.from_path(source_checkpoint)
    selection = select_qwen_image_edit_svdquant_linears(source.keys())
    awq_selection = _select_qwen_image_edit_awq_modulation(source.keys())
    activation_stats_coverage: dict[str, Any] = {}
    activation_stats_state = "not_required"
    gptq_hessian_coverage: dict[str, Any] = {}
    gptq_hessian_stats_state = "not_required"
    gptq_hessian_layer_count = 0
    activation_samples_coverage: dict[str, Any] = {}
    activation_samples_state = "not_required" if _config_requires_activation_samples(config) else "not_provided"
    activation_samples_layer_count = 0
    activation_sample_ref_count = 0
    if (
        _mode_requires_activation_stats(config.quantization_mode)
        or _mode_requires_gptq_hessians(config.quantization_mode)
        or _config_uses_activation_samples(config)
    ):
        selection = _selection_with_source_shapes(source, selection)
    if awq_selection:
        awq_selection = _awq_selection_with_source_shapes(source, awq_selection)
    if _mode_requires_activation_stats(config.quantization_mode):
        try:
            activation_stats = load_activation_stats_map(config.activation_stats_path, device="cpu")
            coverage = _build_activation_stats_coverage_report(
                activation_stats_path=config.activation_stats_path,
                selection=selection,
                stats=activation_stats,
            )
        except ValueError as exc:
            coverage = _build_activation_stats_coverage_report(
                activation_stats_path=config.activation_stats_path,
                selection=selection,
                load_error=str(exc),
            )
        activation_stats_coverage = coverage.to_dict()
        activation_stats_state = coverage.state
    if _mode_requires_gptq_hessians(config.quantization_mode):
        try:
            gptq_hessian_records = load_gptq_hessian_manifest(config.gptq_hessian_stats_path)
            hessian_coverage = _build_gptq_hessian_coverage_report(
                gptq_hessian_stats_path=config.gptq_hessian_stats_path,
                selection=selection,
                records=gptq_hessian_records,
            )
            gptq_hessian_layer_count = len(gptq_hessian_records)
        except ValueError as exc:
            hessian_coverage = _build_gptq_hessian_coverage_report(
                gptq_hessian_stats_path=config.gptq_hessian_stats_path,
                selection=selection,
                load_error=str(exc),
            )
        gptq_hessian_coverage = hessian_coverage.to_dict()
        gptq_hessian_stats_state = hessian_coverage.state
    if _config_uses_activation_samples(config):
        try:
            activation_sample_refs = load_activation_sample_refs(
                config.activation_samples_path,
                input_root=config.activation_samples_input_root,
            )
            sample_coverage = _build_activation_sample_coverage_report(
                activation_samples_path=config.activation_samples_path,
                selection=selection,
                refs=activation_sample_refs,
            )
            activation_samples_layer_count = len(_activation_sample_refs_by_layer(activation_sample_refs))
            activation_sample_ref_count = len(activation_sample_refs)
        except ValueError as exc:
            sample_coverage = _build_activation_sample_coverage_report(
                activation_samples_path=config.activation_samples_path,
                selection=selection,
                load_error=str(exc),
            )
        activation_samples_coverage = sample_coverage.to_dict()
        activation_samples_state = sample_coverage.state
    contract_fields = _algorithm_contract_fields(config.quantization_mode)
    runtime_layout_fields = _svdquant_runtime_layout_fields(config.lowrank_branch_input_basis)
    mixed_quantization_state = _mixed_quantization_state_for_awq_count(len(awq_selection))
    algorithm_state = _algorithm_state_for_pipeline(config.quantization_mode, str(contract_fields["algorithm_state"]), len(awq_selection))
    return {
        "schema_version": "int4_full_pipeline_plan.v1",
        "pipeline_kind": "direct_quantize_to_kitchen_tilepack",
        "model_family": config.model_family,
        "target_format": config.target_format,
        "storage_layout": KITCHEN_TILEPACK_LAYOUT_NAME,
        "quantization_mode": config.quantization_mode,
        "algorithm_state": algorithm_state,
        "publishable_svdquant_gptq": is_publishable_svdquant_gptq_state(algorithm_state),
        "gptq_state": contract_fields["gptq_state"],
        "runtime_contract_state": contract_fields["runtime_contract_state"],
        "runtime_reference_state": runtime_layout_fields["runtime_reference_state"],
        "lowrank_branch_input_basis": runtime_layout_fields["lowrank_branch_input_basis"],
        "proj_down_smooth_folded": runtime_layout_fields["proj_down_smooth_folded"],
        "mixed_quantization_state": mixed_quantization_state,
        "algorithm_notes": contract_fields["algorithm_notes"],
        "rank": int(config.rank),
        "group_size": int(config.group_size),
        "source": source.describe(),
        "selected_layer_count": len(selection),
        "selected_layers": [item.to_dict() for item in selection],
        "awq_modulation_layer_count": len(awq_selection),
        "awq_modulation_layers": [item.to_dict() for item in awq_selection],
        "calibration_path": config.calibration_path,
        "calibration_state": "provided_not_consumed" if config.calibration_path else "not_provided",
        "activation_stats_path": config.activation_stats_path,
        "activation_stats_state": activation_stats_state,
        "activation_stats_coverage": activation_stats_coverage,
        "gptq_hessian_stats_path": config.gptq_hessian_stats_path,
        "gptq_hessian_stats_state": gptq_hessian_stats_state,
        "gptq_hessian_layer_count": gptq_hessian_layer_count,
        "gptq_hessian_coverage": gptq_hessian_coverage,
        "activation_samples_path": config.activation_samples_path,
        "activation_samples_input_root": config.activation_samples_input_root,
        "activation_samples_state": activation_samples_state,
        "activation_samples_layer_count": activation_samples_layer_count,
        "activation_sample_ref_count": activation_sample_ref_count,
        "activation_samples_coverage": activation_samples_coverage,
        "lowrank_calibration": config.lowrank_calibration,
        "lowrank_ridge": float(config.lowrank_ridge),
        "gptq_config": {
            "damp_percentage": float(config.gptq_damp_percentage),
            "block_size": int(config.gptq_block_size),
            "num_inv_tries": int(config.gptq_num_inv_tries),
            "hessian_block_size": int(config.gptq_hessian_block_size),
        },
    }


def build_qwen_image_edit_svdquant_w4a4_natural_state_dict(
    *,
    source_tensors: dict[str, Any],
    config: Int4SvdquantPipelineConfig,
    device: str = "auto",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a natural SVDQuant state dict directly from dense source tensors."""
    torch = _require_torch()
    config.validate()
    execution_device_obj = _resolve_torch_device(device)
    execution_device = str(execution_device_obj)
    requested_device = str(device or "auto")
    if execution_device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(execution_device_obj)

    selection = _selection_with_shapes(source_tensors, select_qwen_image_edit_svdquant_linears(source_tensors.keys()))
    awq_selection = _awq_selection_with_shapes(source_tensors, _select_qwen_image_edit_awq_modulation(source_tensors.keys()))
    if not selection:
        raise PayloadWriteError("no Qwen-Image-Edit SVDQuant W4A4 candidate layers were found in the checkpoint")

    activation_stats: dict[str, ActivationStats] = {}
    activation_stats_coverage: ActivationStatsCoverageReport | None = None
    if _mode_requires_activation_stats(config.quantization_mode):
        try:
            activation_stats = load_activation_stats_map(config.activation_stats_path, device=execution_device_obj)
        except ValueError as exc:
            raise PayloadWriteError(f"failed to load activation stats: {exc}") from exc
        activation_stats_coverage = _build_activation_stats_coverage_report(
            activation_stats_path=config.activation_stats_path,
            selection=selection,
            stats=activation_stats,
        )
        if activation_stats_coverage.state != "valid":
            raise PayloadWriteError(_activation_stats_coverage_error_message(activation_stats_coverage))

    gptq_hessian_records: dict[str, GptqHessianLayerRecord] = {}
    gptq_hessian_coverage: GptqHessianCoverageReport | None = None
    if _mode_requires_gptq_hessians(config.quantization_mode):
        try:
            gptq_hessian_records = load_gptq_hessian_manifest(config.gptq_hessian_stats_path)
        except ValueError as exc:
            raise PayloadWriteError(f"failed to load GPTQ Hessian stats: {exc}") from exc
        gptq_hessian_coverage = _build_gptq_hessian_coverage_report(
            gptq_hessian_stats_path=config.gptq_hessian_stats_path,
            selection=selection,
            records=gptq_hessian_records,
        )
        if gptq_hessian_coverage.state != "valid":
            raise PayloadWriteError(_gptq_hessian_coverage_error_message(gptq_hessian_coverage))

    activation_sample_refs: list[ActivationSampleRef] = []
    activation_sample_refs_by_layer: dict[str, list[ActivationSampleRef]] = {}
    activation_samples_coverage: ActivationSampleCoverageReport | None = None
    if _config_uses_activation_samples(config):
        try:
            activation_sample_refs = load_activation_sample_refs(
                config.activation_samples_path,
                input_root=config.activation_samples_input_root,
            )
        except ValueError as exc:
            raise PayloadWriteError(f"failed to load activation sample manifest: {exc}") from exc
        activation_sample_refs_by_layer = _activation_sample_refs_by_layer(activation_sample_refs)
        activation_samples_coverage = _build_activation_sample_coverage_report(
            activation_samples_path=config.activation_samples_path,
            selection=selection,
            refs=activation_sample_refs,
        )
        if activation_samples_coverage.state != "valid":
            raise PayloadWriteError(_activation_sample_coverage_error_message(activation_samples_coverage))

    output_tensors: dict[str, Any] = {}
    skip_source_names: set[str] = set()
    replaced_param_keys = (
        "weight",
        "bias",
        "weight_scale",
        "weight_zero",
        "smooth_factor",
        "proj_down",
        "proj_up",
        "input_scale",
        "comfy_quant",
    )
    for item in selection:
        for key in replaced_param_keys:
            skip_source_names.add(f"{item.source_prefix}.{key}")
    for item in awq_selection:
        for key in replaced_param_keys:
            skip_source_names.add(f"{item.source_prefix}.{key}")

    for name, tensor in source_tensors.items():
        if name not in skip_source_names:
            output_tensors[name] = tensor.detach().cpu().contiguous()

    cuda_peak_allocated: int | None = None
    cuda_peak_reserved: int | None = None
    quantized_layers: list[dict[str, Any]] = []
    awq_quantized_layers: list[dict[str, Any]] = []
    runtime_layout_fields = _svdquant_runtime_layout_fields(config.lowrank_branch_input_basis)
    gptq_cfg = GptqConfig(
        damp_percentage=float(config.gptq_damp_percentage),
        block_size=int(config.gptq_block_size),
        num_inv_tries=int(config.gptq_num_inv_tries),
        hessian_block_size=int(config.gptq_hessian_block_size),
    )
    for index, item in enumerate(selection, start=1):
        weight_name = f"{item.source_prefix}.weight"
        bias_name = f"{item.source_prefix}.bias"
        _emit_progress(
            progress,
            stage="quantize_layer",
            source_prefix=item.source_prefix,
            output_prefix=item.output_prefix,
            layer_index=index,
            layer_count=len(selection),
            execution_device=execution_device,
        )
        weight = source_tensors[weight_name].to(device=execution_device_obj, non_blocking=execution_device_obj.type == "cuda")
        try:
            activation_samples_key = ""
            activation_sample_count = 0
            activation_sample_channel_dim = None
            if config.quantization_mode == CALIBRATED_SVDQUANT_MODE:
                layer_stats, stats_key = _activation_stats_for_selection(activation_stats, item)
                hessian_key = ""
                natural = quantize_linear_weight_to_calibrated_natural_svdquant(
                    weight,
                    activation_stats=layer_stats,
                    rank=config.rank,
                    group_size=config.group_size,
                    scale_dtype=config.scale_dtype,
                    smooth_alpha=config.smooth_alpha,
                    smooth_min=config.smooth_min,
                    smooth_max=config.smooth_max,
                ).to_dict()
            elif config.quantization_mode == SVDQUANT_GPTQ_EXPERIMENTAL_MODE:
                layer_stats, stats_key = _activation_stats_for_selection(activation_stats, item)
                hessian_record, hessian_key = _gptq_hessian_for_selection(gptq_hessian_records, item)
                hessian = load_gptq_hessian_tensor(
                    hessian_record,
                    manifest_path=config.gptq_hessian_stats_path,
                    device=execution_device_obj,
                )
                activation_sample_tensors = None
                if config.lowrank_calibration == LOWRANK_CALIBRATION_OUTPUT_ERROR:
                    layer_sample_refs, activation_samples_key = _activation_sample_refs_for_selection(activation_sample_refs_by_layer, item)
                    activation_sample_count = len(layer_sample_refs)
                    activation_sample_channel_dim = int(layer_sample_refs[0].channel_dim)
                    activation_sample_tensors = _load_activation_sample_tensors(
                        layer_sample_refs,
                        device=execution_device_obj,
                        progress=progress,
                    )
                natural = quantize_linear_weight_to_gptq_natural_svdquant(
                    weight,
                    activation_stats=layer_stats,
                    activation_samples=activation_sample_tensors,
                    gptq_hessian=hessian,
                    rank=config.rank,
                    group_size=config.group_size,
                    scale_dtype=config.scale_dtype,
                    smooth_alpha=config.smooth_alpha,
                    smooth_min=config.smooth_min,
                    smooth_max=config.smooth_max,
                    activation_channel_dim=activation_sample_channel_dim if activation_sample_channel_dim is not None else -1,
                    gptq_config=gptq_cfg,
                    gptq_hessian_input_basis="raw_activation",
                    lowrank_calibration=config.lowrank_calibration,
                    lowrank_ridge=config.lowrank_ridge,
                ).to_dict()
            else:
                stats_key = ""
                hessian_key = ""
                natural = quantize_linear_weight_to_natural_svdquant(
                    weight,
                    rank=config.rank,
                    group_size=config.group_size,
                    scale_dtype=config.scale_dtype,
                ).to_dict()
        except ValueError as exc:
            raise PayloadWriteError(f"failed to quantize {weight_name}: {exc}") from exc

        if item.has_bias:
            natural["bias"] = source_tensors[bias_name].detach().to(device=execution_device_obj).contiguous()
        if bool(runtime_layout_fields["proj_down_smooth_folded"]):
            natural["proj_down"] = fold_proj_down_for_raw_branch(natural["proj_down"], natural["smooth_factor"])
        natural["comfy_quant"] = encode_quant_config_tensor(
            svdquant_w4a4_checkpoint_quant_config(
                act_unsigned=item.act_unsigned,
                lowrank_branch_input_basis=str(runtime_layout_fields["lowrank_branch_input_basis"]),
                proj_down_smooth_folded=bool(runtime_layout_fields["proj_down_smooth_folded"]),
            )
        )
        for key, tensor in natural.items():
            output_tensors[f"{item.output_prefix}.{key}"] = tensor.detach().cpu().contiguous()
        layer_report = item.to_dict()
        if stats_key:
            layer_report["activation_stats_key"] = stats_key
        if hessian_key:
            layer_report["gptq_hessian_key"] = hessian_key
        if activation_samples_key:
            layer_report["activation_samples_key"] = activation_samples_key
            layer_report["activation_sample_count"] = activation_sample_count
            layer_report["activation_sample_channel_dim"] = activation_sample_channel_dim
        quantized_layers.append(layer_report)
        if execution_device_obj.type == "cuda":
            del weight, natural
            if "hessian" in locals():
                del hessian
            if "activation_sample_tensors" in locals() and activation_sample_tensors is not None:
                del activation_sample_tensors
            torch.cuda.empty_cache()

    for index, item in enumerate(awq_selection, start=1):
        weight_name = f"{item.source_prefix}.weight"
        bias_name = f"{item.source_prefix}.bias"
        _emit_progress(
            progress,
            stage="quantize_awq_modulation_layer",
            source_prefix=item.source_prefix,
            output_prefix=item.output_prefix,
            layer_index=index,
            layer_count=len(awq_selection),
            execution_device=execution_device,
        )
        weight = source_tensors[weight_name].to(device=execution_device_obj, non_blocking=execution_device_obj.type == "cuda")
        try:
            natural = quantize_linear_weight_to_awq_w4a16(
                weight,
                group_size=config.group_size,
                scale_dtype=config.scale_dtype,
            ).to_dict()
        except ValueError as exc:
            raise PayloadWriteError(f"failed to quantize AWQ modulation {weight_name}: {exc}") from exc
        if item.has_bias:
            natural["bias"] = source_tensors[bias_name].detach().to(device=execution_device_obj).contiguous()
        natural["comfy_quant"] = encode_quant_config_tensor(awq_w4a16_checkpoint_quant_config(group_size=config.group_size))
        for key, tensor in natural.items():
            output_tensors[f"{item.output_prefix}.{key}"] = tensor.detach().cpu().contiguous()
        awq_quantized_layers.append(item.to_dict())
        if execution_device_obj.type == "cuda":
            del weight, natural
            torch.cuda.empty_cache()

    if execution_device_obj.type == "cuda":
        torch.cuda.synchronize(execution_device_obj)
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated(execution_device_obj))
        cuda_peak_reserved = int(torch.cuda.max_memory_reserved(execution_device_obj))
        torch.cuda.empty_cache()

    contract_fields = _algorithm_contract_fields(config.quantization_mode)
    runtime_layout_fields = _svdquant_runtime_layout_fields(config.lowrank_branch_input_basis)
    mixed_quantization_state = _mixed_quantization_state_for_awq_count(len(awq_quantized_layers))
    algorithm_state = _algorithm_state_for_pipeline(config.quantization_mode, str(contract_fields["algorithm_state"]), len(awq_quantized_layers))
    metadata = {
        "model_family": config.model_family,
        "target_format": config.target_format,
        "storage_layout": KITCHEN_TILEPACK_LAYOUT_NAME,
        "quantization_mode": config.quantization_mode,
        "algorithm_state": algorithm_state,
        "publishable_svdquant_gptq": is_publishable_svdquant_gptq_state(algorithm_state),
        "gptq_state": contract_fields["gptq_state"],
        "runtime_contract_state": contract_fields["runtime_contract_state"],
        "runtime_reference_state": runtime_layout_fields["runtime_reference_state"],
        "lowrank_branch_input_basis": runtime_layout_fields["lowrank_branch_input_basis"],
        "proj_down_smooth_folded": runtime_layout_fields["proj_down_smooth_folded"],
        "mixed_quantization_state": mixed_quantization_state,
        "algorithm_notes": contract_fields["algorithm_notes"],
        "rank": int(config.rank),
        "group_size": int(config.group_size),
        "requested_device": requested_device,
        "execution_device": execution_device,
        "selected_layer_count": len(selection),
        "quantized_layer_count": len(quantized_layers),
        "quantized_layers": quantized_layers,
        "awq_modulation_layer_count": len(awq_quantized_layers),
        "awq_modulation_layers": awq_quantized_layers,
        "skipped_tensors": sorted(skip_source_names),
        "cuda_max_memory_allocated_bytes": cuda_peak_allocated,
        "cuda_max_memory_reserved_bytes": cuda_peak_reserved,
        "calibration_path": config.calibration_path,
        "calibration_state": "provided_not_consumed" if config.calibration_path else "not_provided",
        "activation_stats_path": config.activation_stats_path,
        "activation_stats_state": "loaded" if activation_stats else ("not_required" if not _mode_requires_activation_stats(config.quantization_mode) else "missing"),
        "activation_stats_layer_count": len(activation_stats),
        "activation_stats_coverage": activation_stats_coverage.to_dict() if activation_stats_coverage is not None else {},
        "gptq_hessian_stats_path": config.gptq_hessian_stats_path,
        "gptq_hessian_stats_state": "loaded" if gptq_hessian_records else ("not_required" if not _mode_requires_gptq_hessians(config.quantization_mode) else "missing"),
        "gptq_hessian_layer_count": len(gptq_hessian_records),
        "gptq_hessian_coverage": gptq_hessian_coverage.to_dict() if gptq_hessian_coverage is not None else {},
        "activation_samples_path": config.activation_samples_path,
        "activation_samples_input_root": config.activation_samples_input_root,
        "activation_samples_state": (
            "loaded"
            if activation_sample_refs
            else ("not_required" if not _config_requires_activation_samples(config) else "missing")
        ),
        "activation_samples_layer_count": len(activation_sample_refs_by_layer),
        "activation_sample_ref_count": len(activation_sample_refs),
        "activation_samples_coverage": activation_samples_coverage.to_dict() if activation_samples_coverage is not None else {},
        "lowrank_calibration": config.lowrank_calibration,
        "lowrank_ridge": float(config.lowrank_ridge),
        "gptq_config": {
            "damp_percentage": float(config.gptq_damp_percentage),
            "block_size": int(config.gptq_block_size),
            "num_inv_tries": int(config.gptq_num_inv_tries),
            "hessian_block_size": int(config.gptq_hessian_block_size),
        },
    }
    return output_tensors, metadata


def write_qwen_image_edit_svdquant_w4a4_pipeline_checkpoint(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    config: Int4SvdquantPipelineConfig | None = None,
    device: str = "auto",
    hash_output: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
    report_path: str | Path | None = None,
) -> Int4FullPipelineReport:
    """Quantize dense Qwen-Image-Edit tensors and write one tile-packed file."""
    cfg = config or Int4SvdquantPipelineConfig()
    cfg.validate()
    source = SafetensorsTensorSource.from_path(source_checkpoint)
    output_path = Path(output_checkpoint).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_output_not_source(output_path, source)

    source_tensors = _read_safetensors_state_dict(source, progress=progress)
    natural_state, pipeline_meta = build_qwen_image_edit_svdquant_w4a4_natural_state_dict(
        source_tensors=source_tensors,
        config=cfg,
        device=device,
        progress=progress,
    )
    contract_fields = _algorithm_contract_fields(cfg.quantization_mode)
    metadata = {
        "export_format": cfg.target_format,
        "source_format": "safetensors",
        "pipeline_kind": "direct_quantize_to_kitchen_tilepack",
        "model_family": cfg.model_family,
        "quantization_mode": cfg.quantization_mode,
        "algorithm_state": pipeline_meta["algorithm_state"],
        "publishable_svdquant_gptq": pipeline_meta["publishable_svdquant_gptq"],
        "gptq_state": contract_fields["gptq_state"],
        "runtime_contract_state": contract_fields["runtime_contract_state"],
        "runtime_reference_state": pipeline_meta["runtime_reference_state"],
        "lowrank_branch_input_basis": pipeline_meta["lowrank_branch_input_basis"],
        "proj_down_smooth_folded": pipeline_meta["proj_down_smooth_folded"],
        "lowrank_calibration": pipeline_meta["lowrank_calibration"],
        "lowrank_ridge": pipeline_meta["lowrank_ridge"],
        "mixed_quantization_state": pipeline_meta["mixed_quantization_state"],
        "rank": int(cfg.rank),
        "activation_stats_path": cfg.activation_stats_path,
        "gptq_hessian_stats_path": cfg.gptq_hessian_stats_path,
        "awq_modulation_layer_count": int(pipeline_meta.get("awq_modulation_layer_count") or 0),
    }
    kitchen_report = write_svdquant_w4a4_kitchen_checkpoint(
        tensors=natural_state,
        output_checkpoint=output_path,
        source_checkpoint=str(source.source_path),
        source_layout=source.layout,
        device=device,
        require_svdquant=True,
        hash_output=hash_output,
        metadata=metadata,
        progress=progress,
    )
    kitchen_report.source_tensor_count = len(source.file_map)
    kitchen_report.source_file_count = len(set(source.file_map.values()))

    skipped_tensors = list(pipeline_meta["skipped_tensors"])
    quant_peak_allocated = pipeline_meta.get("cuda_max_memory_allocated_bytes")
    quant_peak_reserved = pipeline_meta.get("cuda_max_memory_reserved_bytes")
    peak_allocated_values = [value for value in (quant_peak_allocated, kitchen_report.cuda_max_memory_allocated_bytes) if value is not None]
    peak_reserved_values = [value for value in (quant_peak_reserved, kitchen_report.cuda_max_memory_reserved_bytes) if value is not None]
    cuda_peak_allocated = max(peak_allocated_values) if peak_allocated_values else None
    cuda_peak_reserved = max(peak_reserved_values) if peak_reserved_values else None
    report = Int4FullPipelineReport(
        source_checkpoint=str(source.source_path),
        output_checkpoint=str(output_path),
        status=kitchen_report.status,
        model_family=cfg.model_family,
        target_format=SVDQUANT_W4A4_FORMAT_NAME,
        storage_layout=KITCHEN_TILEPACK_LAYOUT_NAME,
        quantization_mode=cfg.quantization_mode,
        algorithm_state=str(pipeline_meta["algorithm_state"]),
        publishable_svdquant_gptq=bool(pipeline_meta["publishable_svdquant_gptq"]),
        gptq_state=str(pipeline_meta["gptq_state"]),
        runtime_contract_state=str(pipeline_meta["runtime_contract_state"]),
        runtime_reference_state=str(pipeline_meta["runtime_reference_state"]),
        lowrank_branch_input_basis=str(pipeline_meta["lowrank_branch_input_basis"]),
        proj_down_smooth_folded=bool(pipeline_meta["proj_down_smooth_folded"]),
        mixed_quantization_state=str(pipeline_meta["mixed_quantization_state"]),
        group_size=KITCHEN_GROUP_SIZE,
        rank=int(cfg.rank),
        requested_device=str(device or "auto"),
        execution_device=str(kitchen_report.execution_device),
        selected_layer_count=int(pipeline_meta["selected_layer_count"]),
        quantized_layer_count=int(pipeline_meta["quantized_layer_count"]),
        awq_modulation_layer_count=int(pipeline_meta.get("awq_modulation_layer_count") or 0),
        skipped_tensor_count=len(skipped_tensors),
        output_tensor_count=kitchen_report.output_tensor_count,
        copied_tensor_count=kitchen_report.copied_tensor_count,
        source_layout=source.layout,
        source_tensor_count=len(source.file_map),
        source_file_count=len(set(source.file_map.values())),
        output_bytes=kitchen_report.output_bytes,
        output_hash=kitchen_report.output_hash,
        output_hash_state=kitchen_report.output_hash_state,
        calibration_path=cfg.calibration_path,
        calibration_state=str(pipeline_meta["calibration_state"]),
        activation_stats_path=cfg.activation_stats_path,
        activation_stats_state=str(pipeline_meta["activation_stats_state"]),
        activation_stats_layer_count=int(pipeline_meta["activation_stats_layer_count"]),
        activation_stats_coverage=dict(pipeline_meta.get("activation_stats_coverage") or {}),
        gptq_hessian_stats_path=cfg.gptq_hessian_stats_path,
        gptq_hessian_stats_state=str(pipeline_meta["gptq_hessian_stats_state"]),
        gptq_hessian_layer_count=int(pipeline_meta["gptq_hessian_layer_count"]),
        gptq_hessian_coverage=dict(pipeline_meta.get("gptq_hessian_coverage") or {}),
        activation_samples_path=cfg.activation_samples_path,
        activation_samples_input_root=cfg.activation_samples_input_root,
        activation_samples_state=str(pipeline_meta["activation_samples_state"]),
        activation_samples_layer_count=int(pipeline_meta["activation_samples_layer_count"]),
        activation_sample_ref_count=int(pipeline_meta["activation_sample_ref_count"]),
        activation_samples_coverage=dict(pipeline_meta.get("activation_samples_coverage") or {}),
        lowrank_calibration=str(pipeline_meta["lowrank_calibration"]),
        lowrank_ridge=float(pipeline_meta["lowrank_ridge"]),
        gptq_config=dict(pipeline_meta.get("gptq_config") or {}),
        algorithm_notes=list(pipeline_meta.get("algorithm_notes") or []),
        selected_layers=list(pipeline_meta["quantized_layers"]),
        awq_modulation_layers=list(pipeline_meta.get("awq_modulation_layers") or []),
        skipped_tensors=skipped_tensors,
        kitchen_export=kitchen_report.to_dict(),
        cuda_max_memory_allocated_bytes=cuda_peak_allocated,
        cuda_max_memory_reserved_bytes=cuda_peak_reserved,
        written_files=kitchen_report.written_files,
    )
    if report_path is not None:
        write_json(report_path, report.to_dict())
    return report


class Int4FullPipelineExportBackend:
    backend_name = "int4_full_pipeline_export"
    version = "0.1.0"

    def check_compatibility(self, artifact: QuantArtifact) -> dict:
        return {"backend": self.backend_name, "level": "direct_quantize_writer", "artifact_id": artifact.artifact_id}

    def export(self, artifact: QuantArtifact, output_dir: str) -> dict:
        return {"backend": self.backend_name, "output_dir": output_dir, "artifact_id": artifact.artifact_id}


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_backend(Int4FullPipelineExportBackend())
