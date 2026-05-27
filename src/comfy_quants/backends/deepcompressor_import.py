"""DeepCompressor PTQ artifact import for Qwen-Image-Edit INT4 exports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from comfy_quants.backends.int4_kitchen_export import write_svdquant_w4a4_kitchen_checkpoint
from comfy_quants.algorithms.int4_svdquant.branch_basis import fold_proj_down_for_raw_branch
from comfy_quants.core.artifact import QuantArtifact
from comfy_quants.core.errors import PayloadWriteError
from comfy_quants.formats.int4_common import encode_quant_config_tensor, pack_signed_int4_pairs
from comfy_quants.formats.kitchen_tilepack import KITCHEN_GROUP_SIZE, SVDQUANT_W4A4_FORMAT_NAME
from comfy_quants.formats.svdquant_w4a4 import (
    LOWRANK_BRANCH_INPUT_BASIS_RAW,
    svdquant_w4a4_checkpoint_quant_config,
)
from comfy_quants.model_adapters.qwen_image_edit_int4 import (
    GROUPED_QKV_BRANCH_SPECS,
    GroupedQKVBranchSpec,
    QwenImageEditInt4LinearSpec,
    is_act_unsigned_prefix,
    iter_svdquant_linear_mappings,
    transformer_block_prefixes,
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without package deps
        raise PayloadWriteError("torch is required for DeepCompressor INT4 import") from exc
    return torch


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


def _torch_load_tensor_dict(path: Path) -> dict[str, Any]:
    torch = _require_torch()
    if not path.is_file():
        raise PayloadWriteError(f"DeepCompressor artifact file is missing: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch does not support weights_only
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise PayloadWriteError(f"expected {path.name} to contain a state dict, got {type(value).__name__}")
    return value


@dataclass
class DeepCompressorPTQArtifacts:
    """Loaded local PTQ artifact dictionaries."""

    quant_path: str
    model: dict[str, Any]
    scales: dict[str, Any]
    smooth: dict[str, Any]
    branch: dict[str, Any]
    files: dict[str, str] = field(default_factory=dict)


@dataclass
class DeepCompressorSVDQuantImportReport:
    """Summary of a DeepCompressor artifact import into natural SVDQuant."""

    schema_version: str = "deepcompressor_svdquant_import_report.v1"
    source_format: str = "deepcompressor_ptq_artifacts"
    model_family: str = "qwen_image_edit"
    target_format: str = SVDQUANT_W4A4_FORMAT_NAME
    requested_device: str = "auto"
    execution_device: str = "cpu"
    source_tensor_count: int = 0
    source_scale_count: int = 0
    source_smooth_count: int = 0
    source_branch_count: int = 0
    imported_layer_count: int = 0
    skipped_no_scale_count: int = 0
    copied_tensor_count: int = 0
    output_tensor_count: int = 0
    imported_prefixes: list[str] = field(default_factory=list)
    skipped_no_scale_prefixes: list[str] = field(default_factory=list)
    source_files: dict[str, str] = field(default_factory=dict)
    cuda_max_memory_allocated_bytes: int | None = None
    cuda_max_memory_reserved_bytes: int | None = None
    lowrank_branch_input_basis: str = LOWRANK_BRANCH_INPUT_BASIS_RAW
    proj_down_smooth_folded: bool = True
    shift_bias_correction_count: int = 0
    shift_bias_corrected_prefixes: list[str] = field(default_factory=list)
    grouped_qkv_branch_count: int = 0
    grouped_qkv_branch_anchors: list[str] = field(default_factory=list)
    grouped_qkv_split_prefixes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_deepcompressor_ptq_artifacts(quant_path: str | Path) -> DeepCompressorPTQArtifacts:
    """Load ``model.pt`` / ``scale.pt`` and optional SVDQuant side-artifact files."""
    root = Path(quant_path).expanduser()
    if not root.is_dir():
        raise PayloadWriteError(f"DeepCompressor quant path is not a directory: {root}")

    model_path = root / "model.pt"
    scale_path = root / "scale.pt"
    smooth_path = root / "smooth.pt"
    branch_path = root / "branch.pt"

    files = {"model": str(model_path), "scale": str(scale_path)}
    model = _torch_load_tensor_dict(model_path)
    scales = _torch_load_tensor_dict(scale_path)
    smooth: dict[str, Any] = {}
    branch: dict[str, Any] = {}
    if smooth_path.is_file():
        smooth = _torch_load_tensor_dict(smooth_path)
        files["smooth"] = str(smooth_path)
    if branch_path.is_file():
        branch = _torch_load_tensor_dict(branch_path)
        files["branch"] = str(branch_path)

    return DeepCompressorPTQArtifacts(
        quant_path=str(root),
        model=model,
        scales=scales,
        smooth=smooth,
        branch=branch,
        files=files,
    )


def _emit_progress(progress: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if progress is not None:
        progress(event)


def _side_dtype(*tensors: Any):
    torch = _require_torch()
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.dtype in (torch.float16, torch.bfloat16):
            return tensor.dtype
    return torch.float16


def _to_tensor(value: Any, *, name: str):
    torch = _require_torch()
    if not torch.is_tensor(value):
        raise PayloadWriteError(f"expected tensor for {name}, got {type(value).__name__}")
    return value


def _as_weight_scale(scale: Any, *, out_features: int, in_features: int, dtype: Any, device: Any, name: str):
    torch = _require_torch()
    scale_tensor = _to_tensor(scale, name=name).to(device=device)
    if in_features % KITCHEN_GROUP_SIZE != 0:
        raise PayloadWriteError(f"{name}: in_features={in_features} is not divisible by group size {KITCHEN_GROUP_SIZE}")
    target_groups = in_features // KITCHEN_GROUP_SIZE

    def _normalize_groups(value: Any, groups: int):
        if groups == target_groups:
            return value
        if groups == 1:
            return value.expand(target_groups, out_features)
        raise PayloadWriteError(
            f"{name}: scale has {groups} K groups, expected {target_groups} or 1 for group size {KITCHEN_GROUP_SIZE}"
        )

    if int(scale_tensor.numel()) == 1:
        natural = scale_tensor.reshape(1, 1).expand(target_groups, out_features)
    elif int(scale_tensor.ndim) == 4:
        if int(scale_tensor.shape[0]) != out_features or int(scale_tensor.shape[1]) != 1 or int(scale_tensor.shape[3]) != 1:
            raise PayloadWriteError(f"{name}: expected scale shape (N, 1, K_groups, 1), got {tuple(scale_tensor.shape)}")
        groups = int(scale_tensor.shape[2])
        natural = _normalize_groups(scale_tensor.reshape(out_features, groups).transpose(0, 1), groups)
    elif int(scale_tensor.ndim) == 2:
        shape = tuple(int(x) for x in scale_tensor.shape)
        if shape == (target_groups, out_features):
            natural = scale_tensor
        elif shape == (out_features, target_groups):
            natural = scale_tensor.transpose(0, 1)
        elif shape == (1, out_features):
            natural = scale_tensor.expand(target_groups, out_features)
        elif shape == (out_features, 1):
            natural = scale_tensor.transpose(0, 1).expand(target_groups, out_features)
        else:
            raise PayloadWriteError(
                f"{name}: expected scale shape (N,K_groups), (K_groups,N), (N,1), or (1,N), got {shape}"
            )
    elif int(scale_tensor.ndim) == 1:
        if int(scale_tensor.numel()) == out_features:
            natural = scale_tensor.reshape(1, out_features).expand(target_groups, out_features)
        elif int(scale_tensor.numel()) == target_groups:
            natural = scale_tensor.reshape(target_groups, 1).expand(target_groups, out_features)
        else:
            raise PayloadWriteError(
                f"{name}: expected 1D scale length N={out_features} or K_groups={target_groups}, got {int(scale_tensor.numel())}"
            )
    else:
        raise PayloadWriteError(f"{name}: unsupported scale rank {int(scale_tensor.ndim)}")

    return natural.to(dtype=dtype).contiguous()


def _smooth_factor(smooth: Any | None, *, in_features: int, dtype: Any, device: Any, prefix: str):
    torch = _require_torch()
    if smooth is None:
        return torch.ones((in_features,), dtype=dtype, device=device)
    tensor = _to_tensor(smooth, name=f"{prefix}.smooth").to(device=device, dtype=dtype).reshape(-1)
    if int(tensor.numel()) == 1:
        return tensor.expand(in_features).contiguous()
    if int(tensor.numel()) != in_features:
        raise PayloadWriteError(f"{prefix}: smooth factor length {int(tensor.numel())} does not match K={in_features}")
    return tensor.contiguous()


def _branch_tensors(branch: Any, *, out_features: int, in_features: int, dtype: Any, device: Any, prefix: str):
    if not isinstance(branch, Mapping):
        raise PayloadWriteError(f"{prefix}: expected low-rank branch dict with a.weight and b.weight")
    if "a.weight" not in branch or "b.weight" not in branch:
        raise PayloadWriteError(f"{prefix}: low-rank branch is missing a.weight or b.weight")
    down_raw = _to_tensor(branch["a.weight"], name=f"{prefix}.branch.a.weight").to(device=device, dtype=dtype)
    up_raw = _to_tensor(branch["b.weight"], name=f"{prefix}.branch.b.weight").to(device=device, dtype=dtype)

    if int(down_raw.ndim) != 2:
        raise PayloadWriteError(f"{prefix}: branch a.weight must be rank 2, got {tuple(down_raw.shape)}")
    if int(up_raw.ndim) != 2:
        raise PayloadWriteError(f"{prefix}: branch b.weight must be rank 2, got {tuple(up_raw.shape)}")

    if int(down_raw.shape[1]) == in_features:
        proj_down = down_raw.transpose(0, 1).contiguous()
    elif int(down_raw.shape[0]) == in_features:
        proj_down = down_raw.contiguous()
    else:
        raise PayloadWriteError(f"{prefix}: branch a.weight shape {tuple(down_raw.shape)} is incompatible with K={in_features}")

    rank = int(proj_down.shape[1])
    if int(up_raw.shape[0]) == out_features:
        proj_up = up_raw.contiguous()
    elif int(up_raw.shape[1]) == out_features:
        proj_up = up_raw.transpose(0, 1).contiguous()
    else:
        raise PayloadWriteError(f"{prefix}: branch b.weight shape {tuple(up_raw.shape)} is incompatible with N={out_features}")
    if int(proj_up.shape[1]) != rank:
        raise PayloadWriteError(f"{prefix}: proj_down rank {rank} does not match proj_up rank {int(proj_up.shape[1])}")
    return proj_down, proj_up


def _quantize_weight(weight: Any, weight_scale: Any, *, prefix: str):
    torch = _require_torch()
    weight_tensor = _to_tensor(weight, name=f"{prefix}.weight")
    if int(weight_tensor.ndim) > 2:
        if int(weight_tensor.numel()) != int(weight_tensor.shape[0]) * int(weight_tensor.shape[1]):
            raise PayloadWriteError(f"{prefix}: only pointwise-conv-compatible rank > 2 weights can be flattened")
        weight_tensor = weight_tensor.reshape(weight_tensor.shape[0], weight_tensor.shape[1])
    if int(weight_tensor.ndim) != 2:
        raise PayloadWriteError(f"{prefix}: expected rank-2 weight, got {tuple(weight_tensor.shape)}")

    out_features, in_features = int(weight_tensor.shape[0]), int(weight_tensor.shape[1])
    groups = in_features // KITCHEN_GROUP_SIZE
    scale_for_weight = weight_scale.transpose(0, 1).to(device=weight_tensor.device, dtype=torch.float32).reshape(
        out_features, groups, 1
    )
    quantized = (weight_tensor.to(dtype=torch.float32).reshape(out_features, groups, KITCHEN_GROUP_SIZE) / scale_for_weight).round()
    if int(quantized.numel()) > 0:
        min_value = float(quantized.min().item())
        max_value = float(quantized.max().item())
        if min_value < -7 or max_value > 7:
            raise PayloadWriteError(
                f"{prefix}: quantized weight range [{min_value:g}, {max_value:g}] exceeds SVDQuant signed emission range [-7, 7]"
            )
    dense_int4 = quantized.to(torch.int8).reshape(out_features, in_features).contiguous()
    return pack_signed_int4_pairs(dense_int4, validate=False)


def _as_shift_vector(shift: Any | None, *, in_features: int, device: Any, prefix: str):
    torch = _require_torch()
    if shift is None:
        return None
    tensor = _to_tensor(shift, name=f"{prefix}.shift").to(device=device, dtype=torch.float32).reshape(-1)
    if int(tensor.numel()) == 1:
        return tensor.expand(in_features).contiguous()
    if int(tensor.numel()) != in_features:
        raise PayloadWriteError(f"{prefix}: shift length {int(tensor.numel())} does not match K={in_features}")
    if bool((~torch.isfinite(tensor)).any().item()):
        raise PayloadWriteError(f"{prefix}: shift contains NaN or Inf values")
    return tensor.contiguous()


def _apply_shift_bias_correction(
    *,
    bias: Any | None,
    shift: Any | None,
    proj_down_raw: Any,
    proj_up: Any,
    out_features: int,
    in_features: int,
    dtype: Any,
    device: Any,
    prefix: str,
):
    torch = _require_torch()
    bias_tensor = None
    if bias is not None:
        bias_tensor = _to_tensor(bias, name=f"{prefix}.bias").to(device=device, dtype=dtype).reshape(-1)
        if int(bias_tensor.numel()) != out_features:
            raise PayloadWriteError(f"{prefix}: bias length {int(bias_tensor.numel())} does not match N={out_features}")

    shift_vector = _as_shift_vector(shift, in_features=in_features, device=device, prefix=prefix)
    if shift_vector is None:
        return bias_tensor, False

    if bias_tensor is None:
        bias_base = torch.zeros((out_features,), dtype=torch.float32, device=device)
    else:
        bias_base = bias_tensor.to(device=device, dtype=torch.float32)
    down = proj_down_raw.to(device=device, dtype=torch.float32)
    up = proj_up.to(device=device, dtype=torch.float32)
    correction = (up @ (down.t() @ shift_vector.reshape(in_features, 1))).reshape(out_features)
    corrected = bias_base + correction
    if bool((~torch.isfinite(corrected)).any().item()):
        raise PayloadWriteError(f"{prefix}: shift bias correction produced NaN or Inf values")
    return corrected.to(dtype=dtype).contiguous(), True


def _lookup_shift(model: Mapping[str, Any], *, source_prefix: str, output_prefix: str) -> Any | None:
    candidates: list[str] = [f"{source_prefix}.shift"]
    if source_prefix.endswith(".linear"):
        candidates.append(f"{source_prefix[: -len('.linear')]}.shift")
    if output_prefix != source_prefix:
        candidates.append(f"{output_prefix}.shift")
    for key in candidates:
        if key in model:
            return model[key]
    return None


def deepcompressor_linear_to_natural_svdquant_params(
    *,
    prefix: str,
    weight: Any,
    scale: Any,
    smooth: Any | None,
    branch: Any,
    bias: Any | None = None,
    shift: Any | None = None,
    subscale: Any | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Convert one DeepCompressor linear artifact family to natural SVDQuant."""
    torch = _require_torch()
    execution_device = _resolve_torch_device(device)
    weight_tensor = _to_tensor(weight, name=f"{prefix}.weight").to(device=execution_device)
    if int(weight_tensor.ndim) > 2:
        weight_tensor = weight_tensor.reshape(weight_tensor.shape[0], weight_tensor.shape[1])
    out_features, in_features = int(weight_tensor.shape[0]), int(weight_tensor.shape[1])
    dtype = _side_dtype(weight_tensor, scale)

    weight_scale = _as_weight_scale(
        scale,
        out_features=out_features,
        in_features=in_features,
        dtype=dtype,
        device=execution_device,
        name=f"{prefix}.weight.scale.0",
    )
    if subscale is not None:
        subscale_tensor = _as_weight_scale(
            subscale,
            out_features=out_features,
            in_features=in_features,
            dtype=dtype,
            device=execution_device,
            name=f"{prefix}.weight.scale.1",
        )
        weight_scale = weight_scale.mul(subscale_tensor).contiguous()

    smooth_factor = _smooth_factor(smooth, in_features=in_features, dtype=dtype, device=execution_device, prefix=prefix)
    proj_down, proj_up = _branch_tensors(
        branch,
        out_features=out_features,
        in_features=in_features,
        dtype=dtype,
        device=execution_device,
        prefix=prefix,
    )
    proj_down_raw = fold_proj_down_for_raw_branch(proj_down, smooth_factor).to(device=execution_device, dtype=dtype)
    bias_tensor, _shift_correction_applied = _apply_shift_bias_correction(
        bias=bias,
        shift=shift,
        proj_down_raw=proj_down_raw,
        proj_up=proj_up,
        out_features=out_features,
        in_features=in_features,
        dtype=dtype,
        device=execution_device,
        prefix=prefix,
    )

    params: dict[str, Any] = {
        "weight": _quantize_weight(weight_tensor, weight_scale, prefix=prefix).to(device="cpu").contiguous(),
        "weight_scale": weight_scale.to(device="cpu").contiguous(),
        "smooth_factor": smooth_factor.to(device="cpu").contiguous(),
        "proj_down": proj_down_raw.to(device="cpu").contiguous(),
        "proj_up": proj_up.to(device="cpu").contiguous(),
    }
    if bias_tensor is not None:
        params["bias"] = bias_tensor.to(device="cpu").contiguous()
    return params


def _drop_prefix_tensors(out: dict[str, Any], prefix: str) -> int:
    keys = [key for key in out if key.startswith(f"{prefix}.")]
    for key in keys:
        out.pop(key, None)
    return len(keys)


def _block_prefix(output_prefix: str, spec: QwenImageEditInt4LinearSpec) -> str:
    suffix = f".{spec.output_suffix}"
    if not output_prefix.endswith(suffix):
        raise PayloadWriteError(f"internal mapping error: {output_prefix} does not end with {suffix}")
    return output_prefix[: -len(suffix)]


def _linear_out_in_features(weight: Any, *, prefix: str) -> tuple[int, int]:
    tensor = _to_tensor(weight, name=f"{prefix}.weight")
    if int(tensor.ndim) > 2:
        if int(tensor.numel()) != int(tensor.shape[0]) * int(tensor.shape[1]):
            raise PayloadWriteError(f"{prefix}: only pointwise-conv-compatible rank > 2 weights can be flattened")
        return int(tensor.shape[0]), int(tensor.shape[1])
    if int(tensor.ndim) != 2:
        raise PayloadWriteError(f"{prefix}: expected rank-2 weight, got {tuple(tensor.shape)}")
    return int(tensor.shape[0]), int(tensor.shape[1])


def _branch_down_rank(branch: Any, *, in_features: int, prefix: str) -> int:
    if not isinstance(branch, Mapping):
        raise PayloadWriteError(f"{prefix}: expected low-rank branch dict with a.weight and b.weight")
    if "a.weight" not in branch or "b.weight" not in branch:
        raise PayloadWriteError(f"{prefix}: low-rank branch is missing a.weight or b.weight")
    down_raw = _to_tensor(branch["a.weight"], name=f"{prefix}.branch.a.weight")
    if int(down_raw.ndim) != 2:
        raise PayloadWriteError(f"{prefix}: branch a.weight must be rank 2, got {tuple(down_raw.shape)}")
    if int(down_raw.shape[1]) == in_features:
        return int(down_raw.shape[0])
    if int(down_raw.shape[0]) == in_features:
        return int(down_raw.shape[1])
    raise PayloadWriteError(f"{prefix}: branch a.weight shape {tuple(down_raw.shape)} is incompatible with K={in_features}")


def _split_grouped_qkv_branch(
    branch: Any,
    *,
    split_sizes: tuple[int, int, int],
    in_features: int,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Return per-target branches when one branch spans concatenated Q/K/V rows."""
    rank = _branch_down_rank(branch, in_features=in_features, prefix=prefix)
    up_raw = _to_tensor(branch["b.weight"], name=f"{prefix}.branch.b.weight")
    if int(up_raw.ndim) != 2:
        raise PayloadWriteError(f"{prefix}: branch b.weight must be rank 2, got {tuple(up_raw.shape)}")

    total_out = sum(split_sizes)
    if int(up_raw.shape[0]) == total_out and int(up_raw.shape[1]) == rank:
        chunks = up_raw.split(split_sizes, dim=0)
    elif int(up_raw.shape[1]) == total_out and int(up_raw.shape[0]) == rank:
        chunks = up_raw.split(split_sizes, dim=1)
    elif int(up_raw.shape[0]) == total_out or int(up_raw.shape[1]) == total_out:
        raise PayloadWriteError(
            f"{prefix}: grouped branch b.weight shape {tuple(up_raw.shape)} does not align with rank={rank}"
        )
    else:
        return None

    return tuple({"a.weight": branch["a.weight"], "b.weight": chunk.contiguous()} for chunk in chunks)  # type: ignore[return-value]


def _try_grouped_qkv_branch_targets(
    *,
    artifacts: DeepCompressorPTQArtifacts,
    block_prefix: str,
    group: GroupedQKVBranchSpec,
    mapping_by_output: Mapping[str, tuple[str, QwenImageEditInt4LinearSpec]],
) -> tuple[list[tuple[str, str, QwenImageEditInt4LinearSpec, dict[str, Any]]], str] | None:
    anchor_prefix = f"{block_prefix}.{group.anchor_suffix}"
    branch = artifacts.branch.get(anchor_prefix)
    if branch is None:
        return None

    target_rows: list[int] = []
    in_features: int | None = None
    target_entries: list[tuple[str, str, QwenImageEditInt4LinearSpec]] = []
    for suffix in group.target_suffixes:
        output_prefix = f"{block_prefix}.{suffix}"
        entry = mapping_by_output.get(output_prefix)
        if entry is None:
            return None
        source_prefix, spec = entry
        if f"{source_prefix}.weight.scale.0" not in artifacts.scales:
            return None
        out_features, target_in_features = _linear_out_in_features(artifacts.model[f"{source_prefix}.weight"], prefix=source_prefix)
        if in_features is None:
            in_features = target_in_features
        elif target_in_features != in_features:
            raise PayloadWriteError(
                f"{anchor_prefix}: grouped QKV targets have mismatched input dimensions "
                f"({in_features} vs {target_in_features})"
            )
        target_rows.append(out_features)
        target_entries.append((output_prefix, source_prefix, spec))

    split_branches = _split_grouped_qkv_branch(
        branch,
        split_sizes=(target_rows[0], target_rows[1], target_rows[2]),
        in_features=int(in_features),
        prefix=anchor_prefix,
    )
    if split_branches is None:
        return None
    return [
        (output_prefix, source_prefix, spec, split_branch)
        for (output_prefix, source_prefix, spec), split_branch in zip(target_entries, split_branches, strict=True)
    ], anchor_prefix


def build_qwen_image_edit_svdquant_natural_state_dict(
    artifacts: DeepCompressorPTQArtifacts,
    *,
    device: str = "auto",
    require_svdquant: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], DeepCompressorSVDQuantImportReport]:
    """Build a natural SVDQuant state dict from Qwen-Image-Edit PTQ artifacts."""
    torch = _require_torch()
    requested_device = str(device or "auto")
    execution_device = _resolve_torch_device(requested_device)
    if execution_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(execution_device)

    output: dict[str, Any] = {}
    copied = 0
    for name, value in artifacts.model.items():
        if torch.is_tensor(value):
            output[name] = value.detach().cpu().contiguous()
            copied += 1

    mappings = iter_svdquant_linear_mappings(artifacts.model.keys())
    mapping_by_output = {output_prefix: (source_prefix, spec) for output_prefix, source_prefix, spec in mappings}
    imported_prefixes: list[str] = []
    skipped_no_scale: list[str] = []
    shift_bias_corrected_prefixes: list[str] = []
    handled_grouped_prefixes: set[str] = set()
    grouped_qkv_branch_anchors: list[str] = []
    grouped_qkv_split_prefixes: list[str] = []

    _emit_progress(
        progress,
        stage="deepcompressor_import_prepare",
        source_tensor_count=len(artifacts.model),
        candidate_layer_count=len(mappings),
        requested_device=requested_device,
        execution_device=str(execution_device),
    )

    for block_prefix in transformer_block_prefixes(artifacts.model.keys()):
        for group in GROUPED_QKV_BRANCH_SPECS:
            grouped = _try_grouped_qkv_branch_targets(
                artifacts=artifacts,
                block_prefix=block_prefix,
                group=group,
                mapping_by_output=mapping_by_output,
            )
            if grouped is None:
                continue
            grouped_targets, anchor_prefix = grouped
            smooth = artifacts.smooth.get(anchor_prefix)
            grouped_qkv_branch_anchors.append(anchor_prefix)
            for output_prefix, source_prefix, _spec, split_branch in grouped_targets:
                scale = artifacts.scales[f"{source_prefix}.weight.scale.0"]
                subscale = artifacts.scales.get(f"{source_prefix}.weight.scale.1")
                _emit_progress(
                    progress,
                    stage="deepcompressor_import_grouped_qkv_layer",
                    prefix=output_prefix,
                    source_prefix=source_prefix,
                    grouped_branch_anchor=anchor_prefix,
                    layer_index=len(imported_prefixes) + 1,
                    layer_count=len(mappings),
                    execution_device=str(execution_device),
                )
                shift = _lookup_shift(artifacts.model, source_prefix=source_prefix, output_prefix=output_prefix)
                params = deepcompressor_linear_to_natural_svdquant_params(
                    prefix=output_prefix,
                    weight=artifacts.model[f"{source_prefix}.weight"],
                    scale=scale,
                    smooth=smooth,
                    branch=split_branch,
                    bias=artifacts.model.get(f"{source_prefix}.bias"),
                    shift=shift,
                    subscale=subscale,
                    device=str(execution_device),
                )
                if shift is not None:
                    shift_bias_corrected_prefixes.append(output_prefix)
                params["comfy_quant"] = encode_quant_config_tensor(
                    svdquant_w4a4_checkpoint_quant_config(
                        act_unsigned=is_act_unsigned_prefix(output_prefix),
                        lowrank_branch_input_basis=LOWRANK_BRANCH_INPUT_BASIS_RAW,
                        proj_down_smooth_folded=True,
                    )
                )

                _drop_prefix_tensors(output, source_prefix)
                _drop_prefix_tensors(output, output_prefix)
                for key, tensor in params.items():
                    output[f"{output_prefix}.{key}"] = tensor.detach().cpu().contiguous()
                imported_prefixes.append(output_prefix)
                grouped_qkv_split_prefixes.append(output_prefix)
                handled_grouped_prefixes.add(output_prefix)
                if execution_device.type == "cuda":
                    torch.cuda.empty_cache()

    for index, (output_prefix, source_prefix, spec) in enumerate(mappings, start=1):
        if output_prefix in handled_grouped_prefixes:
            continue
        scale_key = f"{source_prefix}.weight.scale.0"
        scale = artifacts.scales.get(scale_key)
        if scale is None:
            skipped_no_scale.append(output_prefix)
            continue
        subscale = artifacts.scales.get(f"{source_prefix}.weight.scale.1")
        block = _block_prefix(output_prefix, spec)
        smooth_key = f"{block}.{spec.smooth_lookup_suffix()}"
        branch_key = f"{block}.{spec.branch_lookup_suffix()}"
        if smooth_key not in artifacts.smooth and source_prefix != output_prefix and source_prefix in artifacts.smooth:
            smooth_key = source_prefix
        if branch_key not in artifacts.branch and source_prefix != output_prefix and source_prefix in artifacts.branch:
            branch_key = source_prefix
        branch = artifacts.branch.get(branch_key)
        if branch is None:
            raise PayloadWriteError(f"{output_prefix}: low-rank branch is missing in branch.pt at {branch_key}")

        _emit_progress(
            progress,
            stage="deepcompressor_import_layer",
            prefix=output_prefix,
            source_prefix=source_prefix,
            layer_index=index,
            layer_count=len(mappings),
            execution_device=str(execution_device),
        )
        params = deepcompressor_linear_to_natural_svdquant_params(
            prefix=output_prefix,
            weight=artifacts.model[f"{source_prefix}.weight"],
            scale=scale,
            smooth=artifacts.smooth.get(smooth_key),
            branch=branch,
            bias=artifacts.model.get(f"{source_prefix}.bias"),
            shift=_lookup_shift(artifacts.model, source_prefix=source_prefix, output_prefix=output_prefix),
            subscale=subscale,
            device=str(execution_device),
        )
        if _lookup_shift(artifacts.model, source_prefix=source_prefix, output_prefix=output_prefix) is not None:
            shift_bias_corrected_prefixes.append(output_prefix)
        params["comfy_quant"] = encode_quant_config_tensor(
            svdquant_w4a4_checkpoint_quant_config(
                act_unsigned=is_act_unsigned_prefix(output_prefix),
                lowrank_branch_input_basis=LOWRANK_BRANCH_INPUT_BASIS_RAW,
                proj_down_smooth_folded=True,
            )
        )

        _drop_prefix_tensors(output, source_prefix)
        _drop_prefix_tensors(output, output_prefix)
        for key, tensor in params.items():
            output[f"{output_prefix}.{key}"] = tensor.detach().cpu().contiguous()
        imported_prefixes.append(output_prefix)
        if execution_device.type == "cuda":
            torch.cuda.empty_cache()

    if require_svdquant and not imported_prefixes:
        raise PayloadWriteError("no SVDQuant W4A4 layers were imported from DeepCompressor artifacts")

    cuda_peak_allocated: int | None = None
    cuda_peak_reserved: int | None = None
    if execution_device.type == "cuda":
        torch.cuda.synchronize(execution_device)
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated(execution_device))
        cuda_peak_reserved = int(torch.cuda.max_memory_reserved(execution_device))
        torch.cuda.empty_cache()

    report = DeepCompressorSVDQuantImportReport(
        requested_device=requested_device,
        execution_device=str(execution_device),
        source_tensor_count=len(artifacts.model),
        source_scale_count=len(artifacts.scales),
        source_smooth_count=len(artifacts.smooth),
        source_branch_count=len(artifacts.branch),
        imported_layer_count=len(imported_prefixes),
        skipped_no_scale_count=len(skipped_no_scale),
        copied_tensor_count=copied,
        output_tensor_count=len(output),
        imported_prefixes=imported_prefixes,
        skipped_no_scale_prefixes=skipped_no_scale,
        source_files=artifacts.files,
        cuda_max_memory_allocated_bytes=cuda_peak_allocated,
        cuda_max_memory_reserved_bytes=cuda_peak_reserved,
        shift_bias_correction_count=len(shift_bias_corrected_prefixes),
        shift_bias_corrected_prefixes=shift_bias_corrected_prefixes,
        grouped_qkv_branch_count=len(grouped_qkv_branch_anchors),
        grouped_qkv_branch_anchors=grouped_qkv_branch_anchors,
        grouped_qkv_split_prefixes=grouped_qkv_split_prefixes,
    )
    return output, report


def _max_optional_int(lhs: int | None, rhs: int | None) -> int | None:
    values = [value for value in (lhs, rhs) if value is not None]
    return max(values) if values else None


def write_qwen_image_edit_deepcompressor_svdquant_kitchen_checkpoint(
    *,
    quant_path: str | Path,
    output_checkpoint: str | Path,
    device: str = "auto",
    require_svdquant: bool = True,
    hash_output: bool = False,
    metadata: dict[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
):
    """Import Qwen-Image-Edit DeepCompressor artifacts and write a kitchen checkpoint."""
    artifacts = load_deepcompressor_ptq_artifacts(quant_path)
    natural_tensors, import_report = build_qwen_image_edit_svdquant_natural_state_dict(
        artifacts,
        device=device,
        require_svdquant=require_svdquant,
        progress=progress,
    )
    output_metadata = dict(metadata or {})
    output_metadata.update(
        {
            "source_import_format": "deepcompressor_ptq_artifacts",
            "source_model_family": "qwen_image_edit",
        }
    )
    export_report = write_svdquant_w4a4_kitchen_checkpoint(
        tensors=natural_tensors,
        output_checkpoint=output_checkpoint,
        source_checkpoint=str(quant_path),
        source_layout="deepcompressor_ptq_artifact_dir",
        device=device,
        require_svdquant=require_svdquant,
        hash_output=hash_output,
        metadata=output_metadata,
        progress=progress,
    )
    export_report.source_format = "deepcompressor_ptq_artifacts"
    export_report.source_file_count = len(artifacts.files)
    export_report.source_tensor_count = import_report.source_tensor_count
    export_report.selected_source_files = {Path(path).name: 0 for path in artifacts.files.values()}
    export_report.source_import = import_report.to_dict()
    export_report.cuda_max_memory_allocated_bytes = _max_optional_int(
        export_report.cuda_max_memory_allocated_bytes,
        import_report.cuda_max_memory_allocated_bytes,
    )
    export_report.cuda_max_memory_reserved_bytes = _max_optional_int(
        export_report.cuda_max_memory_reserved_bytes,
        import_report.cuda_max_memory_reserved_bytes,
    )
    return export_report


class DeepCompressorInt4ImportBackend:
    backend_name = "deepcompressor_int4_import"
    version = "0.1.0"

    def check_compatibility(self, artifact: QuantArtifact) -> dict:
        return {"backend": self.backend_name, "level": "artifact_import_bridge", "artifact_id": artifact.artifact_id}

    def export(self, artifact: QuantArtifact, output_dir: str) -> dict:
        return {"backend": self.backend_name, "output_dir": output_dir, "artifact_id": artifact.artifact_id}


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_backend(DeepCompressorInt4ImportBackend())
