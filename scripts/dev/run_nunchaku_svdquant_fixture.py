#!/usr/bin/env python3
"""Run a SVDQuant W4A4 runtime fixture through a Nunchaku-style fused layer.

This is a development harness, not a package entry point.  It exists to compare
this repository's runtime fixture oracle with an optional external fused
runtime.  The library code must remain independent of that runtime.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from comfy_quants.algorithms.int4_svdquant.reference import dequantize_svdquant_w4a4_effective_weight
from comfy_quants.algorithms.int4_svdquant.runtime_reference import (
    GELU_UNSIGNED_SHIFT,
    quantize_activation_w4_signed,
    quantize_activation_w4_unsigned,
)
from comfy_quants.formats.int4_common import decode_quant_config_tensor, unpack_signed_int4_pairs
from comfy_quants.formats.kitchen_tilepack import (
    KITCHEN_GROUP_SIZE,
    SVDQUANT_W4A4_FORMAT_NAME,
    unpack_n_axis,
    unpack_weight_scale,
    unpack_weight_tile,
)
from comfy_quants.utils.jsonio import write_json


DEFAULT_RUNTIME_OUTPUT_TENSOR = "runtime.output"
DEFAULT_EXPECTED_INPUT_TENSOR = "fixture.input"


class HarnessError(RuntimeError):
    """Raised when a fixture cannot be executed by this dev harness."""


@dataclass(frozen=True)
class NunchakuSvdquantFixturePayload:
    """Loaded single-layer fixture tensors used by this dev harness."""

    layer_prefix: str
    qweight: Any
    wscales: Any
    smooth_factor: Any
    proj_down: Any
    proj_up: Any
    bias: Any | None
    inputs: Any
    quant_config: dict[str, object]
    activation_q_values: Any | None = None
    activation_scale: Any | None = None

    @property
    def out_features(self) -> int:
        return int(self.qweight.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.qweight.shape[1]) * 2

    @property
    def rank(self) -> int:
        return int(self.proj_down.shape[1])

    @property
    def batch_rows(self) -> int:
        return int(self.inputs.reshape(-1, self.in_features).shape[0])


@dataclass(frozen=True)
class HarnessReport:
    """JSON report produced by the dev runtime harness."""

    status: str
    fixture: str
    output: str | None
    runtime_output_tensor: str
    assignment_layout: str
    layer_prefix: str
    external_runtime: str
    validation_scope: str
    external_runtime_validation: str
    publishable_svdquant_gptq: bool
    in_features: int
    out_features: int
    rank: int
    group_size: int
    batch_rows: int
    dtype: str
    device: str
    activation_signedness: str
    lowrank_branch_input_basis: str
    proj_down_smooth_folded: bool | None
    notes: list[str]
    component_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_dump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is expected in dev runtime environments
        raise HarnessError("torch is required to execute the Nunchaku fixture harness") from exc
    return torch


def _dtype_from_name(dtype_name: str):
    torch = _load_torch()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise HarnessError(f"unsupported dtype {dtype_name!r}; expected 'float16' or 'bfloat16'")


def _resolve_device(device: str):
    torch = _load_torch()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        raise HarnessError("Nunchaku fused runtime execution requires CUDA; no CUDA device is available")
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise HarnessError(f"Nunchaku fused runtime execution requires a CUDA device, got {resolved}")
    if not torch.cuda.is_available():
        raise HarnessError("requested CUDA execution, but torch.cuda.is_available() is false")
    return resolved


def _infer_layer_prefix(tensors: dict[str, Any], requested: str | None) -> tuple[str, dict[str, object]]:
    if requested:
        key = f"{requested}.comfy_quant"
        if key not in tensors:
            raise HarnessError(f"requested layer prefix {requested!r} is missing {key!r}")
        config = decode_quant_config_tensor(tensors[key])
        if config is None:
            raise HarnessError(f"requested layer prefix {requested!r} has no quant config")
        return requested, config

    matches: list[tuple[str, dict[str, object]]] = []
    for key in sorted(tensors):
        if not key.endswith(".comfy_quant"):
            continue
        prefix = key[: -len(".comfy_quant")]
        config = decode_quant_config_tensor(tensors[key])
        if config is not None and config.get("format") == SVDQUANT_W4A4_FORMAT_NAME:
            matches.append((prefix, config))

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise HarnessError("fixture does not contain a SVDQuant W4A4 layer quant config")
    prefixes = ", ".join(prefix for prefix, _config in matches)
    raise HarnessError(f"fixture contains multiple SVDQuant W4A4 layer prefixes; pass --layer-prefix. Found: {prefixes}")


def _require_tensor(tensors: dict[str, Any], key: str):
    if key not in tensors:
        raise HarnessError(f"fixture is missing required tensor {key!r}")
    return tensors[key]


def _load_payload(
    fixture_path: str | Path,
    *,
    layer_prefix: str | None = None,
    input_tensor: str = DEFAULT_EXPECTED_INPUT_TENSOR,
    require_raw_branch: bool = True,
    require_rank_multiple_of_16: bool = True,
) -> NunchakuSvdquantFixturePayload:
    tensors = load_file(str(Path(fixture_path).expanduser()), device="cpu")
    prefix, quant_config = _infer_layer_prefix(tensors, layer_prefix)
    basis = str(quant_config.get("lowrank_branch_input_basis", ""))
    smooth_folded = quant_config.get("proj_down_smooth_folded")
    if require_raw_branch and (basis != "raw" or smooth_folded is not True):
        raise HarnessError(
            "Nunchaku-style fused SVDQuant computes the low-rank branch from raw inputs. "
            "Generate a fixture with --lowrank-branch-input-basis raw so proj_down is smooth-folded."
        )

    qweight = unpack_weight_tile(_require_tensor(tensors, f"{prefix}.weight")).contiguous()
    wscales = unpack_weight_scale(_require_tensor(tensors, f"{prefix}.weight_scale")).contiguous()
    smooth_factor = _require_tensor(tensors, f"{prefix}.smooth_factor").contiguous()
    proj_down = _require_tensor(tensors, f"{prefix}.proj_down").contiguous()
    proj_up_tensor = _require_tensor(tensors, f"{prefix}.proj_up")
    proj_up = unpack_n_axis(proj_up_tensor).contiguous() if int(proj_up_tensor.ndim) >= 3 else proj_up_tensor.contiguous()
    bias = tensors.get(f"{prefix}.bias")
    if bias is not None:
        bias = bias.contiguous()
    inputs = _require_tensor(tensors, input_tensor).contiguous()
    activation_q_values = tensors.get("fixture.activation_q_values")
    if activation_q_values is not None:
        activation_q_values = activation_q_values.contiguous()
    activation_scale = tensors.get("fixture.activation_scale")
    if activation_scale is not None:
        activation_scale = activation_scale.contiguous()

    payload = NunchakuSvdquantFixturePayload(
        layer_prefix=prefix,
        qweight=qweight,
        wscales=wscales,
        smooth_factor=smooth_factor,
        proj_down=proj_down,
        proj_up=proj_up,
        bias=bias,
        inputs=inputs,
        quant_config=quant_config,
        activation_q_values=activation_q_values,
        activation_scale=activation_scale,
    )
    _validate_payload_shapes(payload)
    if require_rank_multiple_of_16 and payload.rank % 16 != 0:
        raise HarnessError(
            f"fixture rank {payload.rank} is not a multiple of 16; generate a direct-runtime fixture with --rank 16"
        )
    return payload


def _validate_payload_shapes(payload: NunchakuSvdquantFixturePayload) -> None:
    n = payload.out_features
    k = payload.in_features
    rank = payload.rank
    if k <= 0 or n <= 0:
        raise HarnessError(f"invalid qweight shape {tuple(payload.qweight.shape)}")
    if k % KITCHEN_GROUP_SIZE != 0:
        raise HarnessError(f"in_features={k} is not divisible by group size {KITCHEN_GROUP_SIZE}")
    expected_wscales = (k // KITCHEN_GROUP_SIZE, n)
    if tuple(int(x) for x in payload.wscales.shape) != expected_wscales:
        raise HarnessError(f"wscales shape {tuple(payload.wscales.shape)} does not match expected {expected_wscales}")
    if tuple(int(x) for x in payload.smooth_factor.shape) != (k,):
        raise HarnessError(f"smooth_factor shape {tuple(payload.smooth_factor.shape)} does not match expected {(k,)}")
    if tuple(int(x) for x in payload.proj_down.shape) != (k, rank):
        raise HarnessError(f"proj_down shape {tuple(payload.proj_down.shape)} does not match expected {(k, rank)}")
    if tuple(int(x) for x in payload.proj_up.shape) != (n, rank):
        raise HarnessError(f"proj_up shape {tuple(payload.proj_up.shape)} does not match expected {(n, rank)}")
    if payload.bias is not None and tuple(int(x) for x in payload.bias.shape) != (n,):
        raise HarnessError(f"bias shape {tuple(payload.bias.shape)} does not match expected {(n,)}")
    if int(payload.inputs.shape[-1]) != k:
        raise HarnessError(f"input tensor last dimension {int(payload.inputs.shape[-1])} does not match in_features={k}")


def _ceil_divide(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _pad_tensor(tensor: Any, *, divisor: int | tuple[int, ...], dim: int | tuple[int, ...], fill_value: float | int = 0):
    torch = _load_torch()
    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(int(x) for x in dim)
    if isinstance(divisor, int):
        divisors = (int(divisor),) * len(dims)
    else:
        divisors = tuple(int(x) for x in divisor)
    if len(dims) != len(divisors):
        raise HarnessError(f"internal pad error: dims={dims} divisors={divisors}")

    shape = list(tensor.shape)
    changed = False
    for axis, div in zip(dims, divisors, strict=True):
        if div <= 1:
            continue
        padded = _ceil_divide(int(shape[axis]), div) * div
        changed = changed or padded != int(shape[axis])
        shape[axis] = padded
    if not changed:
        return tensor.contiguous()

    out = torch.full(tuple(shape), fill_value, dtype=tensor.dtype, device=tensor.device)
    out[tuple(slice(0, int(extent)) for extent in tensor.shape)] = tensor
    return out.contiguous()


@dataclass(frozen=True)
class _NunchakuW4A4Layout:
    bits: int = 4
    warp_n: int = 128
    comp_n: int = 16

    @property
    def comp_k(self) -> int:
        return 256 // self.bits

    @property
    def num_lanes(self) -> int:
        return 32

    @property
    def num_k_lanes(self) -> int:
        return 4

    @property
    def num_n_lanes(self) -> int:
        return 8

    @property
    def reg_k(self) -> int:
        return 32 // self.bits

    @property
    def reg_n(self) -> int:
        return 1

    @property
    def k_pack_size(self) -> int:
        return self.comp_k // (self.num_k_lanes * self.reg_k)

    @property
    def n_pack_size(self) -> int:
        return self.comp_n // (self.num_n_lanes * self.reg_n)

    @property
    def mem_k(self) -> int:
        return self.comp_k

    @property
    def mem_n(self) -> int:
        return self.warp_n

    @property
    def num_k_packs(self) -> int:
        return self.mem_k // (self.k_pack_size * self.num_k_lanes * self.reg_k)

    @property
    def num_n_packs(self) -> int:
        return self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n)

    @property
    def num_k_unrolls(self) -> int:
        return 2


_NUNCHAKU_W4A4_LAYOUT = _NunchakuW4A4Layout()


def _pack_nunchaku_weight_dense_int4(weight: Any):
    torch = _load_torch()
    layout = _NUNCHAKU_W4A4_LAYOUT
    if int(weight.ndim) != 2:
        raise HarnessError(f"expected dense signed INT4 weight shape (N, K), got {tuple(weight.shape)}")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if n % layout.mem_n != 0:
        raise HarnessError(f"Nunchaku W4A4 runtime pack requires N divisible by {layout.mem_n}, got {n}")
    required_k = layout.mem_k * layout.num_k_unrolls
    if k % required_k != 0:
        raise HarnessError(f"Nunchaku W4A4 runtime pack requires K divisible by {required_k}, got {k}")

    packed = weight.to(dtype=torch.int32).reshape(
        n // layout.mem_n,
        layout.num_n_packs,
        layout.n_pack_size,
        layout.num_n_lanes,
        layout.reg_n,
        k // layout.mem_k,
        layout.num_k_packs,
        layout.k_pack_size,
        layout.num_k_lanes,
        layout.reg_k,
    )
    packed = packed.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
    packed = packed.bitwise_and(0xF)
    shift = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    packed = packed.bitwise_left_shift(shift).sum(dim=-1, dtype=torch.int32)
    return packed.view(dtype=torch.int8).view(n, -1).contiguous()


def _pack_nunchaku_weight_from_pair_bytes(qweight: Any):
    dense = unpack_signed_int4_pairs(qweight).to(dtype=_load_torch().int32)
    return _pack_nunchaku_weight_dense_int4(dense)


def _pack_nunchaku_scale(scale: Any, *, group_size: int):
    layout = _NUNCHAKU_W4A4_LAYOUT
    n = int(scale.shape[0])
    if n % layout.warp_n != 0:
        raise HarnessError(f"Nunchaku scale pack requires leading dimension divisible by {layout.warp_n}, got {n}")
    s_pack_size = min(max(layout.warp_n // layout.num_lanes, 2), 8)
    num_s_lanes = min(layout.num_lanes, layout.warp_n // s_pack_size)
    num_s_packs = layout.warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    if warp_s != layout.warp_n:
        raise HarnessError("internal Nunchaku scale pack configuration is inconsistent")

    packed = scale.reshape(n // warp_s, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)
    packed = packed.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
    return packed.view(-1).contiguous() if int(group_size) == -1 else packed.view(-1, n).contiguous()


def _pack_nunchaku_weight_scale(weight_scale: Any, *, dtype: Any):
    layout = _NUNCHAKU_W4A4_LAYOUT
    if int(weight_scale.ndim) != 2:
        raise HarnessError(f"expected natural weight_scale shape (K/64, N), got {tuple(weight_scale.shape)}")
    groups, n = int(weight_scale.shape[0]), int(weight_scale.shape[1])
    scale = weight_scale.t().contiguous().to(dtype=dtype).view(n, 1, groups, 1)
    scale = _pad_tensor(scale, divisor=(layout.warp_n, layout.num_k_unrolls), dim=(0, 2), fill_value=1)
    return _pack_nunchaku_scale(scale, group_size=KITCHEN_GROUP_SIZE)


def _pack_nunchaku_scale_vector(vector: Any, *, dtype: Any):
    layout = _NUNCHAKU_W4A4_LAYOUT
    scale = vector.reshape(-1, 1).contiguous().to(dtype=dtype)
    scale = _pad_tensor(scale, divisor=layout.warp_n, dim=0, fill_value=1)
    return _pack_nunchaku_scale(scale, group_size=-1)


def _pack_nunchaku_lowrank_weight(weight: Any, *, down: bool, dtype: Any):
    layout = _NUNCHAKU_W4A4_LAYOUT
    if int(weight.ndim) != 2:
        raise HarnessError(f"expected low-rank weight rank 2, got {tuple(weight.shape)}")
    reg_n, reg_k = 1, 2
    pack_n = layout.n_pack_size * layout.num_n_lanes * reg_n
    pack_k = layout.k_pack_size * layout.num_k_lanes * reg_k
    packed = _pad_tensor(weight.contiguous().to(dtype=dtype), divisor=(pack_n, pack_k), dim=(0, 1), fill_value=0)
    if down:
        r, c = int(packed.shape[0]), int(packed.shape[1])
        r_packs, c_packs = r // pack_n, c // pack_k
        packed = packed.view(r_packs, pack_n, c_packs, pack_k).permute(2, 0, 1, 3)
    else:
        c, r = int(packed.shape[0]), int(packed.shape[1])
        c_packs, r_packs = c // pack_n, r // pack_k
        packed = packed.view(c_packs, pack_n, r_packs, pack_k).permute(0, 2, 1, 3)
    packed = packed.reshape(
        c_packs,
        r_packs,
        layout.n_pack_size,
        layout.num_n_lanes,
        reg_n,
        layout.k_pack_size,
        layout.num_k_lanes,
        reg_k,
    )
    packed = packed.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
    return packed.view(c, r).contiguous()


def _to_nunchaku_runtime_packed_payload(payload: NunchakuSvdquantFixturePayload, *, dtype: Any) -> NunchakuSvdquantFixturePayload:
    proj_down_rank_in = payload.proj_down.t().contiguous()
    packed = NunchakuSvdquantFixturePayload(
        layer_prefix=payload.layer_prefix,
        qweight=_pack_nunchaku_weight_from_pair_bytes(payload.qweight),
        wscales=_pack_nunchaku_weight_scale(payload.wscales, dtype=dtype),
        smooth_factor=_pack_nunchaku_scale_vector(payload.smooth_factor, dtype=dtype),
        proj_down=_pack_nunchaku_lowrank_weight(proj_down_rank_in, down=True, dtype=dtype),
        proj_up=_pack_nunchaku_lowrank_weight(payload.proj_up, down=False, dtype=dtype),
        bias=None if payload.bias is None else _pack_nunchaku_scale_vector(payload.bias, dtype=dtype),
        inputs=payload.inputs,
        quant_config=payload.quant_config,
        activation_q_values=payload.activation_q_values,
        activation_scale=payload.activation_scale,
    )
    _validate_payload_shapes(packed)
    return packed


def _bootstrap_nunchaku_source(nunchaku_source: str | Path) -> None:
    """Load only the external package fragments needed by this dev harness."""

    root = Path(nunchaku_source).expanduser()
    package_dir = root / "nunchaku"
    if not package_dir.is_dir():
        raise HarnessError(f"Nunchaku source root {str(root)!r} does not contain a nunchaku package directory")

    extension_candidates = sorted(package_dir.glob("_C*.so"))
    if not extension_candidates:
        raise HarnessError(
            f"Nunchaku extension was not found under {str(package_dir)!r}; build it with `python setup.py build_ext --inplace`"
        )
    extension_path = extension_candidates[-1]

    for module_name in list(sys.modules):
        if module_name == "nunchaku" or module_name.startswith("nunchaku."):
            del sys.modules[module_name]

    package = types.ModuleType("nunchaku")
    package.__file__ = str(package_dir / "__init__.py")
    package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    package.__package__ = "nunchaku"
    sys.modules["nunchaku"] = package

    for subpackage_name in ("ops", "models"):
        subpackage = types.ModuleType(f"nunchaku.{subpackage_name}")
        subpackage.__file__ = str(package_dir / subpackage_name / "__init__.py")
        subpackage.__path__ = [str(package_dir / subpackage_name)]  # type: ignore[attr-defined]
        subpackage.__package__ = f"nunchaku.{subpackage_name}"
        sys.modules[f"nunchaku.{subpackage_name}"] = subpackage
        setattr(package, subpackage_name, subpackage)

    spec = importlib.util.spec_from_file_location("nunchaku._C", extension_path)
    if spec is None or spec.loader is None:
        raise HarnessError(f"failed to create import spec for Nunchaku extension {str(extension_path)!r}")
    extension_module = importlib.util.module_from_spec(spec)
    sys.modules["nunchaku._C"] = extension_module
    spec.loader.exec_module(extension_module)
    setattr(package, "_C", extension_module)


def _import_nunchaku_model(nunchaku_source: str | Path | None):
    if nunchaku_source is not None:
        _bootstrap_nunchaku_source(nunchaku_source)
    try:
        linear_module = importlib.import_module("nunchaku.models.linear")
        SVDQW4A4Linear = linear_module.SVDQW4A4Linear
    except Exception as exc:  # noqa: BLE001 - external dev dependency failures should be reported cleanly
        raise HarnessError(
            "failed to import nunchaku.models.linear.SVDQW4A4Linear. "
            "Install/build Nunchaku in a separate dev environment or pass --nunchaku-source."
        ) from exc
    return SVDQW4A4Linear


def _prepare_nunchaku_layer(
    payload: NunchakuSvdquantFixturePayload,
    *,
    nunchaku_source: str | Path | None,
    dtype_name: str,
    device_name: str,
    assignment_layout: str,
):
    torch = _load_torch()
    dtype = _dtype_from_name(dtype_name)
    device = _resolve_device(device_name)
    SVDQW4A4Linear = _import_nunchaku_model(nunchaku_source)
    if assignment_layout == "natural":
        runtime_payload = payload
    elif assignment_layout == "nunchaku-packed":
        runtime_payload = _to_nunchaku_runtime_packed_payload(payload, dtype=dtype)
    else:
        raise HarnessError(f"unsupported assignment layout {assignment_layout!r}")
    act_unsigned = bool(payload.quant_config.get("act_unsigned", False))
    layer = SVDQW4A4Linear(
        in_features=payload.in_features,
        out_features=payload.out_features,
        rank=payload.rank,
        bias=payload.bias is not None,
        precision="int4",
        act_unsigned=act_unsigned,
        torch_dtype=dtype,
        device=device,
    )
    layer.eval()
    with torch.no_grad():
        layer.qweight.copy_(runtime_payload.qweight.to(device=device))
        layer.wscales.copy_(runtime_payload.wscales.to(device=device, dtype=dtype))
        layer.smooth_factor.copy_(runtime_payload.smooth_factor.to(device=device, dtype=dtype))
        layer.smooth_factor_orig.copy_(runtime_payload.smooth_factor.to(device=device, dtype=dtype))
        layer.proj_down.copy_(runtime_payload.proj_down.to(device=device, dtype=dtype))
        layer.proj_up.copy_(runtime_payload.proj_up.to(device=device, dtype=dtype))
        if runtime_payload.bias is not None and layer.bias is not None:
            layer.bias.copy_(runtime_payload.bias.to(device=device, dtype=dtype))
    return torch, dtype, device, layer


def _run_nunchaku_forward(
    payload: NunchakuSvdquantFixturePayload,
    *,
    nunchaku_source: str | Path | None,
    dtype_name: str,
    device_name: str,
    assignment_layout: str,
):
    torch, dtype, device, layer = _prepare_nunchaku_layer(
        payload,
        nunchaku_source=nunchaku_source,
        dtype_name=dtype_name,
        device_name=device_name,
        assignment_layout=assignment_layout,
    )
    with torch.no_grad():
        x = payload.inputs.reshape(1, payload.batch_rows, payload.in_features).to(device=device, dtype=dtype)
        output = layer(x).reshape(payload.batch_rows, payload.out_features).detach().float().cpu().contiguous()
    return output, str(device)


def _tensor_error_summary(actual: Any, expected: Any) -> dict[str, Any]:
    torch = _load_torch()
    diff = actual.detach().float() - expected.detach().float()
    abs_diff = diff.abs()
    expected_abs = expected.detach().float().abs()
    relative = abs_diff / torch.clamp(expected_abs, min=1e-12)
    return {
        "max_abs_error": float(abs_diff.max().item()) if int(abs_diff.numel()) else 0.0,
        "mean_abs_error": float(abs_diff.mean().item()) if int(abs_diff.numel()) else 0.0,
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()) if int(diff.numel()) else 0.0,
        "max_relative_error": float(relative.max().item()) if int(relative.numel()) else 0.0,
    }


def _unpack_nunchaku_activation_q(quantized_x: Any, *, in_features: int, signed: bool = True):
    torch = _load_torch()
    if int(in_features) <= 0 or int(in_features) % KITCHEN_GROUP_SIZE != 0:
        raise HarnessError(f"in_features={in_features} must be a positive multiple of {KITCHEN_GROUP_SIZE}")
    q32 = quantized_x.detach().cpu().contiguous().view(torch.int32).view(-1, 4)
    k_groups = int(in_features) // KITCHEN_GROUP_SIZE
    records_per_page_group = 8 * 2 * 32
    if int(q32.shape[0]) % records_per_page_group != 0:
        raise HarnessError(
            f"packed activation has {int(q32.shape[0])} int32 records, which is not divisible by "
            f"{records_per_page_group}"
        )
    page_groups = int(q32.shape[0]) // records_per_page_group
    if page_groups % k_groups != 0:
        raise HarnessError(f"packed activation page groups={page_groups} is not divisible by k_groups={k_groups}")
    m_blocks = page_groups // k_groups
    decoded = torch.zeros((m_blocks * 256, int(in_features)), dtype=torch.int8)
    for page_group in range(page_groups):
        bm = page_group // k_groups
        kg = page_group % k_groups
        for warp in range(8):
            for tile in range(2):
                for lane in range(32):
                    index = ((page_group * 8 + warp) * 2 + tile) * 32 + lane
                    fields = [int(q32[index, field].item()) & 0xFFFFFFFF for field in range(4)]
                    row_base = bm * 256 + warp * 32 + tile * 16 + lane // 4
                    lane_mod = lane % 4
                    destinations = (
                        (row_base, lane_mod, fields[0]),
                        (row_base + 8, lane_mod, fields[1]),
                        (row_base, 4 + lane_mod, fields[2]),
                        (row_base + 8, 4 + lane_mod, fields[3]),
                    )
                    for row, cpack, packed in destinations:
                        base_col = kg * KITCHEN_GROUP_SIZE + cpack * 8
                        for nibble in range(8):
                            value = (packed >> (4 * nibble)) & 0xF
                            if signed and value >= 8:
                                value -= 16
                            decoded[row, base_col + nibble] = value
    return decoded.contiguous()


def _unpack_nunchaku_activation_scales(ascales: Any, *, in_features: int, padded_rows: int):
    torch = _load_torch()
    if int(in_features) <= 0 or int(in_features) % KITCHEN_GROUP_SIZE != 0:
        raise HarnessError(f"in_features={in_features} must be a positive multiple of {KITCHEN_GROUP_SIZE}")
    if int(padded_rows) <= 0:
        raise HarnessError(f"padded_rows={padded_rows} must be positive")
    k_groups = int(in_features) // KITCHEN_GROUP_SIZE
    m_blocks = _ceil_divide(int(padded_rows), 256)
    flat = ascales.detach().cpu().contiguous().view(-1)
    required_values = m_blocks * k_groups * 8 * 16 * 2
    if int(flat.numel()) < required_values:
        raise HarnessError(
            f"packed activation scales contain {int(flat.numel())} values, but {required_values} are required"
        )
    decoded = torch.zeros((m_blocks * 256, k_groups), dtype=flat.dtype)
    for page_group in range(m_blocks * k_groups):
        bm = page_group // k_groups
        kg = page_group % k_groups
        for warp in range(8):
            for lane in range(16):
                offset = ((page_group * 8 + warp) * 16 + lane) * 2
                row0 = bm * 256 + warp * 32 + (lane // 8) * 16 + (lane % 8)
                decoded[row0, kg] = flat[offset]
                decoded[row0 + 8, kg] = flat[offset + 1]
    return decoded[: int(padded_rows)].contiguous()


def _activation_decode_diagnostics(
    payload: NunchakuSvdquantFixturePayload,
    *,
    decoded_q: Any,
    decoded_scales: Any,
    signed_decode: bool,
) -> dict[str, Any]:
    torch = _load_torch()
    result: dict[str, Any] = {
        "scope": "external_runtime_activation_decode",
        "decoded_q_shape": [int(x) for x in decoded_q.shape],
        "decoded_scale_shape": [int(x) for x in decoded_scales.shape],
        "signed_decode": bool(signed_decode),
        "unpadded_rows_compared": int(payload.batch_rows),
        "note": (
            "Decoded q/scale tensors are compared with the repo fixture oracle when those tensors are present; "
            "small differences can come from external CUDA/BF16 rounding."
        ),
    }
    if payload.activation_q_values is not None:
        expected_q = payload.activation_q_values.reshape(payload.batch_rows, payload.in_features).detach().cpu()
        actual_q = decoded_q[: payload.batch_rows, : payload.in_features].to(dtype=expected_q.dtype)
        q_summary = _tensor_error_summary(actual_q, expected_q)
        q_summary["mismatch_count"] = int(torch.count_nonzero(actual_q != expected_q).item())
        q_summary["numel"] = int(expected_q.numel())
        result["q_vs_fixture_q_values"] = q_summary
    else:
        result["q_vs_fixture_q_values"] = None

    if payload.activation_scale is not None:
        expected_scale = payload.activation_scale.reshape(payload.batch_rows, -1).detach().cpu()
        actual_scale = decoded_scales[: payload.batch_rows, : int(expected_scale.shape[1])].to(dtype=expected_scale.dtype)
        scale_summary = _tensor_error_summary(actual_scale, expected_scale)
        scale_summary["numel"] = int(expected_scale.numel())
        result["scale_vs_fixture_activation_scale"] = scale_summary
    else:
        result["scale_vs_fixture_activation_scale"] = None
    return result


def _decoded_activation_main_replay_diagnostics(
    payload: NunchakuSvdquantFixturePayload,
    *,
    runtime_main: Any,
    decoded_q: Any,
    decoded_scales: Any,
    dtype: Any,
) -> dict[str, Any]:
    torch = _load_torch()
    rows = payload.batch_rows
    groups = payload.in_features // KITCHEN_GROUP_SIZE
    q = decoded_q[:rows, : payload.in_features].detach().float()
    scales = decoded_scales[:rows, :groups].detach().float()
    decoded_activation = (q.reshape(rows, groups, KITCHEN_GROUP_SIZE) * scales.unsqueeze(-1)).reshape(
        rows, payload.in_features
    )

    fp32_weight = dequantize_svdquant_w4a4_effective_weight(
        payload.qweight,
        payload.wscales,
        group_size=KITCHEN_GROUP_SIZE,
    ).detach().float()
    runtime_dtype_weight_scale = dequantize_svdquant_w4a4_effective_weight(
        payload.qweight,
        payload.wscales.to(dtype=dtype),
        group_size=KITCHEN_GROUP_SIZE,
    ).detach().float()
    fp32_replay = torch.matmul(decoded_activation, fp32_weight.t()).contiguous()
    runtime_dtype_replay = torch.matmul(decoded_activation, runtime_dtype_weight_scale.t()).contiguous()
    runtime_like_replay = _decoded_activation_group_fma_main_replay(
        qweight=payload.qweight,
        wscales=payload.wscales,
        decoded_q=decoded_q,
        decoded_scales=decoded_scales,
        rows=rows,
        in_features=payload.in_features,
        out_features=payload.out_features,
        dtype=dtype,
    )
    return {
        "scope": "decoded_activation_dense_main_replay",
        "note": (
            "Dense replay uses decoded external q/scale activations and natural dequantized weights. "
            "The group-fma replay applies per-group scale and accumulator rounding in the runtime dtype."
        ),
        "main_vs_decoded_activation_dense_fp32_weight_scale": _tensor_error_summary(runtime_main, fp32_replay),
        "main_vs_decoded_activation_dense_runtime_dtype_weight_scale": _tensor_error_summary(
            runtime_main,
            runtime_dtype_replay,
        ),
        "main_vs_decoded_activation_group_dtype_fma_runtime_like": _tensor_error_summary(
            runtime_main,
            runtime_like_replay,
        ),
    }


def _decoded_activation_group_fma_main_replay(
    *,
    qweight: Any,
    wscales: Any,
    decoded_q: Any,
    decoded_scales: Any,
    rows: int,
    in_features: int,
    out_features: int,
    dtype: Any,
):
    torch = _load_torch()
    if int(in_features) % KITCHEN_GROUP_SIZE != 0:
        raise HarnessError(f"in_features={in_features} is not divisible by group size {KITCHEN_GROUP_SIZE}")
    groups = int(in_features) // KITCHEN_GROUP_SIZE
    q = decoded_q[: int(rows), : int(in_features)].detach().to(dtype=torch.int32)
    weight = unpack_signed_int4_pairs(qweight).detach().to(dtype=torch.int32)
    if tuple(int(x) for x in weight.shape) != (int(out_features), int(in_features)):
        raise HarnessError(
            f"decoded qweight shape {tuple(weight.shape)} does not match expected {(int(out_features), int(in_features))}"
        )
    if tuple(int(x) for x in wscales.shape) != (groups, int(out_features)):
        raise HarnessError(f"wscales shape {tuple(wscales.shape)} does not match expected {(groups, int(out_features))}")
    scale = decoded_scales[: int(rows), :groups].detach().to(dtype=dtype)
    weight_scale = wscales.detach().to(dtype=dtype)
    out = torch.zeros((int(rows), int(out_features)), dtype=dtype)
    for group in range(groups):
        start = group * KITCHEN_GROUP_SIZE
        stop = start + KITCHEN_GROUP_SIZE
        intacc = torch.matmul(q[:, start:stop], weight[:, start:stop].t())
        product = (scale[:, group].unsqueeze(1) * weight_scale[group].unsqueeze(0)).to(dtype=dtype)
        out = (intacc.to(dtype=dtype).float() * product.float() + out.float()).to(dtype=dtype)
    return out.contiguous()


def _lowrank_activation_runtime_like_replay(payload: NunchakuSvdquantFixturePayload, *, dtype: Any):
    torch = _load_torch()
    raw_inputs = payload.inputs.reshape(payload.batch_rows, payload.in_features).detach().to(dtype=dtype).float()
    proj_down = payload.proj_down.detach().to(dtype=dtype).float()
    return torch.matmul(raw_inputs, proj_down)


def _lowrank_runtime_like_replay(payload: NunchakuSvdquantFixturePayload, *, dtype: Any):
    torch = _load_torch()
    lora_act = _lowrank_activation_runtime_like_replay(payload, dtype=dtype)
    proj_up = payload.proj_up.detach().to(dtype=dtype).float()
    return torch.matmul(lora_act.to(dtype=dtype).float(), proj_up.t()).to(dtype=dtype).contiguous()


def _bias_runtime_like_replay(payload: NunchakuSvdquantFixturePayload, *, dtype: Any):
    torch = _load_torch()
    if payload.bias is None:
        return torch.zeros((payload.batch_rows, payload.out_features), dtype=dtype)
    return payload.bias.detach().to(dtype=dtype).reshape(1, payload.out_features).expand(
        payload.batch_rows,
        payload.out_features,
    ).contiguous()


def _full_runtime_like_replay(
    payload: NunchakuSvdquantFixturePayload,
    *,
    decoded_q: Any,
    decoded_scales: Any,
    dtype: Any,
):
    main = _decoded_activation_group_fma_main_replay(
        qweight=payload.qweight,
        wscales=payload.wscales,
        decoded_q=decoded_q,
        decoded_scales=decoded_scales,
        rows=payload.batch_rows,
        in_features=payload.in_features,
        out_features=payload.out_features,
        dtype=dtype,
    )
    bias = _bias_runtime_like_replay(payload, dtype=dtype)
    main_bias = (main.float() + bias.float()).to(dtype=dtype)
    lora_act = _lowrank_activation_runtime_like_replay(payload, dtype=dtype)
    proj_up = payload.proj_up.detach().to(dtype=dtype).float()
    torch = _load_torch()
    lowrank_acc = torch.matmul(lora_act.to(dtype=dtype).float(), proj_up.t())
    return (main_bias.float() + lowrank_acc.float()).to(dtype=dtype).contiguous()


def _activation_quant(payload: NunchakuSvdquantFixturePayload, post_smoothing_inputs: Any):
    if bool(payload.quant_config.get("act_unsigned", False)):
        return quantize_activation_w4_unsigned(post_smoothing_inputs, group_size=KITCHEN_GROUP_SIZE)
    return quantize_activation_w4_signed(post_smoothing_inputs, group_size=KITCHEN_GROUP_SIZE)


def _reference_component_outputs(payload: NunchakuSvdquantFixturePayload) -> dict[str, Any]:
    torch = _load_torch()
    raw_inputs = payload.inputs.reshape(payload.batch_rows, payload.in_features).detach().float()
    smooth = payload.smooth_factor.detach().float().reshape(1, payload.in_features)
    if bool(payload.quant_config.get("act_unsigned", False)):
        main_inputs = raw_inputs + GELU_UNSIGNED_SHIFT
    else:
        main_inputs = raw_inputs
    post_smoothing_inputs = main_inputs / smooth
    activation = _activation_quant(payload, post_smoothing_inputs)
    dense_weight = dequantize_svdquant_w4a4_effective_weight(
        payload.qweight,
        payload.wscales,
        group_size=KITCHEN_GROUP_SIZE,
    ).detach().float()

    main = torch.matmul(activation.dequantized.detach().float(), dense_weight.t()).contiguous()
    lowrank = torch.matmul(torch.matmul(raw_inputs, payload.proj_down.detach().float()), payload.proj_up.detach().float().t())
    lowrank = lowrank.contiguous()
    bias = torch.zeros_like(main)
    if payload.bias is not None:
        bias = payload.bias.detach().float().reshape(1, payload.out_features).expand_as(main).contiguous()
    full = (main + lowrank + bias).contiguous()
    return {"main": main, "lowrank": lowrank, "bias": bias, "full": full}


def _run_nunchaku_variant(
    payload: NunchakuSvdquantFixturePayload,
    *,
    nunchaku_source: str | Path | None,
    dtype_name: str,
    device_name: str,
    assignment_layout: str,
    zero_main: bool = False,
    zero_lowrank: bool = False,
    zero_bias: bool = False,
):
    torch, dtype, device, layer = _prepare_nunchaku_layer(
        payload,
        nunchaku_source=nunchaku_source,
        dtype_name=dtype_name,
        device_name=device_name,
        assignment_layout=assignment_layout,
    )
    with torch.no_grad():
        if zero_main:
            layer.qweight.zero_()
        if zero_lowrank:
            layer.proj_up.zero_()
        if zero_bias and layer.bias is not None:
            layer.bias.zero_()
        x = payload.inputs.reshape(1, payload.batch_rows, payload.in_features).to(device=device, dtype=dtype)
        output = layer(x).reshape(payload.batch_rows, payload.out_features).detach().float().cpu().contiguous()
    return output


def _run_nunchaku_component_diagnostics(
    payload: NunchakuSvdquantFixturePayload,
    *,
    nunchaku_source: str | Path | None,
    dtype_name: str,
    device_name: str,
    assignment_layout: str,
) -> dict[str, Any]:
    torch = _load_torch()
    references = _reference_component_outputs(payload)

    full = _run_nunchaku_variant(
        payload,
        nunchaku_source=nunchaku_source,
        dtype_name=dtype_name,
        device_name=device_name,
        assignment_layout=assignment_layout,
    )
    main = _run_nunchaku_variant(
        payload,
        nunchaku_source=nunchaku_source,
        dtype_name=dtype_name,
        device_name=device_name,
        assignment_layout=assignment_layout,
        zero_lowrank=True,
        zero_bias=True,
    )
    lowrank = _run_nunchaku_variant(
        payload,
        nunchaku_source=nunchaku_source,
        dtype_name=dtype_name,
        device_name=device_name,
        assignment_layout=assignment_layout,
        zero_main=True,
        zero_bias=True,
    )
    bias = _run_nunchaku_variant(
        payload,
        nunchaku_source=nunchaku_source,
        dtype_name=dtype_name,
        device_name=device_name,
        assignment_layout=assignment_layout,
        zero_main=True,
        zero_lowrank=True,
    )

    torch_mod, dtype, device, layer = _prepare_nunchaku_layer(
        payload,
        nunchaku_source=nunchaku_source,
        dtype_name=dtype_name,
        device_name=device_name,
        assignment_layout=assignment_layout,
    )
    with torch_mod.no_grad():
        x2d = payload.inputs.reshape(payload.batch_rows, payload.in_features).to(device=device, dtype=dtype)
        x3d = x2d.reshape(1, payload.batch_rows, payload.in_features)
        forward_output = layer(x3d).reshape(payload.batch_rows, payload.out_features).detach().float().cpu()
        quantized_x, ascales, lora_act = layer.quantize(x2d)
        replay_buffer = torch_mod.empty(payload.batch_rows, payload.out_features, dtype=dtype, device=device)
        replay_output = (
            layer.forward_quant(quantized_x, ascales, lora_act, replay_buffer)
            .reshape(payload.batch_rows, payload.out_features)
            .detach()
            .float()
            .cpu()
        )
        signed_decode = not bool(payload.quant_config.get("act_unsigned", False))
        decoded_q = _unpack_nunchaku_activation_q(
            quantized_x,
            in_features=payload.in_features,
            signed=signed_decode,
        )
        decoded_scales = _unpack_nunchaku_activation_scales(
            ascales,
            in_features=payload.in_features,
            padded_rows=int(decoded_q.shape[0]),
        )
    activation_decode = _activation_decode_diagnostics(
        payload,
        decoded_q=decoded_q,
        decoded_scales=decoded_scales,
        signed_decode=signed_decode,
    )
    dense_main_replay = _decoded_activation_main_replay_diagnostics(
        payload,
        runtime_main=main,
        decoded_q=decoded_q,
        decoded_scales=decoded_scales,
        dtype=dtype,
    )
    lowrank_replay = _lowrank_runtime_like_replay(payload, dtype=dtype)
    bias_replay = _bias_runtime_like_replay(payload, dtype=dtype)
    full_replay = _full_runtime_like_replay(
        payload,
        decoded_q=decoded_q,
        decoded_scales=decoded_scales,
        dtype=dtype,
    )

    return {
        "scope": "single_layer_runtime_segment_diagnostic",
        "expected_component_oracle": "repo_fp32_natural_layout_runtime_like_reference",
        "activation_layout_note": "external quantized activation and activation-scale tensors are runtime-packed, not natural pair-byte layout",
        "quantized_activation_shape": [int(x) for x in quantized_x.shape],
        "activation_scale_shape": [int(x) for x in ascales.shape],
        "lowrank_activation_shape": [int(x) for x in lora_act.shape],
        "activation_decode": activation_decode,
        "dense_main_replay": dense_main_replay,
        "lowrank_runtime_like_replay": {
            "scope": "natural_runtime_dtype_lowrank_replay",
            "lowrank_vs_natural_runtime_dtype_down_up": _tensor_error_summary(lowrank, lowrank_replay),
        },
        "bias_runtime_like_replay": {
            "scope": "runtime_dtype_bias_broadcast",
            "bias_vs_runtime_dtype_bias_broadcast": _tensor_error_summary(bias, bias_replay),
        },
        "full_runtime_like_replay": {
            "scope": "decoded_activation_main_bias_lowrank_runtime_dtype_epilogue",
            "full_vs_decoded_main_bias_lowrank_runtime_dtype_epilogue": _tensor_error_summary(full, full_replay),
        },
        "forward_vs_quantize_forward_quant": _tensor_error_summary(replay_output, forward_output),
        "full_vs_reference": _tensor_error_summary(full, references["full"]),
        "main_vs_reference": _tensor_error_summary(main, references["main"]),
        "lowrank_vs_reference": _tensor_error_summary(lowrank, references["lowrank"]),
        "bias_vs_reference": _tensor_error_summary(bias, references["bias"]),
    }


def _build_report(
    payload: NunchakuSvdquantFixturePayload,
    *,
    fixture: str | Path,
    output: str | Path | None,
    runtime_output_tensor: str,
    status: str,
    external_runtime_validation: str,
    dtype_name: str,
    device_name: str,
    assignment_layout: str,
    notes: list[str],
    component_diagnostics: dict[str, Any] | None = None,
) -> HarnessReport:
    return HarnessReport(
        status=status,
        fixture=str(Path(fixture).expanduser()),
        output=None if output is None else str(Path(output).expanduser()),
        runtime_output_tensor=runtime_output_tensor,
        assignment_layout=assignment_layout,
        layer_prefix=payload.layer_prefix,
        external_runtime="nunchaku.SVDQW4A4Linear",
        validation_scope="single_layer_svdquant_w4a4_linear_forward",
        external_runtime_validation=external_runtime_validation,
        publishable_svdquant_gptq=False,
        in_features=payload.in_features,
        out_features=payload.out_features,
        rank=payload.rank,
        group_size=KITCHEN_GROUP_SIZE,
        batch_rows=payload.batch_rows,
        dtype=dtype_name,
        device=device_name,
        activation_signedness="unsigned" if bool(payload.quant_config.get("act_unsigned", False)) else "signed",
        lowrank_branch_input_basis=str(payload.quant_config.get("lowrank_branch_input_basis", "")),
        proj_down_smooth_folded=payload.quant_config.get("proj_down_smooth_folded")
        if isinstance(payload.quant_config.get("proj_down_smooth_folded"), bool)
        else None,
        notes=notes,
        component_diagnostics=component_diagnostics,
    )


def run(args: argparse.Namespace) -> int:
    try:
        payload = _load_payload(
            args.fixture,
            layer_prefix=args.layer_prefix,
            input_tensor=args.input_tensor,
            require_raw_branch=not args.allow_post_smoothing_branch,
            require_rank_multiple_of_16=not args.allow_non_multiple_of_16_rank,
        )
        if args.dry_run_layout:
            report = _build_report(
                payload,
                fixture=args.fixture,
                output=None,
                runtime_output_tensor=args.runtime_output_tensor,
                status="layout_ready_external_runtime_not_executed",
                external_runtime_validation="not_run",
                dtype_name=args.dtype,
                device_name=args.device,
                assignment_layout=args.assignment_layout,
                notes=[
                    "Dry-run layout check only; no external fused runtime was imported or executed.",
                    "Use validate-runtime-fixture-output only after a real runtime.output safetensors is written.",
                ],
            )
            if args.report:
                write_json(args.report, report.to_dict())
            if args.json:
                print(_json_dump(report.to_dict()))
            return 0

        output, resolved_device = _run_nunchaku_forward(
            payload,
            nunchaku_source=args.nunchaku_source,
            dtype_name=args.dtype,
            device_name=args.device,
            assignment_layout=args.assignment_layout,
        )
        component_diagnostics = None
        if args.diagnose_components:
            component_diagnostics = _run_nunchaku_component_diagnostics(
                payload,
                nunchaku_source=args.nunchaku_source,
                dtype_name=args.dtype,
                device_name=resolved_device,
                assignment_layout=args.assignment_layout,
            )
        output_path = Path(args.out).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file({args.runtime_output_tensor: output}, str(output_path))
        report = _build_report(
            payload,
            fixture=args.fixture,
            output=output_path,
            runtime_output_tensor=args.runtime_output_tensor,
            status="runtime_output_written",
            external_runtime_validation="not_validated_by_this_script",
            dtype_name=args.dtype,
            device_name=resolved_device,
            assignment_layout=args.assignment_layout,
            notes=[
                "This script only writes external runtime output.",
                "Run comfy-quants validate-runtime-fixture-output to compare it with fixture.expected_output.",
            ],
            component_diagnostics=component_diagnostics,
        )
        if args.report:
            write_json(args.report, report.to_dict())
        if args.json:
            print(_json_dump(report.to_dict()))
        return 0
    except HarnessError as exc:
        if args.json:
            print(_json_dump({"status": "failed", "error": str(exc), "publishable_svdquant_gptq": False}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a SVDQuant W4A4 runtime fixture through an optional Nunchaku fused layer.",
    )
    parser.add_argument("--fixture", required=True, help="Path to svdquant_w4a4_runtime_fixture.safetensors")
    parser.add_argument("--out", help="Output safetensors path containing runtime.output")
    parser.add_argument("--report", help="Optional JSON report path for this dev harness")
    parser.add_argument("--layer-prefix", help="Fixture layer prefix; inferred when exactly one SVDQuant layer exists")
    parser.add_argument("--input-tensor", default=DEFAULT_EXPECTED_INPUT_TENSOR, help="Input tensor name inside the fixture")
    parser.add_argument(
        "--runtime-output-tensor",
        default=DEFAULT_RUNTIME_OUTPUT_TENSOR,
        help="Tensor name to write into the output safetensors",
    )
    parser.add_argument(
        "--nunchaku-source",
        help="Optional source checkout root to prepend to sys.path; use only in a separate dev environment",
    )
    parser.add_argument("--device", default="auto", help="CUDA device for external runtime execution; auto uses cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"), help="External layer parameter/input dtype")
    parser.add_argument(
        "--assignment-layout",
        default="nunchaku-packed",
        choices=("nunchaku-packed", "natural"),
        help=(
            "How fixture tensors are assigned into the external layer. "
            "'nunchaku-packed' applies the dev-only MMA runtime layout bridge; "
            "'natural' preserves the previous direct assignment for diagnosis."
        ),
    )
    parser.add_argument(
        "--dry-run-layout",
        action="store_true",
        help="Only unpack and validate fixture layout; do not import or execute Nunchaku",
    )
    parser.add_argument(
        "--diagnose-components",
        action="store_true",
        help=(
            "After writing runtime output, run extra dev-only single-layer segment checks "
            "(forward replay, main-only, lowrank-only, bias-only) and include them in the report."
        ),
    )
    parser.add_argument(
        "--allow-post-smoothing-branch",
        action="store_true",
        help="Allow fixtures whose low-rank branch is not raw-basis; this is expected to mismatch Nunchaku-style fused kernels",
    )
    parser.add_argument(
        "--allow-non-multiple-of-16-rank",
        action="store_true",
        help="Allow ranks that are not multiples of 16; direct fused kernels commonly reject them",
    )
    parser.add_argument("--json", action="store_true", help="Print a compact JSON report to stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run_layout and not args.out:
        parser.error("--out is required unless --dry-run-layout is set")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
