"""Qwen-Image-Edit INT4 static mapping helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QwenImageEditInt4LinearSpec:
    """Static source-to-output mapping for one SVDQuant linear family."""

    output_suffix: str
    candidate_suffixes: tuple[str, ...]
    smooth_suffix: str | None = None
    branch_suffix: str | None = None

    def smooth_lookup_suffix(self) -> str:
        return self.output_suffix if self.smooth_suffix is None else self.smooth_suffix

    def branch_lookup_suffix(self) -> str:
        return self.output_suffix if self.branch_suffix is None else self.branch_suffix


@dataclass(frozen=True)
class GroupedQKVBranchSpec:
    """Low-rank branch anchor whose output rows cover split Q/K/V targets."""

    anchor_suffix: str
    target_suffixes: tuple[str, str, str]


SVDQUANT_LINEAR_SPECS: tuple[QwenImageEditInt4LinearSpec, ...] = (
    QwenImageEditInt4LinearSpec("attn.to_q", ("attn.to_q",)),
    QwenImageEditInt4LinearSpec("attn.to_k", ("attn.to_k",)),
    QwenImageEditInt4LinearSpec("attn.to_v", ("attn.to_v",)),
    QwenImageEditInt4LinearSpec("attn.add_q_proj", ("attn.add_q_proj",)),
    QwenImageEditInt4LinearSpec("attn.add_k_proj", ("attn.add_k_proj",)),
    QwenImageEditInt4LinearSpec("attn.add_v_proj", ("attn.add_v_proj",)),
    QwenImageEditInt4LinearSpec("attn.to_out.0", ("attn.to_out.0",)),
    QwenImageEditInt4LinearSpec("attn.to_add_out", ("attn.to_add_out",), smooth_suffix="attn.to_out.0"),
    QwenImageEditInt4LinearSpec("img_mlp.net.0.proj", ("img_mlp.net.0.proj",)),
    QwenImageEditInt4LinearSpec("img_mlp.net.2", ("img_mlp.net.2.linear", "img_mlp.net.2")),
    QwenImageEditInt4LinearSpec("txt_mlp.net.0.proj", ("txt_mlp.net.0.proj",)),
    QwenImageEditInt4LinearSpec("txt_mlp.net.2", ("txt_mlp.net.2.linear", "txt_mlp.net.2")),
)

GROUPED_QKV_BRANCH_SPECS: tuple[GroupedQKVBranchSpec, ...] = (
    GroupedQKVBranchSpec("attn.to_q", ("attn.to_q", "attn.to_k", "attn.to_v")),
    GroupedQKVBranchSpec("attn.add_k_proj", ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")),
)

AWQ_MODULATION_SUFFIXES = (".img_mod.1", ".txt_mod.1")
ACT_UNSIGNED_SUFFIXES = (".img_mlp.net.2", ".txt_mlp.net.2", ".ff.net.2")
QKV_SPLIT_TARGETS: Mapping[str, tuple[str, str, str]] = {
    "attn.to_qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "attn.add_qkv_proj": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
}

_BLOCK_PREFIX_RE = re.compile(r"^(transformer_blocks\.\d+)\.")


def is_awq_modulation_prefix(prefix: str) -> bool:
    """Return whether a prefix is a Qwen modulation linear for AWQ W4A16."""
    return any(prefix.endswith(suffix) for suffix in AWQ_MODULATION_SUFFIXES)


def is_act_unsigned_prefix(prefix: str) -> bool:
    """Return whether a Qwen SVDQuant layer uses unsigned activations."""
    return any(prefix.endswith(suffix) for suffix in ACT_UNSIGNED_SUFFIXES)


def qkv_split_prefixes(prefix: str) -> tuple[str, str, str] | None:
    """Return split output prefixes for a fused QKV prefix, if applicable."""
    for suffix, targets in QKV_SPLIT_TARGETS.items():
        if prefix.endswith(suffix):
            stem = prefix[: -len(suffix)]
            return tuple(f"{stem}{target}" for target in targets)
    return None


def equal_qkv_split_sizes(total_out_features: int) -> tuple[int, int, int]:
    """Return equal Q/K/V split sizes along the output-feature axis."""
    total = int(total_out_features)
    if total % 3 != 0:
        raise ValueError(f"fused QKV output features must be divisible by 3, got {total}")
    chunk = total // 3
    return (chunk, chunk, chunk)


def transformer_block_prefixes(keys: Iterable[str]) -> list[str]:
    """Return sorted ``transformer_blocks.<index>`` prefixes present in keys."""
    blocks: set[str] = set()
    for key in keys:
        match = _BLOCK_PREFIX_RE.match(key)
        if match is not None:
            blocks.add(match.group(1))
    return sorted(blocks, key=lambda item: int(item.rsplit(".", 1)[1]))


def iter_svdquant_linear_mappings(keys: Iterable[str]) -> list[tuple[str, str, QwenImageEditInt4LinearSpec]]:
    """Return ``(output_prefix, source_prefix, spec)`` entries available in keys."""
    keyset = set(keys)
    mappings: list[tuple[str, str, QwenImageEditInt4LinearSpec]] = []
    for block_prefix in transformer_block_prefixes(keyset):
        for spec in SVDQUANT_LINEAR_SPECS:
            output_prefix = f"{block_prefix}.{spec.output_suffix}"
            for candidate_suffix in spec.candidate_suffixes:
                source_prefix = f"{block_prefix}.{candidate_suffix}"
                if f"{source_prefix}.weight" in keyset:
                    mappings.append((output_prefix, source_prefix, spec))
                    break
    return mappings


def iter_awq_modulation_prefixes(keys: Iterable[str]) -> list[str]:
    """Return Qwen modulation linear prefixes available for AWQ W4A16."""
    keyset = set(keys)
    prefixes: list[str] = []
    for block_prefix in transformer_block_prefixes(keyset):
        for suffix in AWQ_MODULATION_SUFFIXES:
            prefix = f"{block_prefix}{suffix}"
            if f"{prefix}.weight" in keyset:
                prefixes.append(prefix)
    return prefixes


def split_natural_svdquant_params(
    params: Mapping[str, Any],
    split_sizes: Sequence[int],
) -> list[dict[str, Any]]:
    """Split a natural SVDQuant parameter set along output features.

    ``weight``, ``proj_up`` and ``bias`` split along dimension 0.  ``weight_scale``
    stores its output-feature axis in dimension 1, so it splits along that axis.
    ``smooth_factor`` and ``proj_down`` are input-feature side tensors and are
    shared by each split.
    """

    sizes = tuple(int(size) for size in split_sizes)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"split sizes must be positive, got {tuple(split_sizes)}")
    total = sum(sizes)
    if int(params["weight"].shape[0]) != total:
        raise ValueError(f"weight output dimension {int(params['weight'].shape[0])} does not match split total {total}")
    if int(params["weight_scale"].shape[1]) != total:
        raise ValueError(
            f"weight_scale output dimension {int(params['weight_scale'].shape[1])} does not match split total {total}"
        )
    if int(params["proj_up"].shape[0]) != total:
        raise ValueError(f"proj_up output dimension {int(params['proj_up'].shape[0])} does not match split total {total}")

    weight_parts = params["weight"].split(sizes, dim=0)
    scale_parts = params["weight_scale"].split(sizes, dim=1)
    up_parts = params["proj_up"].split(sizes, dim=0)
    bias_parts = params["bias"].split(sizes, dim=0) if "bias" in params else None
    out: list[dict[str, Any]] = []
    for index, (weight, weight_scale, proj_up) in enumerate(zip(weight_parts, scale_parts, up_parts, strict=True)):
        item = {
            "weight": weight.contiguous(),
            "weight_scale": weight_scale.contiguous(),
            "smooth_factor": params["smooth_factor"].contiguous(),
            "proj_down": params["proj_down"].contiguous(),
            "proj_up": proj_up.contiguous(),
        }
        if bias_parts is not None:
            item["bias"] = bias_parts[index].contiguous()
        if "comfy_quant" in params:
            item["comfy_quant"] = params["comfy_quant"].contiguous()
        out.append(item)
    return out
