"""Layer selection for Qwen-Image-Edit SVDQuant W4A4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Any

from comfy_quants.model_adapters.qwen_image_edit_int4 import iter_svdquant_linear_mappings


@dataclass(frozen=True)
class Int4LinearSelection:
    """One source linear selected for SVDQuant conversion."""

    output_prefix: str
    source_prefix: str
    smooth_lookup_suffix: str
    branch_lookup_suffix: str
    act_unsigned: bool
    has_bias: bool
    shape: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.shape is not None:
            data["shape"] = list(self.shape)
        return data


def select_qwen_image_edit_svdquant_linears(keys: Iterable[str]) -> list[Int4LinearSelection]:
    """Return Qwen-Image-Edit linear prefixes available in a checkpoint."""
    from comfy_quants.model_adapters.qwen_image_edit_int4 import is_act_unsigned_prefix

    keyset = set(keys)
    selected: list[Int4LinearSelection] = []
    for output_prefix, source_prefix, spec in iter_svdquant_linear_mappings(keyset):
        selected.append(
            Int4LinearSelection(
                output_prefix=output_prefix,
                source_prefix=source_prefix,
                smooth_lookup_suffix=spec.smooth_lookup_suffix(),
                branch_lookup_suffix=spec.branch_lookup_suffix(),
                act_unsigned=is_act_unsigned_prefix(output_prefix),
                has_bias=f"{source_prefix}.bias" in keyset,
            )
        )
    return selected


def transformer_block_prefix(output_prefix: str) -> str | None:
    """Return ``transformer_blocks.<index>`` for a selected layer prefix."""
    parts = output_prefix.split(".")
    if len(parts) >= 2 and parts[0] == "transformer_blocks":
        return ".".join(parts[:2])
    return None


def activation_stats_lookup_candidates(item: Int4LinearSelection) -> list[str]:
    """Return accepted activation-stats keys for one selected linear."""
    candidates: list[str] = []
    for name in (item.output_prefix, item.source_prefix):
        if name not in candidates:
            candidates.append(name)
    block = transformer_block_prefix(item.output_prefix)
    if block is not None:
        for suffix in (item.smooth_lookup_suffix, item.branch_lookup_suffix):
            name = f"{block}.{suffix}"
            if name not in candidates:
                candidates.append(name)
    return candidates
