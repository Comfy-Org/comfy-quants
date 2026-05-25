"""Runtime-independent activation reduction helpers for INT4 SVDQuant."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from comfy_quants.algorithms.int4_svdquant.stats import ActivationStats, activation_amax_from_samples, merge_activation_stats


def _require_safe_open():
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - depends on optional package dependency
        raise ImportError("safetensors is required for activation sample reduction") from exc
    return safe_open


@dataclass(frozen=True)
class ActivationSampleRef:
    """Reference to one captured activation tensor stored outside the model runtime."""

    layer_name: str
    file_path: str
    tensor_name: str = "activation"
    channel_dim: int = -1
    sample_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActivationStatsAccumulator:
    """Accumulate per-layer activation statistics from captured tensors."""

    def __init__(self, *, default_channel_dim: int = -1) -> None:
        self.default_channel_dim = int(default_channel_dim)
        self._stats_by_layer: dict[str, list[ActivationStats]] = {}

    def update(self, layer_name: str, sample: Any, *, channel_dim: int | None = None) -> None:
        """Reduce one activation tensor and merge it into the named layer."""
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError("layer_name must be a non-empty string")
        stats = activation_amax_from_samples([sample], channel_dim=self.default_channel_dim if channel_dim is None else int(channel_dim))
        self.add_stats(layer_name, stats)

    def add_stats(self, layer_name: str, stats: ActivationStats) -> None:
        """Merge a pre-reduced stats record into the named layer."""
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError("layer_name must be a non-empty string")
        self._stats_by_layer.setdefault(layer_name, []).append(stats)

    def to_stats_map(self) -> dict[str, ActivationStats]:
        """Return one merged ActivationStats record per layer."""
        return {layer_name: merge_activation_stats(records) for layer_name, records in sorted(self._stats_by_layer.items())}

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-compatible summary of accumulated stats."""
        stats = self.to_stats_map()
        return summarize_activation_stats_map(stats)


def summarize_activation_stats_map(stats: Mapping[str, ActivationStats]) -> dict[str, Any]:
    """Return compact counts for a layer-to-stats mapping."""
    total_samples = 0
    total_elements = 0
    layers: list[dict[str, Any]] = []
    for layer_name, item in sorted(stats.items()):
        channel_count = int(item.input_amax.numel())
        sample_count = int(item.sample_count)
        element_count = int(item.element_count)
        total_samples += sample_count
        total_elements += element_count
        layers.append(
            {
                "layer_name": layer_name,
                "channel_count": channel_count,
                "sample_count": sample_count,
                "element_count": element_count,
            }
        )
    return {
        "schema_version": "int4_activation_stats_summary.v1",
        "layer_count": len(stats),
        "sample_count": total_samples,
        "element_count": total_elements,
        "layers": layers,
    }


def _load_sample_rows(path: str | Path) -> list[Mapping[str, Any]]:
    samples_path = Path(path).expanduser()
    text = samples_path.read_text(encoding="utf-8")
    if samples_path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"activation sample manifest is invalid JSON: {samples_path}") from exc
        if isinstance(payload, Mapping):
            rows = payload.get("samples")
            if not isinstance(rows, list):
                raise ValueError(f"activation sample manifest JSON requires a samples list: {samples_path}")
            if not all(isinstance(row, Mapping) for row in rows):
                raise ValueError(f"activation sample manifest samples must be JSON objects: {samples_path}")
            return rows
        if isinstance(payload, list) and all(isinstance(row, Mapping) for row in payload):
            return payload
        raise ValueError(f"activation sample manifest JSON must be an object or list: {samples_path}")

    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid activation sample JSONL at {samples_path}:{line_number}") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"activation sample row must be a JSON object at {samples_path}:{line_number}")
        rows.append(row)
    return rows


def _resolve_sample_file(path: str, *, samples_path: Path, input_root: str | Path | None) -> Path:
    file_path = Path(path).expanduser()
    if file_path.is_absolute():
        return file_path
    if input_root is not None:
        return Path(input_root).expanduser() / file_path
    return samples_path.parent / file_path


def load_activation_sample_refs(
    path: str | Path,
    *,
    input_root: str | Path | None = None,
    default_tensor_name: str = "activation",
    default_channel_dim: int = -1,
) -> list[ActivationSampleRef]:
    """Load activation tensor references from JSONL or JSON."""
    samples_path = Path(path).expanduser()
    refs: list[ActivationSampleRef] = []
    for index, row in enumerate(_load_sample_rows(samples_path), start=1):
        layer_name = row.get("layer_name", row.get("layer"))
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError(f"activation sample row {index} requires a non-empty layer or layer_name")
        file_value = row.get("file_path", row.get("file"))
        if not isinstance(file_value, str) or not file_value:
            raise ValueError(f"activation sample row {index} requires a non-empty file or file_path")
        tensor_name = row.get("tensor_name", row.get("tensor", default_tensor_name))
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"activation sample row {index} tensor name must be a non-empty string")
        channel_dim = row.get("channel_dim", default_channel_dim)
        sample_id = row.get("sample_id", row.get("id"))
        if sample_id is not None and not isinstance(sample_id, str):
            sample_id = str(sample_id)
        refs.append(
            ActivationSampleRef(
                layer_name=layer_name,
                file_path=str(_resolve_sample_file(file_value, samples_path=samples_path, input_root=input_root)),
                tensor_name=tensor_name,
                channel_dim=int(channel_dim),
                sample_id=sample_id,
            )
        )
    if not refs:
        raise ValueError(f"activation sample manifest contains no rows: {samples_path}")
    return refs


def reduce_activation_samples_from_safetensors(
    sample_refs: Iterable[ActivationSampleRef],
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, ActivationStats]:
    """Reduce safetensors activation dumps into per-layer activation statistics."""
    safe_open = _require_safe_open()
    accumulator = ActivationStatsAccumulator()
    refs = list(sample_refs)
    for index, ref in enumerate(refs, start=1):
        if progress is not None:
            progress(
                {
                    "stage": "reduce_activation_sample",
                    "sample_index": index,
                    "sample_count": len(refs),
                    "layer_name": ref.layer_name,
                    "file_path": ref.file_path,
                    "tensor_name": ref.tensor_name,
                }
            )
        file_path = Path(ref.file_path).expanduser()
        if not file_path.is_file():
            raise ValueError(f"activation sample file is missing: {file_path}")
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            if ref.tensor_name not in handle.keys():
                raise ValueError(f"activation sample tensor {ref.tensor_name!r} is missing from {file_path}")
            accumulator.update(ref.layer_name, handle.get_tensor(ref.tensor_name), channel_dim=ref.channel_dim)
    return accumulator.to_stats_map()
