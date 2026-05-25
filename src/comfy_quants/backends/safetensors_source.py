"""Safetensors checkpoint source helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from comfy_quants.core.errors import PayloadWriteError


SAFETENSORS_INDEX_SUFFIX = ".safetensors.index.json"
DEFAULT_SAFETENSORS_INDEX_NAMES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
)


def _require_safe_open():
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - exercised only without package deps
        raise PayloadWriteError("safetensors is required for checkpoint source reading") from exc
    return safe_open


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def is_safetensors_index_file(path: Path) -> bool:
    return path.is_file() and path.name.endswith(SAFETENSORS_INDEX_SUFFIX)


def _safe_file_ref(file_ref: str) -> str:
    rel = PurePosixPath(file_ref)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise PayloadWriteError(f"safetensors index file reference is invalid: {file_ref}")
    return str(rel)


def _source_name(row: dict[str, Any]) -> str:
    name = row.get("source_name") or row.get("name")
    if not isinstance(name, str) or not name:
        raise PayloadWriteError(f"tensor row has no source name: {row.get('name')}")
    return name


@dataclass(frozen=True)
class SafetensorsTensorSource:
    """A local safetensors source, either one file or an indexed shard set."""

    source_path: Path
    base_dir: Path
    file_map: dict[str, str]
    index_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    layout: str = "single_file"

    @classmethod
    def from_path(cls, source_path: str | Path) -> "SafetensorsTensorSource":
        path = _expand_path(source_path)
        if path.is_dir():
            return cls._from_directory(path)
        if is_safetensors_index_file(path):
            return cls._from_index(path)
        if path.is_file() and path.suffix == ".safetensors":
            return cls._from_single_file(path)
        raise PayloadWriteError(f"safetensors source must be a .safetensors file, an index JSON, or an indexed directory: {path}")

    @classmethod
    def _from_single_file(cls, path: Path) -> "SafetensorsTensorSource":
        safe_open = _require_safe_open()
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            file_map = {str(key): path.name for key in handle.keys()}
        return cls(source_path=path, base_dir=path.parent, file_map=file_map, layout="single_file")

    @classmethod
    def _from_index(cls, index_path: Path) -> "SafetensorsTensorSource":
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PayloadWriteError(f"safetensors index JSON is invalid: {index_path}") from exc
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise PayloadWriteError(f"safetensors index has no weight_map: {index_path}")
        file_map: dict[str, str] = {}
        for tensor_name, file_ref in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise PayloadWriteError(f"safetensors index contains an invalid tensor name: {index_path}")
            if not isinstance(file_ref, str) or not file_ref:
                raise PayloadWriteError(f"safetensors index contains an invalid file reference for {tensor_name}")
            file_map[tensor_name] = _safe_file_ref(file_ref)
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        return cls(
            source_path=index_path,
            base_dir=index_path.parent,
            file_map=file_map,
            index_path=index_path,
            metadata=dict(metadata),
            layout="indexed_shards",
        )

    @classmethod
    def _from_directory(cls, path: Path) -> "SafetensorsTensorSource":
        for name in DEFAULT_SAFETENSORS_INDEX_NAMES:
            candidate = path / name
            if candidate.is_file():
                source = cls._from_index(candidate)
                return cls(
                    source_path=path,
                    base_dir=source.base_dir,
                    file_map=source.file_map,
                    index_path=source.index_path,
                    metadata=source.metadata,
                    layout=source.layout,
                )
        indexes = sorted(path.glob(f"*{SAFETENSORS_INDEX_SUFFIX}"))
        if len(indexes) == 1:
            source = cls._from_index(indexes[0])
            return cls(
                source_path=path,
                base_dir=source.base_dir,
                file_map=source.file_map,
                index_path=source.index_path,
                metadata=source.metadata,
                layout=source.layout,
            )
        if len(indexes) > 1:
            names = ", ".join(index.name for index in indexes[:8])
            suffix = "" if len(indexes) <= 8 else f", ... ({len(indexes)} total)"
            raise PayloadWriteError(f"directory contains multiple safetensors index files: {names}{suffix}")
        files = sorted(path.glob("*.safetensors"))
        if len(files) == 1:
            source = cls._from_single_file(files[0])
            return cls(source_path=path, base_dir=source.base_dir, file_map=source.file_map, index_path=None, metadata={}, layout=source.layout)
        raise PayloadWriteError(f"directory must contain a safetensors index or exactly one .safetensors file: {path}")

    def keys(self) -> set[str]:
        return set(self.file_map)

    def file_ref_for(self, tensor_name: str) -> str:
        try:
            return self.file_map[tensor_name]
        except KeyError as exc:
            raise PayloadWriteError(f"selected tensor is absent from safetensors source: {tensor_name}") from exc

    def file_path_for(self, tensor_name: str) -> Path:
        rel = PurePosixPath(self.file_ref_for(tensor_name))
        return self.base_dir.joinpath(*rel.parts)

    def missing_tensors(self, tensor_names: Iterable[str]) -> list[str]:
        available = self.keys()
        return [name for name in tensor_names if name not in available]

    def selected_file_counts(self, tensor_names: Iterable[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in tensor_names:
            if name not in self.file_map:
                continue
            ref = self.file_map[name]
            counts[ref] = counts.get(ref, 0) + 1
        return dict(sorted(counts.items()))

    def group_rows_by_file(self, rows: Iterable[dict[str, Any]]) -> dict[Path, list[dict[str, Any]]]:
        grouped: dict[Path, list[dict[str, Any]]] = {}
        for row in rows:
            source_name = _source_name(row)
            if source_name not in self.file_map:
                continue
            grouped.setdefault(self.file_path_for(source_name), []).append(row)
        return dict(sorted(grouped.items(), key=lambda item: str(item[0])))

    def describe(self, selected_tensor_names: Iterable[str] | None = None) -> dict[str, Any]:
        unique_files = sorted(set(self.file_map.values()))
        description: dict[str, Any] = {
            "path": str(self.source_path),
            "layout": self.layout,
            "tensor_count": len(self.file_map),
            "file_count": len(unique_files),
        }
        if self.index_path is not None:
            description["index_path"] = str(self.index_path)
            if self.metadata:
                description["index_metadata"] = dict(self.metadata)
        if selected_tensor_names is not None:
            description["selected_file_counts"] = self.selected_file_counts(selected_tensor_names)
        return description


@dataclass
class SafetensorsSourceCoverageReport:
    """Name and optional shape coverage for selected tensors."""

    source: str
    source_layout: str
    selected_tensor_count: int
    matched_tensor_count: int
    missing_tensor_count: int
    selected_file_counts: dict[str, int]
    shape_checked_tensor_count: int = 0
    shape_mismatch_count: int = 0
    missing_tensors: list[str] = field(default_factory=list)
    shape_mismatches: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "source_key_coverage.v1"
    source_format: str = "safetensors"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_safetensors_source_coverage(
    *,
    source_checkpoint: str | Path,
    tensor_index: dict[str, Any],
    check_shapes: bool = False,
    device: str = "cpu",
) -> SafetensorsSourceCoverageReport:
    """Build a coverage report for selected tensor names in a safetensors source."""
    source = SafetensorsTensorSource.from_path(source_checkpoint)
    rows = list(tensor_index.get("tensors") or [])
    selected_names = [_source_name(row) for row in rows]
    missing = source.missing_tensors(selected_names)
    shape_mismatches: list[dict[str, Any]] = []
    shape_checked = 0

    if check_shapes:
        safe_open = _require_safe_open()
        for file_path, file_rows in source.group_rows_by_file(rows).items():
            if not file_path.is_file():
                raise PayloadWriteError(f"source tensor file is missing: {file_path}")
            with safe_open(str(file_path), framework="pt", device=device) as handle:
                for row in file_rows:
                    name = _source_name(row)
                    if name not in handle.keys():
                        missing.append(name)
                        continue
                    expected = [int(dim) for dim in row.get("shape") or []]
                    if hasattr(handle, "get_slice"):
                        actual = [int(dim) for dim in handle.get_slice(name).get_shape()]
                    else:
                        actual = [int(dim) for dim in handle.get_tensor(name).shape]
                    shape_checked += 1
                    if actual != expected:
                        shape_mismatches.append({"name": name, "expected": expected, "actual": actual})

    missing = sorted(set(missing))
    matched = len(selected_names) - len(missing)
    return SafetensorsSourceCoverageReport(
        source=str(source.source_path),
        source_layout=source.layout,
        selected_tensor_count=len(selected_names),
        matched_tensor_count=matched,
        missing_tensor_count=len(missing),
        selected_file_counts=source.selected_file_counts(selected_names),
        shape_checked_tensor_count=shape_checked,
        shape_mismatch_count=len(shape_mismatches),
        missing_tensors=missing,
        shape_mismatches=shape_mismatches,
    )
