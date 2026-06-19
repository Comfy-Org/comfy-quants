"""JSON/YAML file IO helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - dependency should be installed by package metadata
    yaml = None


def _local_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def read_json(path: str | Path) -> Any:
    return json.loads(_local_path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    p = _local_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_yaml(path: str | Path) -> Any:
    p = _local_path(path)
    if yaml is None:
        if p.suffix.lower() == ".json":
            return read_json(p)
        raise RuntimeError("PyYAML is required to read YAML config files")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: str | Path, value: Any) -> None:
    p = _local_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        p.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    p.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
