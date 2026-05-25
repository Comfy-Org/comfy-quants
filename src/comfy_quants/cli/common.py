"""Shared CLI helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from comfy_quants.core.errors import ComfyQuantsError


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def local_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def ensure_dir(path: str | Path) -> Path:
    p = local_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def handle_cli_error(exc: Exception) -> int:
    if isinstance(exc, ComfyQuantsError):
        print(f"comfy_quants: {exc}", file=sys.stderr)
        return 2
    raise exc
