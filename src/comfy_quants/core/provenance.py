"""Provenance helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from comfy_quants import __version__
from comfy_quants.utils.system_info import collect_system_info


def build_provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a lightweight provenance object for jobs and inspections."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "comfy_quants",
        "comfy_quants_version": __version__,
        "system": collect_system_info(),
        "extra": extra or {},
    }
