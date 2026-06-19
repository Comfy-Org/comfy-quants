"""System and hardware discovery used by diagnostics and manifests."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Any

from comfy_quants import __version__


def _package_version_if_importable(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def _run_command(args: list[str], timeout: float = 5.0) -> str | None:
    try:
        proc = subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def collect_system_info() -> dict[str, Any]:
    """Collect lightweight environment information without requiring torch/CUDA."""
    info: dict[str, Any] = {
        "comfy_quants_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "packages": {
            "torch": _package_version_if_importable("torch"),
            "safetensors": _package_version_if_importable("safetensors"),
            "diffusers": _package_version_if_importable("diffusers"),
            "transformers": _package_version_if_importable("transformers"),
            "yaml": _package_version_if_importable("yaml"),
        },
        "gpu": {
            "nvidia_smi_found": shutil.which("nvidia-smi") is not None,
            "nvidia_smi": None,
            "detected_profile": None,
        },
    }
    if info["gpu"]["nvidia_smi_found"]:
        query = _run_command([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ])
        info["gpu"]["nvidia_smi"] = query
        if query and "RTX PRO 6000" in query and "Blackwell" in query:
            info["gpu"]["detected_profile"] = "rtx_pro_6000_blackwell_96gb"
    return info
