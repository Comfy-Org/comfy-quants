"""Backward-compatible module entrypoint for ``python -m comfy_reduction.cli.main``."""

from __future__ import annotations

from comfy_quants.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
