"""Backward-compatible entrypoint for ``python -m genquant.cli.main``."""

from comfy_quants.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
