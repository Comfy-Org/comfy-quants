"""Backward-compatible alias for the renamed :mod:`comfy_quants` package."""

from __future__ import annotations

import sys

import comfy_quants as _comfy_quants
from comfy_quants import *  # noqa: F401,F403
from comfy_quants import __version__  # noqa: F401

# Make ``import comfy_reduction.<submodule>`` resolve through the new package
# path for callers that have not migrated yet.
sys.modules[__name__] = _comfy_quants
