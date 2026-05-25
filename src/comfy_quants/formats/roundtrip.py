"""Roundtrip validators for lightweight format utilities."""

from __future__ import annotations

from comfy_quants.formats.pack_int4 import pack_uint4, unpack_uint4


def validate_uint4_roundtrip(values: list[int]) -> bool:
    """Return True if uint4 values survive pack/unpack exactly."""
    return unpack_uint4(pack_uint4(values), len(values)) == list(values)
