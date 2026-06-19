"""INT4 packing utilities.

Nibble order is explicit and stable: by default the first value is stored in the
low nibble and the second value in the high nibble of each byte.
"""

from __future__ import annotations


def _validate_uint4(value: int) -> int:
    ivalue = int(value)
    if ivalue < 0 or ivalue > 15:
        raise ValueError(f"uint4 value out of range [0, 15]: {value}")
    return ivalue


def pack_uint4(values: list[int] | tuple[int, ...], *, low_first: bool = True) -> bytes:
    """Pack unsigned 4-bit values into bytes."""
    packed = bytearray()
    it = list(values)
    for i in range(0, len(it), 2):
        first = _validate_uint4(it[i])
        second = _validate_uint4(it[i + 1]) if i + 1 < len(it) else 0
        if low_first:
            packed.append(first | (second << 4))
        else:
            packed.append((first << 4) | second)
    return bytes(packed)


def unpack_uint4(data: bytes | bytearray, count: int, *, low_first: bool = True) -> list[int]:
    """Unpack unsigned 4-bit values from bytes."""
    if count < 0:
        raise ValueError("count must be non-negative")
    values: list[int] = []
    for byte in bytes(data):
        if low_first:
            values.extend([byte & 0x0F, (byte >> 4) & 0x0F])
        else:
            values.extend([(byte >> 4) & 0x0F, byte & 0x0F])
        if len(values) >= count:
            break
    if len(values) < count:
        raise ValueError("not enough packed data for requested count")
    return values[:count]


def signed_int4_to_uint4(value: int) -> int:
    """Map signed INT4 [-8, 7] to uint4 storage [0, 15]."""
    ivalue = int(value)
    if ivalue < -8 or ivalue > 7:
        raise ValueError(f"int4 value out of range [-8, 7]: {value}")
    return ivalue & 0x0F


def uint4_to_signed_int4(value: int) -> int:
    """Map uint4 storage [0, 15] to signed INT4 [-8, 7]."""
    ivalue = _validate_uint4(value)
    return ivalue - 16 if ivalue >= 8 else ivalue
