import unittest

from comfy_quants.formats.pack_int4 import pack_uint4, signed_int4_to_uint4, uint4_to_signed_int4, unpack_uint4
from comfy_quants.formats.roundtrip import validate_uint4_roundtrip


class TestInt4Packing(unittest.TestCase):
    def test_uint4_roundtrip_odd_count(self):
        values = [0, 1, 2, 15, 8]
        packed = pack_uint4(values)
        self.assertEqual(unpack_uint4(packed, len(values)), values)
        self.assertTrue(validate_uint4_roundtrip(values))

    def test_signed_mapping(self):
        self.assertEqual(uint4_to_signed_int4(signed_int4_to_uint4(-8)), -8)
        self.assertEqual(uint4_to_signed_int4(signed_int4_to_uint4(7)), 7)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            pack_uint4([16])
        with self.assertRaises(ValueError):
            signed_int4_to_uint4(8)


if __name__ == "__main__":
    unittest.main()
