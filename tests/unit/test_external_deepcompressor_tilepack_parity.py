import importlib.util
import os
import unittest
from pathlib import Path


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _load_deepcompressor_tilepack():
    root = Path(os.environ.get("COMFY_QUANTS_DEEPCOMPRESSOR_SOURCE", str(Path.cwd().parent / "external" / "deepcompressor-yidhar")))
    path = root / "deepcompressor" / "backend" / "kitchen" / "tilepack.py"
    if not path.is_file():
        raise unittest.SkipTest(f"DeepCompressor tilepack oracle is not available at {path}")
    spec = importlib.util.spec_from_file_location("_comfy_quants_dc_tilepack_oracle", path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"cannot import DeepCompressor tilepack oracle at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExternalDeepCompressorTilepackParity(unittest.TestCase):
    def setUp(self):
        self.torch = _torch()
        if self.torch is None:
            self.skipTest("torch is required")
        self.dc = _load_deepcompressor_tilepack()

    def test_signed_int4_pair_codec_matches_deepcompressor_storage_codec(self):
        torch = self.torch
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs, unpack_signed_int4_pairs

        dense = torch.tensor(
            [
                [-8, -7, -3, 0, 1, 2, 6, 7],
                [7, 6, 2, 1, 0, -3, -7, -8],
            ],
            dtype=torch.int8,
        )

        ours = pack_signed_int4_pairs(dense)
        theirs = self.dc.pack_int4_pairs(dense)
        self.assertTrue(torch.equal(ours, theirs))
        self.assertTrue(torch.equal(unpack_signed_int4_pairs(ours), self.dc.unpack_int4_pairs(theirs)))
        self.assertTrue(torch.equal(unpack_signed_int4_pairs(ours), dense))

    def test_weight_tile_pack_matches_deepcompressor(self):
        torch = self.torch
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs
        from comfy_quants.formats.kitchen_tilepack import pack_weight_tile, unpack_weight_tile

        n, k = 256, 128
        dense = torch.arange(n * k, dtype=torch.int16).reshape(n, k).remainder(16).sub(8).to(torch.int8)
        natural = pack_signed_int4_pairs(dense)

        ours = pack_weight_tile(natural)
        theirs = self.dc.pack_weight_tile(self.dc.pack_int4_pairs(dense))
        self.assertTrue(torch.equal(ours, theirs))
        self.assertTrue(torch.equal(unpack_weight_tile(ours), natural))

    def test_n_axis_and_weight_scale_pack_match_deepcompressor(self):
        torch = self.torch
        from comfy_quants.formats.kitchen_tilepack import pack_n_axis, pack_weight_scale

        n = 256
        natural_n_axis = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3).to(torch.float16)
        self.assertTrue(torch.equal(pack_n_axis(natural_n_axis), self.dc.pack_n_axis(natural_n_axis)))

        weight_scale = torch.arange(2 * n, dtype=torch.float32).reshape(2, n).div(1000).to(torch.bfloat16)
        self.assertTrue(torch.equal(pack_weight_scale(weight_scale), self.dc.pack_weight_scale(weight_scale)))


if __name__ == "__main__":
    unittest.main()
