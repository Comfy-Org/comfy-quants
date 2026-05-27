import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


def _deps():
    try:
        import torch  # noqa: F401
        from safetensors.torch import load_file  # noqa: F401
    except ImportError:
        return False
    return True


def _load_script_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "dev" / "run_nunchaku_svdquant_fixture.py"
    spec = importlib.util.spec_from_file_location("run_nunchaku_svdquant_fixture", script)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestDevNunchakuSvdquantFixtureScript(unittest.TestCase):
    def setUp(self):
        if not _deps():
            self.skipTest("torch and safetensors are required")

    def _write_fixture(self, tmp: str, *, rank: int, basis: str):
        from comfy_quants.algorithms.int4_svdquant.runtime_fixture import (
            SVDQuantW4A4RuntimeFixtureConfig,
            write_svdquant_w4a4_runtime_fixture,
        )

        return write_svdquant_w4a4_runtime_fixture(
            Path(tmp) / f"fixture-{basis}-r{rank}",
            config=SVDQuantW4A4RuntimeFixtureConfig(
                seed=9101,
                batch=2,
                rank=rank,
                lowrank_branch_input_basis=basis,
            ),
            hash_fixture=False,
        )

    def test_dry_run_layout_accepts_raw_rank16_fixture_without_importing_nunchaku(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            written = self._write_fixture(tmp, rank=16, basis="raw")
            report_path = Path(tmp) / "layout_report.json"
            captured = StringIO()
            with redirect_stdout(captured):
                rc = module.main(
                    [
                        "--fixture",
                        str(written.fixture_path),
                        "--report",
                        str(report_path),
                        "--dry-run-layout",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "layout_ready_external_runtime_not_executed")
            self.assertEqual(result["external_runtime_validation"], "not_run")
            self.assertIs(result["publishable_svdquant_gptq"], False)
            self.assertEqual(result["layer_prefix"], "fixture_layer")
            self.assertEqual(result["in_features"], 128)
            self.assertEqual(result["out_features"], 128)
            self.assertEqual(result["rank"], 16)
            self.assertEqual(result["lowrank_branch_input_basis"], "raw")
            self.assertIs(result["proj_down_smooth_folded"], True)
            self.assertTrue(report_path.exists())

    def test_layout_check_rejects_post_smoothing_fixture_by_default(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            written = self._write_fixture(tmp, rank=16, basis="post_smoothing")
            captured = StringIO()
            with redirect_stdout(captured):
                rc = module.main(
                    [
                        "--fixture",
                        str(written.fixture_path),
                        "--dry-run-layout",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 2)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "failed")
            self.assertIn("raw inputs", result["error"])
            self.assertIs(result["publishable_svdquant_gptq"], False)

    def test_layout_check_rejects_rank_that_external_kernel_commonly_rejects(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            written = self._write_fixture(tmp, rank=4, basis="raw")
            captured = StringIO()
            with redirect_stdout(captured):
                rc = module.main(
                    [
                        "--fixture",
                        str(written.fixture_path),
                        "--dry-run-layout",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 2)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "failed")
            self.assertIn("not a multiple of 16", result["error"])

    def test_activation_runtime_unpack_helpers_decode_known_positions(self):
        import torch

        module = _load_script_module()
        in_features = 128
        k_groups = in_features // module.KITCHEN_GROUP_SIZE
        records_per_page_group = 8 * 2 * 32
        q32 = torch.zeros((k_groups * records_per_page_group, 4), dtype=torch.int32)

        def pack_signed(values):
            packed = 0
            for index, value in enumerate(values):
                packed |= (int(value) & 0xF) << (4 * index)
            if packed >= 2**31:
                packed -= 2**32
            return packed

        row0_values = [1, -1, 7, -8, 0, 3, -4, 5]
        row8_values = [-2, 2, -3, 3, -4, 4, -5, 5]
        row0_c32_values = [6, -6, 0, 1, 2, 3, 4, -7]
        row8_c32_values = [-1, -2, -3, -4, 7, 6, 5, 4]
        q32[0, 0] = pack_signed(row0_values)
        q32[0, 1] = pack_signed(row8_values)
        q32[0, 2] = pack_signed(row0_c32_values)
        q32[0, 3] = pack_signed(row8_c32_values)
        packed_q = q32.contiguous().view(torch.int8).view(256, 64)

        decoded_q = module._unpack_nunchaku_activation_q(packed_q, in_features=in_features, signed=True)

        self.assertEqual(tuple(decoded_q.shape), (256, 128))
        self.assertEqual(decoded_q[0, 0:8].tolist(), row0_values)
        self.assertEqual(decoded_q[8, 0:8].tolist(), row8_values)
        self.assertEqual(decoded_q[0, 32:40].tolist(), row0_c32_values)
        self.assertEqual(decoded_q[8, 32:40].tolist(), row8_c32_values)
        self.assertEqual(int(torch.count_nonzero(decoded_q).item()), 30)

        scales_flat = torch.zeros(k_groups * 8 * 16 * 2, dtype=torch.bfloat16)
        scales_flat[0] = torch.tensor(1.25, dtype=torch.bfloat16)
        scales_flat[1] = torch.tensor(2.5, dtype=torch.bfloat16)
        second_group_offset = 8 * 16 * 2
        scales_flat[second_group_offset] = torch.tensor(3.75, dtype=torch.bfloat16)
        scales_flat[second_group_offset + 1] = torch.tensor(4.5, dtype=torch.bfloat16)
        packed_scales = scales_flat.view(2, 256)

        decoded_scales = module._unpack_nunchaku_activation_scales(
            packed_scales,
            in_features=in_features,
            padded_rows=256,
        )

        self.assertEqual(tuple(decoded_scales.shape), (256, 2))
        self.assertEqual(float(decoded_scales[0, 0]), 1.25)
        self.assertEqual(float(decoded_scales[8, 0]), 2.5)
        self.assertEqual(float(decoded_scales[0, 1]), 3.75)
        self.assertEqual(float(decoded_scales[8, 1]), 4.5)

    def test_group_fma_main_replay_applies_runtime_dtype_rounding_per_group(self):
        import torch
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs

        module = _load_script_module()
        rows = 2
        in_features = 128
        out_features = 4
        groups = in_features // module.KITCHEN_GROUP_SIZE
        dtype = torch.bfloat16

        q_values = ((torch.arange(out_features * in_features).reshape(out_features, in_features) * 5) % 16 - 8).to(
            torch.int8
        )
        decoded_q = ((torch.arange(rows * in_features).reshape(rows, in_features) * 3) % 16 - 8).to(torch.int8)
        decoded_scales = torch.tensor(
            [
                [0.03125, 0.1875],
                [0.09375, 0.25],
            ],
            dtype=torch.float32,
        )
        wscales = torch.tensor(
            [
                [0.25, 0.375, 0.5, 0.625],
                [0.75, 0.875, 1.0, 1.125],
            ],
            dtype=torch.float32,
        )

        actual = module._decoded_activation_group_fma_main_replay(
            qweight=pack_signed_int4_pairs(q_values),
            wscales=wscales,
            decoded_q=decoded_q,
            decoded_scales=decoded_scales,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            dtype=dtype,
        )

        expected = torch.zeros((rows, out_features), dtype=dtype)
        for group in range(groups):
            start = group * module.KITCHEN_GROUP_SIZE
            stop = start + module.KITCHEN_GROUP_SIZE
            intacc = torch.matmul(decoded_q[:, start:stop].to(torch.int32), q_values[:, start:stop].to(torch.int32).t())
            product = (
                decoded_scales[:, group].to(dtype).unsqueeze(1) * wscales[group].to(dtype).unsqueeze(0)
            ).to(dtype)
            expected = (intacc.to(dtype).float() * product.float() + expected.float()).to(dtype)

        self.assertEqual(actual.dtype, dtype)
        self.assertTrue(torch.equal(actual, expected))

    def test_lowrank_runtime_replay_uses_runtime_dtype_operands(self):
        import torch

        module = _load_script_module()
        dtype = torch.bfloat16
        rows = 2
        in_features = 128
        out_features = 4
        rank = 16
        inputs = (torch.arange(rows * in_features, dtype=torch.float32).reshape(rows, in_features) / 97.0) - 1.25
        proj_down = torch.linspace(-0.45, 0.55, in_features * rank, dtype=torch.float32).reshape(in_features, rank)
        proj_up = torch.linspace(0.35, -0.25, out_features * rank, dtype=torch.float32).reshape(out_features, rank)
        payload = module.NunchakuSvdquantFixturePayload(
            layer_prefix="fixture_layer",
            qweight=torch.zeros((out_features, in_features // 2), dtype=torch.int8),
            wscales=torch.ones((in_features // module.KITCHEN_GROUP_SIZE, out_features), dtype=torch.float32),
            smooth_factor=torch.ones((in_features,), dtype=torch.float32),
            proj_down=proj_down,
            proj_up=proj_up,
            bias=None,
            inputs=inputs,
            quant_config={"act_unsigned": False},
        )

        actual = module._lowrank_runtime_like_replay(payload, dtype=dtype)
        lora_act = torch.matmul(inputs.to(dtype).float(), proj_down.to(dtype).float())
        expected = torch.matmul(lora_act.to(dtype).float(), proj_up.to(dtype).float().t()).to(dtype)

        self.assertEqual(actual.dtype, dtype)
        self.assertTrue(torch.equal(actual, expected))

    def test_bias_runtime_replay_casts_and_broadcasts(self):
        import torch

        module = _load_script_module()
        dtype = torch.bfloat16
        rows = 3
        in_features = 128
        out_features = 4
        rank = 16
        bias = torch.tensor([0.1001, -0.2002, 0.3003, -0.4004], dtype=torch.float32)
        payload = module.NunchakuSvdquantFixturePayload(
            layer_prefix="fixture_layer",
            qweight=torch.zeros((out_features, in_features // 2), dtype=torch.int8),
            wscales=torch.ones((in_features // module.KITCHEN_GROUP_SIZE, out_features), dtype=torch.float32),
            smooth_factor=torch.ones((in_features,), dtype=torch.float32),
            proj_down=torch.zeros((in_features, rank), dtype=torch.float32),
            proj_up=torch.zeros((out_features, rank), dtype=torch.float32),
            bias=bias,
            inputs=torch.zeros((rows, in_features), dtype=torch.float32),
            quant_config={"act_unsigned": False},
        )

        actual = module._bias_runtime_like_replay(payload, dtype=dtype)
        expected = bias.to(dtype).reshape(1, out_features).expand(rows, out_features).contiguous()

        self.assertEqual(actual.dtype, dtype)
        self.assertTrue(torch.equal(actual, expected))

    def test_full_runtime_replay_combines_decoded_main_bias_and_lowrank(self):
        import torch
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs

        module = _load_script_module()
        dtype = torch.bfloat16
        rows = 2
        in_features = 128
        out_features = 4
        rank = 16
        groups = in_features // module.KITCHEN_GROUP_SIZE
        q_values = ((torch.arange(out_features * in_features).reshape(out_features, in_features) * 7) % 16 - 8).to(
            torch.int8
        )
        decoded_q = ((torch.arange(rows * in_features).reshape(rows, in_features) * 5) % 16 - 8).to(torch.int8)
        decoded_scales = torch.tensor([[0.03125, 0.1875], [0.09375, 0.25]], dtype=torch.float32)
        wscales = torch.tensor(
            [[0.25, 0.375, 0.5, 0.625], [0.75, 0.875, 1.0, 1.125]],
            dtype=torch.float32,
        )
        inputs = (torch.arange(rows * in_features, dtype=torch.float32).reshape(rows, in_features) / 113.0) - 0.75
        proj_down = torch.linspace(-0.2, 0.3, in_features * rank, dtype=torch.float32).reshape(in_features, rank)
        proj_up = torch.linspace(0.15, -0.35, out_features * rank, dtype=torch.float32).reshape(out_features, rank)
        bias = torch.tensor([0.1251, -0.2502, 0.3753, -0.5004], dtype=torch.float32)
        payload = module.NunchakuSvdquantFixturePayload(
            layer_prefix="fixture_layer",
            qweight=pack_signed_int4_pairs(q_values),
            wscales=wscales,
            smooth_factor=torch.ones((in_features,), dtype=torch.float32),
            proj_down=proj_down,
            proj_up=proj_up,
            bias=bias,
            inputs=inputs,
            quant_config={"act_unsigned": False},
        )

        actual = module._full_runtime_like_replay(
            payload,
            decoded_q=decoded_q,
            decoded_scales=decoded_scales,
            dtype=dtype,
        )

        main = torch.zeros((rows, out_features), dtype=dtype)
        for group in range(groups):
            start = group * module.KITCHEN_GROUP_SIZE
            stop = start + module.KITCHEN_GROUP_SIZE
            intacc = torch.matmul(decoded_q[:, start:stop].to(torch.int32), q_values[:, start:stop].to(torch.int32).t())
            product = (
                decoded_scales[:, group].to(dtype).unsqueeze(1) * wscales[group].to(dtype).unsqueeze(0)
            ).to(dtype)
            main = (intacc.to(dtype).float() * product.float() + main.float()).to(dtype)
        main_bias = (main.float() + bias.to(dtype).reshape(1, out_features).expand(rows, out_features).float()).to(dtype)
        lora_act = torch.matmul(inputs.to(dtype).float(), proj_down.to(dtype).float())
        lowrank_acc = torch.matmul(lora_act.to(dtype).float(), proj_up.to(dtype).float().t())
        expected = (main_bias.float() + lowrank_acc.float()).to(dtype)

        self.assertEqual(actual.dtype, dtype)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
