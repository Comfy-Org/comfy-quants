import json
import tempfile
import unittest
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from comfy_quants.cli.main import main


def _torch_safetensors_deps():
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        return None
    return torch, save_file


class TestInt4Calibration(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.save_file = deps

    def test_activation_stats_accumulator_merges_samples(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.calibration import ActivationStatsAccumulator

        accumulator = ActivationStatsAccumulator()
        accumulator.update("transformer_blocks.0.attn.to_q", torch.tensor([[1.0, -2.0], [3.0, 4.0]]))
        accumulator.update("transformer_blocks.0.attn.to_q", torch.tensor([[-5.0, 1.0]]))
        stats = accumulator.to_stats_map()["transformer_blocks.0.attn.to_q"]

        self.assertTrue(torch.equal(stats.input_amax, torch.tensor([5.0, 4.0])))
        self.assertEqual(stats.sample_count, 2)
        self.assertEqual(stats.element_count, 3)
        self.assertTrue(torch.allclose(stats.input_rms, torch.sqrt(torch.tensor([(1.0 + 9.0 + 25.0) / 3.0, (4.0 + 16.0 + 1.0) / 3.0]))))

    def test_cli_reduce_int4_activations_writes_stats_json(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.stats import load_activation_stats_map

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.safetensors"
            second = root / "second.safetensors"
            samples = root / "samples.jsonl"
            out = root / "stats"
            layer = "transformer_blocks.0.attn.to_q"
            self.save_file({"hidden": torch.tensor([[1.0, -2.0], [3.0, 4.0]])}, str(first))
            self.save_file({"hidden": torch.tensor([[-5.0, 1.0]])}, str(second))
            samples.write_text(
                "\n".join(
                    [
                        json.dumps({"layer": layer, "file": first.name, "tensor": "hidden"}),
                        json.dumps({"layer": layer, "file": second.name, "tensor": "hidden"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "calib",
                        "reduce-int4-activations",
                        "--samples",
                        str(samples),
                        "--out",
                        str(out),
                        "--no-progress",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["layer_count"], 1)
            self.assertEqual(result["sample_count"], 2)
            stats_path = out / "int4_activation_stats.json"
            self.assertTrue(stats_path.exists())
            loaded = load_activation_stats_map(stats_path)
            self.assertTrue(torch.equal(loaded[layer].input_amax, torch.tensor([5.0, 4.0])))

    def test_cli_records_materializes_manifest_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            image_root.mkdir()
            edit_set = root / "edits.jsonl"
            manifest = root / "calibration_manifest.json"
            out = root / "records"
            edit_set.write_text(
                json.dumps(
                    {
                        "id": "case-1",
                        "image": "source.png",
                        "prompt": "change the jacket to blue",
                        "edit_type": "appearance_preserve",
                        "language": "en",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "id": "case-2",
                        "image": "sign.png",
                        "prompt": "change the sign text",
                        "edit_type": "text_edit",
                        "language": "en",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "0.1.0",
                        "family": "qwen_image_edit",
                        "edit_set": str(edit_set),
                        "image_root": str(image_root),
                        "edit_types": ["text_edit"],
                        "manifest_kind": "calibration_dataset",
                    }
                ),
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(["calib", "records", "--manifest", str(manifest), "--out", str(out), "--json"])

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["record_count"], 1)
            rows = [json.loads(line) for line in (out / "calibration_records.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["case_id"], "case-2")
            self.assertEqual(rows[0]["image"], str(image_root / "sign.png"))

    def test_cli_plan_int4_capture_writes_qwen_targets(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "diffusion_pytorch_model.safetensors"
            records = root / "calibration_records.jsonl"
            out = root / "capture-plan"
            self.save_file(
                {
                    "transformer_blocks.0.attn.to_q.weight": torch.zeros((128, 128), dtype=torch.float16),
                    "transformer_blocks.0.attn.to_q.bias": torch.zeros((128,), dtype=torch.float16),
                },
                str(source),
            )
            records.write_text(
                json.dumps({"case_id": "case-1", "prompt": "make the jacket blue", "image": "source.png"}) + "\n",
                encoding="utf-8",
            )

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "calib",
                        "plan-int4-capture",
                        "--family",
                        "qwen_image_edit",
                        "--source",
                        str(source),
                        "--records",
                        str(records),
                        "--out",
                        str(out),
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "capture_plan_written")
            self.assertEqual(result["selected_layer_count"], 1)
            self.assertEqual(result["record_count"], 1)

            plan_path = out / "capture_plan.json"
            template_path = out / "activation_samples.template.jsonl"
            report_path = out / "capture_report.json"
            self.assertTrue(plan_path.exists())
            self.assertTrue(template_path.exists())
            self.assertTrue(report_path.exists())

            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["schema_version"], "int4_activation_capture_plan.v1")
            self.assertEqual(plan["capture_mode"], "plan_only")
            self.assertEqual(plan["runtime_state"], "not_executed")
            self.assertEqual(plan["selected_layer_count"], 1)
            target = plan["targets"][0]
            self.assertEqual(target["output_prefix"], "transformer_blocks.0.attn.to_q")
            self.assertEqual(target["source_prefix"], "transformer_blocks.0.attn.to_q")
            self.assertEqual(target["input_channels"], 128)
            self.assertEqual(target["output_channels"], 128)
            self.assertEqual(target["capture_tensor_name"], "transformer_blocks.0.attn.to_q.input")
            self.assertEqual(target["stats_lookup_candidates"], ["transformer_blocks.0.attn.to_q"])

            template_rows = [json.loads(line) for line in template_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(template_rows[0]["layer"], "transformer_blocks.0.attn.to_q")
            self.assertEqual(template_rows[0]["tensor"], "transformer_blocks.0.attn.to_q.input")

    def test_cli_materialize_int4_capture_writes_sample_manifest(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "diffusion_pytorch_model.safetensors"
            records = root / "calibration_records.jsonl"
            plan_dir = root / "capture-plan"
            capture_run = root / "capture-run"
            self.save_file(
                {
                    "transformer_blocks.0.attn.to_q.weight": torch.zeros((128, 128), dtype=torch.float16),
                    "transformer_blocks.0.attn.to_q.bias": torch.zeros((128,), dtype=torch.float16),
                },
                str(source),
            )
            records.write_text(
                json.dumps({"case_id": "case-1", "prompt": "make the jacket blue"}) + "\n"
                + json.dumps({"case_id": "case/2", "prompt": "change the sign text"}) + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(StringIO()):
                rc = main(
                    [
                        "calib",
                        "plan-int4-capture",
                        "--family",
                        "qwen_image_edit",
                        "--source",
                        str(source),
                        "--records",
                        str(records),
                        "--out",
                        str(plan_dir),
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "calib",
                        "materialize-int4-capture",
                        "--plan",
                        str(plan_dir / "capture_plan.json"),
                        "--out",
                        str(capture_run),
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "activation_sample_manifest_written")
            self.assertEqual(result["case_count"], 2)
            self.assertEqual(result["target_count"], 1)
            self.assertEqual(result["sample_ref_count"], 2)

            samples_path = capture_run / "activation_samples.jsonl"
            report_path = capture_run / "capture_materialization_report.json"
            self.assertTrue(samples_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue((capture_run / "activation_tensors").is_dir())
            rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["sample_id"], "case-1:transformer_blocks.0.attn.to_q")
            self.assertEqual(rows[0]["case_id"], "case-1")
            self.assertEqual(rows[0]["layer"], "transformer_blocks.0.attn.to_q")
            self.assertEqual(rows[0]["file"], "activation_tensors/case-1.safetensors")
            self.assertEqual(rows[0]["tensor"], "transformer_blocks.0.attn.to_q.input")
            self.assertEqual(rows[0]["channel_dim"], -1)
            self.assertEqual(rows[1]["case_id"], "case/2")
            self.assertEqual(rows[1]["file"], "activation_tensors/case_2.safetensors")

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], "int4_activation_capture_materialization_report.v1")
            self.assertEqual({item["kind"] for item in report["written_files"]}, {"activation_samples", "capture_materialization_report"})

    def test_activation_case_writer_roundtrips_through_reducer(self):
        torch = self.torch
        from comfy_quants.algorithms.int4_svdquant.stats import load_activation_stats_map
        from comfy_quants.backends.activation_capture.materialize import (
            materialize_int4_activation_sample_manifest,
            write_int4_activation_case_safetensors,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "diffusion_pytorch_model.safetensors"
            records = root / "calibration_records.jsonl"
            plan_dir = root / "capture-plan"
            capture_run = root / "capture-run"
            stats_dir = root / "stats"
            layer = "transformer_blocks.0.attn.to_q"
            tensor_name = f"{layer}.input"
            self.save_file(
                {
                    f"{layer}.weight": torch.zeros((128, 128), dtype=torch.float16),
                    f"{layer}.bias": torch.zeros((128,), dtype=torch.float16),
                },
                str(source),
            )
            records.write_text(json.dumps({"case_id": "case-1", "prompt": "make the jacket blue"}) + "\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                rc = main(
                    [
                        "calib",
                        "plan-int4-capture",
                        "--family",
                        "qwen_image_edit",
                        "--source",
                        str(source),
                        "--records",
                        str(records),
                        "--out",
                        str(plan_dir),
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)

            materialize_int4_activation_sample_manifest(plan=plan_dir / "capture_plan.json", out_dir=capture_run)
            activation = torch.stack([torch.arange(128, dtype=torch.float32), -2.0 * torch.arange(128, dtype=torch.float32)])
            write_report = write_int4_activation_case_safetensors(
                plan=plan_dir / "capture_plan.json",
                case_id="case-1",
                tensors={tensor_name: activation},
                out_dir=capture_run,
            )
            self.assertEqual(write_report.tensor_count, 1)
            self.assertTrue((capture_run / "activation_tensors" / "case-1.safetensors").exists())

            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "calib",
                        "reduce-int4-activations",
                        "--samples",
                        str(capture_run / "activation_samples.jsonl"),
                        "--input-root",
                        str(capture_run),
                        "--out",
                        str(stats_dir),
                        "--no-progress",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["layer_count"], 1)
            self.assertEqual(result["sample_count"], 1)
            stats = load_activation_stats_map(stats_dir / "int4_activation_stats.json")
            self.assertEqual(int(stats[layer].input_amax.numel()), 128)
            self.assertTrue(torch.equal(stats[layer].input_amax, 2.0 * torch.arange(128, dtype=torch.float32)))
            self.assertEqual(stats[layer].element_count, 2)

    def test_activation_case_writer_rejects_channel_mismatch(self):
        torch = self.torch
        from comfy_quants.backends.activation_capture.materialize import write_int4_activation_case_safetensors
        from comfy_quants.core.errors import PayloadWriteError

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "diffusion_pytorch_model.safetensors"
            records = root / "calibration_records.jsonl"
            plan_dir = root / "capture-plan"
            layer = "transformer_blocks.0.attn.to_q"
            self.save_file({f"{layer}.weight": torch.zeros((128, 128), dtype=torch.float16)}, str(source))
            records.write_text(json.dumps({"case_id": "case-1", "prompt": "make the jacket blue"}) + "\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                rc = main(
                    [
                        "calib",
                        "plan-int4-capture",
                        "--family",
                        "qwen_image_edit",
                        "--source",
                        str(source),
                        "--records",
                        str(records),
                        "--out",
                        str(plan_dir),
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)

            with self.assertRaisesRegex(PayloadWriteError, "channel count mismatch"):
                write_int4_activation_case_safetensors(
                    plan=plan_dir / "capture_plan.json",
                    case_id="case-1",
                    tensors={f"{layer}.input": torch.zeros((1, 127), dtype=torch.float32)},
                    out_dir=root / "capture-run",
                )

    def test_cli_plan_int4_capture_rejects_unsupported_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "empty.safetensors"
            records = root / "records.jsonl"
            out = root / "capture-plan"
            source.write_bytes(b"")
            records.write_text(json.dumps({"case_id": "case-1", "prompt": "prompt"}) + "\n", encoding="utf-8")

            stderr = StringIO()
            with redirect_stderr(stderr):
                rc = main(
                    [
                        "calib",
                        "plan-int4-capture",
                        "--family",
                        "qwen_image",
                        "--source",
                        str(source),
                        "--records",
                        str(records),
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(rc, 2)
            self.assertIn("unsupported INT4 activation-capture family", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
