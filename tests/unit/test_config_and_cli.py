import json
import tempfile
import unittest
from pathlib import Path

from comfy_quants.cli.main import main
from comfy_quants.core.config import load_quant_config


class TestConfigAndCli(unittest.TestCase):
    def test_load_sample_config(self):
        cfg = load_quant_config("configs/qwen_image_2512_fp8_static.yaml")
        self.assertEqual(cfg.model.family, "qwen_image")
        self.assertEqual(cfg.quant.algorithm, "fp8_static")
        self.assertEqual(cfg.hardware.gpu_profile, "rtx_pro_6000_blackwell_96gb")

    def test_inspect_and_quantize_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            inspect_dir = Path(tmp) / "inspect"
            rc = main([
                "inspect",
                "--model", "Qwen/Qwen-Image-2512",
                "--family", "qwen_image",
                "--out", str(inspect_dir),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((inspect_dir / "model_inspection.json").exists())
            work_dir = Path(tmp) / "job"
            rc = main([
                "quantize",
                "--config", "configs/qwen_image_2512_fp8_static.yaml",
                "--work-dir", str(work_dir),
                "--dry-run",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((work_dir / "job.json").exists())
            self.assertTrue((work_dir / "artifact" / "manifest.json").exists())
            plan = json.loads((work_dir / "plan.json").read_text())
            self.assertGreater(len(plan["steps"]), 0)


if __name__ == "__main__":
    unittest.main()
