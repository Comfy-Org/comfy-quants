import tempfile
import unittest
from pathlib import Path

from comfy_quants.core.errors import ManifestError
from comfy_quants.core.manifest import ArtifactManifest, create_minimal_manifest


class TestManifest(unittest.TestCase):
    def test_minimal_manifest_roundtrip(self):
        manifest = create_minimal_manifest(
            artifact_id="test-artifact",
            family="qwen_image",
            model_id="Qwen/Qwen-Image-2512",
            revision="abc",
            algorithm="fp8_static",
            target_dtype="fp8_e4m3",
            compatibility_level="L0",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            manifest.save(path)
            loaded = ArtifactManifest.load(path)
        self.assertEqual(loaded.artifact_id, "test-artifact")
        self.assertEqual(loaded.compatibility["level"], "L0")

    def test_manifest_requires_compatibility(self):
        with self.assertRaises(ManifestError):
            ArtifactManifest.validate_dict({"schema_version": "0.1.0"})


if __name__ == "__main__":
    unittest.main()
