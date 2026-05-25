"""Local artifact registry."""

from __future__ import annotations

from pathlib import Path

from comfy_quants.utils.jsonio import read_json, write_json


class LocalArtifactRegistry:
    """Simple JSON-backed local artifact index."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "artifact_index.json"
        if not self.index_path.exists():
            write_json(self.index_path, {"artifacts": []})

    def add(self, artifact_id: str, path: str) -> None:
        index = read_json(self.index_path)
        index["artifacts"].append({"artifact_id": artifact_id, "path": path})
        write_json(self.index_path, index)

    def list(self):
        return read_json(self.index_path)["artifacts"]
