"""Backend exporter protocol."""

from __future__ import annotations

from typing import Protocol

from comfy_quants.core.artifact import QuantArtifact


class BackendExporter(Protocol):
    backend_name: str
    version: str

    def check_compatibility(self, artifact: QuantArtifact) -> dict:
        """Return a compatibility report for the backend."""

    def export(self, artifact: QuantArtifact, output_dir: str) -> dict:
        """Export an artifact into backend-specific files."""
