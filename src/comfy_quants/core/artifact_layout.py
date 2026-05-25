"""Artifact file layout declarations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ArtifactPayloadLayout:
    """Paths used by tensor metadata and payload files inside an artifact."""

    schema_version: str = "artifact_payload_layout.v1"
    tensor_index_path: str = "quant_tensor_index.json"
    weight_payload_path: str = "tensors/fp8_weights.safetensors"
    scale_payload_path: str = "scales/fp8_static_scales.safetensors"
    high_precision_payload_path: str = "tensors/bf16_kept.safetensors"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def manifest_index_record(self) -> dict[str, str]:
        return {
            "path": self.tensor_index_path,
            "kind": "quant_tensor_index",
            "state": "metadata_only",
        }


DEFAULT_ARTIFACT_PAYLOAD_LAYOUT = ArtifactPayloadLayout()
