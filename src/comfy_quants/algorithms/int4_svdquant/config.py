"""Configuration objects for the SVDQuant W4A4 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from comfy_quants.formats.kitchen_tilepack import KITCHEN_GROUP_SIZE, SVDQUANT_W4A4_FORMAT_NAME
from comfy_quants.formats.svdquant_w4a4 import (
    DEFAULT_LOWRANK_BRANCH_INPUT_BASIS,
    LOWRANK_BRANCH_INPUT_BASIS_POST_SMOOTHING,
    LOWRANK_BRANCH_INPUT_BASIS_RAW,
)


WEIGHT_ONLY_MODE = "weight_only_initialization"
CALIBRATED_SVDQUANT_MODE = "calibrated_svdquant"
SVDQUANT_GPTQ_EXPERIMENTAL_MODE = "svdquant_gptq_experimental"
SUPPORTED_QUANTIZATION_MODES = (WEIGHT_ONLY_MODE, CALIBRATED_SVDQUANT_MODE, SVDQUANT_GPTQ_EXPERIMENTAL_MODE)

LOWRANK_CALIBRATION_WEIGHT_RESIDUAL = "weight_residual"
LOWRANK_CALIBRATION_OUTPUT_ERROR = "output_error"
SUPPORTED_LOWRANK_CALIBRATION_MODES = (LOWRANK_CALIBRATION_WEIGHT_RESIDUAL, LOWRANK_CALIBRATION_OUTPUT_ERROR)

WEIGHT_ONLY_INITIALIZATION_STATE = "weight_only_initialization_no_calibration_no_gptq"
EXPERIMENTAL_SMOOTH_RTN_SVD_NO_GPTQ_STATE = "experimental_smooth_rtn_svd_no_gptq"
EXPERIMENTAL_SVDQUANT_GPTQ_NO_AWQ_STATE = "experimental_svdquant_gptq_no_awq_runtime_unverified"
EXPERIMENTAL_SVDQUANT_GPTQ_AWQ_RUNTIME_UNVERIFIED_STATE = "experimental_svdquant_gptq_awq_runtime_unverified"
PUBLISHABLE_SVDQUANT_GPTQ_STATE = "svdquant_w4a4_gptq"
GPTQ_STATE_NOT_IMPLEMENTED = "not_implemented"
GPTQ_STATE_LAYER_CORE_INTEGRATED = "layer_core_integrated"
RUNTIME_CONTRACT_STATIC_ARTIFACT_ONLY = "static_artifact_contract_only"
MIXED_QUANTIZATION_SVD_ONLY_STATE = "svdquant_only_awq_modulation_not_implemented"
MIXED_QUANTIZATION_SVD_AWQ_EXPERIMENTAL_STATE = "experimental_svdquant_w4a4_awq_w4a16_runtime_unverified"


def algorithm_state_for_quantization_mode(mode: str) -> str:
    """Return the implementation state represented by a public CLI mode."""
    if mode == WEIGHT_ONLY_MODE:
        return WEIGHT_ONLY_INITIALIZATION_STATE
    if mode == CALIBRATED_SVDQUANT_MODE:
        return EXPERIMENTAL_SMOOTH_RTN_SVD_NO_GPTQ_STATE
    if mode == SVDQUANT_GPTQ_EXPERIMENTAL_MODE:
        return EXPERIMENTAL_SVDQUANT_GPTQ_NO_AWQ_STATE
    raise ValueError(f"unsupported quantization mode: {mode}")


def is_publishable_svdquant_gptq_state(state: str) -> bool:
    """Return whether a state represents the full mixed SVDQuant+GPTQ target."""
    return state == PUBLISHABLE_SVDQUANT_GPTQ_STATE


def algorithm_notes_for_quantization_mode(mode: str) -> list[str]:
    """Return human-readable caveats for reports and dry-run plans."""
    state = algorithm_state_for_quantization_mode(mode)
    if state == WEIGHT_ONLY_INITIALIZATION_STATE:
        return [
            "weight_only_initialization emits the target tensor names and tile-packed layout without calibration.",
            "It uses groupwise signed-INT4 round-to-nearest weights, identity smoothing, and a zero low-rank branch.",
            "It does not implement GPTQ or activation W4 runtime calibration.",
        ]
    if state == EXPERIMENTAL_SMOOTH_RTN_SVD_NO_GPTQ_STATE:
        return [
            "calibrated_svdquant currently consumes per-layer activation statistics for smoothing.",
            "After smoothing it uses groupwise signed-INT4 round-to-nearest and an SVD residual branch.",
            "This RTN milestone mode intentionally does not feed the GPTQ/Hessian weight solve.",
            "The Qwen mixed AWQ W4A16 modulation branch is emitted when modulation tensors are present.",
            "The external runtime full-inference path remains unverified.",
        ]
    if state in {EXPERIMENTAL_SVDQUANT_GPTQ_NO_AWQ_STATE, EXPERIMENTAL_SVDQUANT_GPTQ_AWQ_RUNTIME_UNVERIFIED_STATE}:
        return [
            "svdquant_gptq_experimental consumes activation statistics plus precomputed raw-input GPTQ Hessians.",
            "It applies the smoothing-basis Hessian transform before invoking the repo-native GPTQ layer core.",
            "The low-rank branch defaults to weight-space residual calibration; output-error branch calibration is experimental and requires activation samples.",
            "The Qwen mixed AWQ W4A16 modulation branch is emitted when modulation tensors are present.",
            "The external runtime full-inference path remains unverified.",
        ]
    return []


@dataclass(frozen=True)
class Int4SvdquantPipelineConfig:
    """Pipeline options for producing a tile-packed SVDQuant checkpoint."""

    model_family: str = "qwen_image_edit"
    target_format: str = SVDQUANT_W4A4_FORMAT_NAME
    group_size: int = KITCHEN_GROUP_SIZE
    rank: int = 64
    scale_dtype: str = "source"
    calibration_path: str | None = None
    activation_stats_path: str | None = None
    quantization_mode: str = WEIGHT_ONLY_MODE
    smooth_alpha: float = 0.5
    smooth_min: float = 1.0 / 64.0
    smooth_max: float = 64.0
    gptq_hessian_stats_path: str | None = None
    gptq_damp_percentage: float = 0.01
    gptq_block_size: int = 128
    gptq_num_inv_tries: int = 250
    gptq_hessian_block_size: int = 512
    lowrank_branch_input_basis: str = DEFAULT_LOWRANK_BRANCH_INPUT_BASIS
    activation_samples_path: str | None = None
    activation_samples_input_root: str | None = None
    lowrank_calibration: str = LOWRANK_CALIBRATION_WEIGHT_RESIDUAL
    lowrank_ridge: float = 1.0e-6

    def validate(self) -> None:
        """Validate options that affect tensor shapes and output contracts."""
        if self.model_family != "qwen_image_edit":
            raise ValueError(f"unsupported INT4 model family: {self.model_family}")
        if self.target_format != SVDQUANT_W4A4_FORMAT_NAME:
            raise ValueError(f"unsupported INT4 target format: {self.target_format}")
        if self.group_size != KITCHEN_GROUP_SIZE:
            raise ValueError(f"SVDQuant kitchen tile-pack requires group size {KITCHEN_GROUP_SIZE}")
        if int(self.rank) <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.scale_dtype not in {"source", "float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported scale dtype: {self.scale_dtype}")
        if self.calibration_path is not None and not Path(self.calibration_path).expanduser().exists():
            raise ValueError(f"calibration path does not exist: {self.calibration_path}")
        if self.activation_stats_path is not None and not Path(self.activation_stats_path).expanduser().exists():
            raise ValueError(f"activation stats path does not exist: {self.activation_stats_path}")
        if self.gptq_hessian_stats_path is not None and not Path(self.gptq_hessian_stats_path).expanduser().exists():
            raise ValueError(f"GPTQ Hessian stats path does not exist: {self.gptq_hessian_stats_path}")
        if self.activation_samples_path is not None and not Path(self.activation_samples_path).expanduser().exists():
            raise ValueError(f"activation samples path does not exist: {self.activation_samples_path}")
        if self.activation_samples_input_root is not None and not Path(self.activation_samples_input_root).expanduser().exists():
            raise ValueError(f"activation samples input root does not exist: {self.activation_samples_input_root}")
        if self.quantization_mode not in SUPPORTED_QUANTIZATION_MODES:
            raise ValueError(f"unsupported quantization mode: {self.quantization_mode}")
        if self.lowrank_calibration not in SUPPORTED_LOWRANK_CALIBRATION_MODES:
            raise ValueError(f"unsupported low-rank calibration mode: {self.lowrank_calibration}")
        if self.lowrank_calibration == LOWRANK_CALIBRATION_OUTPUT_ERROR and self.quantization_mode != SVDQUANT_GPTQ_EXPERIMENTAL_MODE:
            raise ValueError("output-error low-rank calibration is only supported with svdquant_gptq_experimental")
        if self.lowrank_calibration == LOWRANK_CALIBRATION_OUTPUT_ERROR and self.activation_samples_path is None:
            raise ValueError("output-error low-rank calibration requires an activation sample manifest")
        if self.quantization_mode == CALIBRATED_SVDQUANT_MODE and self.activation_stats_path is None:
            raise ValueError("calibrated_svdquant requires an activation stats file")
        if self.quantization_mode == SVDQUANT_GPTQ_EXPERIMENTAL_MODE:
            if self.activation_stats_path is None:
                raise ValueError("svdquant_gptq_experimental requires an activation stats file")
            if self.gptq_hessian_stats_path is None:
                raise ValueError("svdquant_gptq_experimental requires a GPTQ Hessian stats manifest")
        if not (0.0 <= float(self.smooth_alpha) <= 1.0):
            raise ValueError(f"smooth_alpha must be in [0, 1], got {self.smooth_alpha}")
        if float(self.smooth_min) <= 0 or float(self.smooth_max) <= 0 or float(self.smooth_min) > float(self.smooth_max):
            raise ValueError(f"invalid smooth clamp range: min={self.smooth_min}, max={self.smooth_max}")
        if float(self.gptq_damp_percentage) < 0.0:
            raise ValueError(f"gptq_damp_percentage must be non-negative, got {self.gptq_damp_percentage}")
        if int(self.gptq_block_size) <= 0:
            raise ValueError(f"gptq_block_size must be positive, got {self.gptq_block_size}")
        if int(self.gptq_num_inv_tries) < 0:
            raise ValueError(f"gptq_num_inv_tries must be non-negative, got {self.gptq_num_inv_tries}")
        if int(self.gptq_hessian_block_size) == 0:
            raise ValueError("gptq_hessian_block_size must be positive or negative")
        if self.lowrank_branch_input_basis not in {LOWRANK_BRANCH_INPUT_BASIS_POST_SMOOTHING, LOWRANK_BRANCH_INPUT_BASIS_RAW}:
            raise ValueError(f"unsupported low-rank branch input basis: {self.lowrank_branch_input_basis}")
        if float(self.lowrank_ridge) < 0.0:
            raise ValueError(f"lowrank_ridge must be non-negative, got {self.lowrank_ridge}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)
