from __future__ import annotations

import json
from pathlib import Path

from comfy_quants.backends.qwen_image_edit_int4_pipeline import DEFAULT_CALIBRATION_RELATIVE_PATH
from comfy_quants.cli.main import main


def _load_stdout_json(capsys) -> dict:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def test_qwen_image_edit_int4_dry_run_uses_verified_defaults(tmp_path: Path, capsys) -> None:
    deep_root = tmp_path / "deepcompressor"
    nunchaku_root = tmp_path / "nunchaku"
    out = tmp_path / "qwen_edit_2511_int4.safetensors"
    report = tmp_path / "pipeline_report.json"

    code = main(
        [
            "qwen-image-edit-2511-int4",
            "--deepcompressor-root",
            str(deep_root),
            "--nunchaku-root",
            str(nunchaku_root),
            "--model",
            "/models/Qwen-Image-Edit-2511",
            "--base-checkpoint",
            str(tmp_path / "base.bf16.safetensors"),
            "--out",
            str(out),
            "--report",
            str(report),
            "--dry-run",
            "--json",
        ]
    )

    assert code == 0
    result = _load_stdout_json(capsys)
    assert result["status"] == "dry_run_planned"
    cfg = result["config"]
    assert cfg["run_ptq"] is True
    assert cfg["search_strength"] == "quality-r64"
    assert cfg["calibration_samples"] == 128
    assert cfg["calibration_path"] == str(deep_root / DEFAULT_CALIBRATION_RELATIVE_PATH)
    assert cfg["ptq_output_dirname"] == "qwen-image-edit-2511-search-quality-r64"
    assert cfg["output"] == str(out)

    labels = [command["label"] for command in result["commands"]]
    assert labels == [
        "materialize_search_configs",
        "deepcompressor_ptq",
        "nunchaku_convert",
        "nunchaku_merge",
        "kitchen_tilepack_convert",
    ]
    ptq_command = next(command for command in result["commands"] if command["label"] == "deepcompressor_ptq")
    assert "CUDA_VISIBLE_DEVICES" in ptq_command["env"]
    assert ptq_command["env"]["QWEN_IMAGE_EDIT_2511_SEARCH_CALIB_PATH"] == str(
        deep_root / DEFAULT_CALIBRATION_RELATIVE_PATH
    )
    assert "--save-model" in ptq_command["args"]
    assert report.exists()


def test_qwen_image_edit_int4_dry_run_allows_calib_and_strength_overrides(tmp_path: Path, capsys) -> None:
    quant_path = tmp_path / "ptq" / "model"
    out = tmp_path / "custom_int4.safetensors"
    custom_calib = tmp_path / "calib" / "qdiff-s64"

    code = main(
        [
            "quantize-qwen-image-edit-2511-int4",
            "--deepcompressor-root",
            str(tmp_path / "deepcompressor"),
            "--nunchaku-root",
            str(tmp_path / "nunchaku"),
            "--base-checkpoint",
            str(tmp_path / "base.safetensors"),
            "--out",
            str(out),
            "--quant-path",
            str(quant_path),
            "--search-strength",
            "fast-r64",
            "--calibration-path",
            str(custom_calib),
            "--calibration-samples",
            "64",
            "--gpus",
            "1",
            "--reuse",
            "--hash-output",
            "--no-inspect",
            "--dry-run",
            "--json",
        ]
    )

    assert code == 0
    result = _load_stdout_json(capsys)
    cfg = result["config"]
    assert cfg["run_ptq"] is False
    assert cfg["quant_path"] == str(quant_path)
    assert cfg["search_strength"] == "fast-r64"
    assert cfg["calibration_path"] == str(custom_calib)
    assert cfg["calibration_path_was_default"] is False
    assert cfg["calibration_samples"] == 64
    assert cfg["ptq_output_dirname"] == "qwen-image-edit-2511-search-fast-r64-customcalib-s64"
    assert cfg["reuse"] is True
    assert cfg["hash_output"] is True
    assert cfg["inspect_output"] is False

    labels = [command["label"] for command in result["commands"]]
    assert labels == ["nunchaku_convert", "nunchaku_merge", "kitchen_tilepack_convert"]
    assert all(command["label"] != "deepcompressor_ptq" for command in result["commands"])


def test_qwen_image_edit_int4_deepcompressor_import_route_does_not_require_base_checkpoint(
    tmp_path: Path, capsys
) -> None:
    quant_path = tmp_path / "ptq" / "model"
    out = tmp_path / "import_route.safetensors"

    code = main(
        [
            "qwen-image-edit-2511-int4",
            "--out",
            str(out),
            "--quant-path",
            str(quant_path),
            "--route",
            "deepcompressor-import",
            "--dry-run",
            "--json",
        ]
    )

    assert code == 0
    result = _load_stdout_json(capsys)
    assert result["config"]["base_checkpoint"] is None
    assert result["config"]["run_ptq"] is False
    labels = [command["label"] for command in result["commands"]]
    assert labels == ["deepcompressor_import_export"]


def test_qwen_image_edit_int4_rejects_invalid_calibration_samples(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "qwen-image-edit-2511-int4",
            "--deepcompressor-root",
            str(tmp_path / "deepcompressor"),
            "--nunchaku-root",
            str(tmp_path / "nunchaku"),
            "--base-checkpoint",
            str(tmp_path / "base.safetensors"),
            "--out",
            str(tmp_path / "out.safetensors"),
            "--calibration-samples",
            "0",
            "--dry-run",
            "--json",
        ]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert "--calibration-samples must be a positive integer" in captured.err
