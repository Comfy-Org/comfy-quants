"""Qwen-Image-Edit-2511 INT4 tile-pack export pipeline.

The runner resolves local tool paths, executes the search/PTQ and conversion
steps, then writes and inspects the final ComfyUI-loadable artifact.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from comfy_quants.backends.int4_artifact_inspect import inspect_svdquant_w4a4_artifact
from comfy_quants.core.errors import ConfigurationError, PayloadWriteError
from comfy_quants.utils.hashing import hash_file
from comfy_quants.utils.jsonio import write_json, write_yaml


DEFAULT_DEEPCOMPRESSOR_ROOT = Path(os.environ.get("COMFY_QUANTS_DEEPCOMPRESSOR_ROOT", "DeepCompressor"))
DEFAULT_NUNCHAKU_ROOT = Path(os.environ.get("COMFY_QUANTS_NUNCHAKU_ROOT", "nunchaku"))
DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_SEARCH_STRENGTH = "quality-r64"
DEFAULT_CALIBRATION_SAMPLES = 128
DEFAULT_GPUS = "0"
DEFAULT_AWQ_GROUP_SIZE = 64
DEFAULT_ROUTE = "nunchaku-bridge"
DEFAULT_NUNCHAKU_MODEL_CLASS = "NunchakuQwenImageTransformer2DModel"

DEFAULT_CALIBRATION_RELATIVE_PATH = Path(
    "datasets/torch.bfloat16/qwen-image-edit-2511/fmeuler50-g4.0/qdiff/s128"
)

MODEL_CFG = "examples/diffusion/configs/model/qwen-image-edit-2511.yaml"
INT4_CFG = "examples/diffusion/configs/svdquant/int4.yaml"
GENERATED_SEARCH_CONFIG_ROOT = Path(".tmp/generated-configs/qwen-image-edit-2511")

SEARCH_STRENGTHS = (
    "fast-r32",
    "fast-r64",
    "fast-r128",
    "balanced-r32",
    "balanced-r32-i64",
    "balanced-r128",
    "balanced-r32-b125",
    "mid-r32",
    "mid-r128",
    "quality-r32",
    "quality-r64",
    "quality-r64-b175",
    "quality-r96",
    "quality-r96-b175",
    "quality-r96-i128",
    "quality-r128",
    "quality-r128-b15",
)

ROUTES = ("nunchaku-bridge", "deepcompressor-import")

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class PipelineCommand:
    """A subprocess command with only the environment deltas recorded."""

    label: str
    cwd: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)

    def shell_display(self) -> str:
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.env.items())
        command = " ".join(shlex.quote(str(item)) for item in self.args)
        return f"{env_prefix} {command}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cwd": self.cwd,
            "args": self.args,
            "env": self.env,
            "shell": self.shell_display(),
        }


@dataclass(frozen=True)
class QwenImageEditInt4PipelineConfig:
    """Configuration for the Qwen-Image-Edit-2511 INT4 tile-pack flow."""

    output: Path
    base_checkpoint: Path | None = None
    model_id: str = DEFAULT_MODEL_ID
    deepcompressor_root: Path = DEFAULT_DEEPCOMPRESSOR_ROOT
    nunchaku_root: Path = DEFAULT_NUNCHAKU_ROOT
    search_strength: str = DEFAULT_SEARCH_STRENGTH
    calibration_path: Path | None = None
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES
    gpus: str = DEFAULT_GPUS
    python_bin: str = "python"
    micromamba_env: Path | None = None
    runs_root: Path | None = None
    export_root: Path | None = None
    export_name: str | None = None
    quant_path: Path | None = None
    ptq_output_dirname: str | None = None
    route: str = DEFAULT_ROUTE
    raw_nunchaku: Path | None = None
    awq_group_size: int = DEFAULT_AWQ_GROUP_SIZE
    no_awq_modulation: bool = False
    reuse: bool = False
    hash_output: bool = False
    inspect_output: bool = True
    strict_inspect: bool = True
    dry_run: bool = False
    report: Path | None = None
    nunchaku_model_class: str = DEFAULT_NUNCHAKU_MODEL_CLASS

    def normalized(self) -> "ResolvedQwenImageEditInt4PipelineConfig":
        output = _local_path(self.output)
        deep_root = _local_path(self.deepcompressor_root)
        nunchaku_root = _local_path(self.nunchaku_root)
        base_checkpoint = _local_path(self.base_checkpoint) if self.base_checkpoint is not None else None
        quant_path = _local_path(self.quant_path) if self.quant_path is not None else None
        runs_root = _local_path(self.runs_root) if self.runs_root is not None else deep_root / "runs"
        export_root = _local_path(self.export_root) if self.export_root is not None else _default_export_root(output)
        export_name = self.export_name or f"qwen-image-edit-2511-{self.search_strength}-s{self.calibration_samples}-gptq"
        raw_nunchaku = _local_path(self.raw_nunchaku) if self.raw_nunchaku is not None else export_root / f"{export_name}-raw-nunchaku-int4.safetensors"
        report = _local_path(self.report) if self.report is not None else output.with_suffix(".pipeline_report.json")

        calibration_path_was_default = self.calibration_path is None
        calibration_path = (
            deep_root / DEFAULT_CALIBRATION_RELATIVE_PATH
            if self.calibration_path is None
            else _local_path(self.calibration_path)
        )
        ptq_output_dirname = self.ptq_output_dirname or _default_ptq_output_dirname(
            search_strength=self.search_strength,
            calibration_samples=self.calibration_samples,
            calibration_path_was_default=calibration_path_was_default,
        )

        resolved = ResolvedQwenImageEditInt4PipelineConfig(
            output=output,
            base_checkpoint=base_checkpoint,
            model_id=self.model_id,
            deepcompressor_root=deep_root,
            nunchaku_root=nunchaku_root,
            search_strength=self.search_strength,
            calibration_path=calibration_path,
            calibration_path_was_default=calibration_path_was_default,
            calibration_samples=int(self.calibration_samples),
            gpus=str(self.gpus),
            python_bin=str(self.python_bin),
            micromamba_env=_local_path(self.micromamba_env) if self.micromamba_env is not None else None,
            runs_root=runs_root,
            export_root=export_root,
            export_name=export_name,
            quant_path=quant_path,
            ptq_output_dirname=ptq_output_dirname,
            route=self.route,
            raw_nunchaku=raw_nunchaku,
            awq_group_size=int(self.awq_group_size),
            no_awq_modulation=bool(self.no_awq_modulation),
            reuse=bool(self.reuse),
            hash_output=bool(self.hash_output),
            inspect_output=bool(self.inspect_output),
            strict_inspect=bool(self.strict_inspect),
            dry_run=bool(self.dry_run),
            report=report,
            nunchaku_model_class=str(self.nunchaku_model_class),
        )
        resolved.validate(static_only=self.dry_run)
        return resolved


@dataclass(frozen=True)
class ResolvedQwenImageEditInt4PipelineConfig:
    """Fully resolved paths and switches used by the runner."""

    output: Path
    base_checkpoint: Path | None
    model_id: str
    deepcompressor_root: Path
    nunchaku_root: Path
    search_strength: str
    calibration_path: Path
    calibration_path_was_default: bool
    calibration_samples: int
    gpus: str
    python_bin: str
    micromamba_env: Path | None
    runs_root: Path
    export_root: Path
    export_name: str
    quant_path: Path | None
    ptq_output_dirname: str
    route: str
    raw_nunchaku: Path
    awq_group_size: int
    no_awq_modulation: bool
    reuse: bool
    hash_output: bool
    inspect_output: bool
    strict_inspect: bool
    dry_run: bool
    report: Path
    nunchaku_model_class: str

    @property
    def run_ptq(self) -> bool:
        return self.quant_path is None

    @property
    def split_dir(self) -> Path:
        return self.export_root / self.export_name

    @property
    def ptq_override_path(self) -> Path:
        return self.export_root / "ptq-overrides" / f"{self.ptq_output_dirname}.yaml"

    def python_command(self) -> list[str]:
        if self.micromamba_env is not None:
            return ["micromamba", "run", "-p", str(self.micromamba_env), self.python_bin]
        return [self.python_bin]

    def validate(self, *, static_only: bool = False) -> None:
        if self.search_strength not in SEARCH_STRENGTHS:
            raise ConfigurationError(
                f"unsupported search strength: {self.search_strength}; expected one of {', '.join(SEARCH_STRENGTHS)}"
            )
        if self.route not in ROUTES:
            raise ConfigurationError(f"unsupported INT4 route: {self.route}; expected one of {', '.join(ROUTES)}")
        if self.calibration_samples <= 0:
            raise ConfigurationError("--calibration-samples must be a positive integer")
        if self.awq_group_size <= 0:
            raise ConfigurationError("--awq-group-size must be a positive integer")
        if self.route == "nunchaku-bridge" and self.base_checkpoint is None:
            raise ConfigurationError("--base-checkpoint is required for the nunchaku-bridge route")
        if static_only:
            return
        if self.run_ptq or self.route == "nunchaku-bridge":
            _require_dir(self.deepcompressor_root, "DeepCompressor root")
        if self.run_ptq:
            _require_file(self.deepcompressor_root / "deepcompressor/app/diffusion/ptq.py", "DeepCompressor PTQ module")
            _require_file(
                self.deepcompressor_root / "examples/diffusion/scripts/qwen_image_edit_2511_configs.py",
                "Qwen-Image-Edit-2511 config materializer",
            )
        if self.route == "nunchaku-bridge":
            _require_file(
                self.deepcompressor_root / "deepcompressor/backend/nunchaku/convert.py",
                "DeepCompressor Nunchaku converter",
            )
            _require_file(
                self.deepcompressor_root / "examples/diffusion/scripts/convert_kitchen_native.py",
                "kitchen-native converter",
            )
        if self.run_ptq:
            _require_dir(self.calibration_path, "calibration dataset")
        if self.route == "nunchaku-bridge":
            _require_dir(self.nunchaku_root, "Nunchaku root")
            _require_file(self.nunchaku_root / "nunchaku/merge_safetensors.py", "Nunchaku merge_safetensors module")
            _require_dir(self.nunchaku_root / "tools/kitchen_native", "Nunchaku kitchen-native helper directory")
            assert self.base_checkpoint is not None
            _require_file(self.base_checkpoint, "base BF16 transformer checkpoint")

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        data["run_ptq"] = self.run_ptq
        data["split_dir"] = str(self.split_dir)
        data["ptq_override_path"] = str(self.ptq_override_path)
        return data


@dataclass
class PipelineResult:
    """Serializable result for a pipeline run or dry-run plan."""

    status: str
    config: dict[str, Any]
    commands: list[dict[str, Any]]
    output: str
    report: str
    quant_path: str | None = None
    raw_nunchaku: str | None = None
    split_dir: str | None = None
    output_bytes: int | None = None
    output_hash: str | None = None
    inspection: dict[str, Any] | None = None
    started_at: float | None = None
    finished_at: float | None = None
    duration_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QwenImageEditInt4TilepackPipeline:
    """Runner for the external DeepCompressor/Nunchaku tile-pack flow."""

    def __init__(self, config: QwenImageEditInt4PipelineConfig | ResolvedQwenImageEditInt4PipelineConfig):
        self.config = config if isinstance(config, ResolvedQwenImageEditInt4PipelineConfig) else config.normalized()

    def plan(self) -> PipelineResult:
        commands = [cmd.to_dict() for cmd in self._planned_commands(include_ptq=self.config.run_ptq)]
        result = PipelineResult(
            status="dry_run_planned",
            config=self.config.to_public_dict(),
            commands=commands,
            output=str(self.config.output),
            report=str(self.config.report),
            quant_path=str(self.config.quant_path) if self.config.quant_path is not None else None,
            raw_nunchaku=str(self.config.raw_nunchaku),
            split_dir=str(self.config.split_dir),
        )
        write_json(self.config.report, result.to_dict())
        return result

    def run(self, *, progress: ProgressCallback | None = None) -> PipelineResult:
        cfg = self.config
        if cfg.dry_run:
            return self.plan()

        started = time.time()
        commands: list[PipelineCommand] = []
        quant_path = cfg.quant_path
        cfg.export_root.mkdir(parents=True, exist_ok=True)
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        cfg.raw_nunchaku.parent.mkdir(parents=True, exist_ok=True)

        if quant_path is None:
            self._emit(progress, stage="ptq", event="materialize_search_configs", search_strength=cfg.search_strength)
            materialize_cmd = self._materialize_command()
            commands.append(materialize_cmd)
            generated_configs = self._run_materialize(materialize_cmd)
            override_path = self._write_ptq_override()
            ptq_cmd = self._ptq_command(generated_configs=generated_configs, override_path=override_path)
            commands.append(ptq_cmd)
            self._emit(progress, stage="ptq", event="run", command=ptq_cmd.shell_display())
            self._run_command(ptq_cmd)
            run_dir = _find_latest_model_run(cfg.runs_root, cfg.ptq_output_dirname)
            quant_path = run_dir / "model"
            self._emit(progress, stage="ptq", event="done", quant_path=str(quant_path))

        _validate_quant_path(quant_path)

        if cfg.route == "deepcompressor-import":
            command = self._deepcompressor_import_command(quant_path)
            commands.append(command)
            self._emit(progress, stage="export", event="deepcompressor_import", command=command.shell_display())
            self._run_command(command)
        else:
            for command in self._nunchaku_bridge_commands(quant_path):
                commands.append(command)
                if self._should_skip_command(command):
                    self._emit(progress, stage="export", event="reuse", label=command.label)
                    continue
                self._emit(progress, stage="export", event=command.label, command=command.shell_display())
                self._run_command(command)

        if not cfg.output.exists():
            raise PayloadWriteError(f"pipeline finished but output checkpoint was not written: {cfg.output}")

        inspection: dict[str, Any] | None = None
        if cfg.inspect_output:
            self._emit(progress, stage="inspect", event="run", artifact=str(cfg.output))
            inspection_report = inspect_svdquant_w4a4_artifact(
                cfg.output,
                family="qwen_image_edit",
                strict_qwen_image_edit_2511=cfg.strict_inspect,
            )
            inspection = inspection_report.to_dict()
            if inspection_report.status != "ok":
                write_json(cfg.report, self._partial_result("failed_inspection", commands, quant_path, started, inspection).to_dict())
                raise PayloadWriteError(f"INT4 artifact inspection failed: {inspection_report.errors}")

        output_hash = hash_file(cfg.output) if cfg.hash_output else None
        finished = time.time()
        result = PipelineResult(
            status="ok",
            config=cfg.to_public_dict(),
            commands=[cmd.to_dict() for cmd in commands],
            output=str(cfg.output),
            report=str(cfg.report),
            quant_path=str(quant_path),
            raw_nunchaku=str(cfg.raw_nunchaku) if cfg.route == "nunchaku-bridge" else None,
            split_dir=str(cfg.split_dir) if cfg.route == "nunchaku-bridge" else None,
            output_bytes=cfg.output.stat().st_size,
            output_hash=output_hash,
            inspection=inspection,
            started_at=started,
            finished_at=finished,
            duration_sec=round(finished - started, 3),
        )
        write_json(cfg.report, result.to_dict())
        self._emit(progress, stage="done", event="ok", output=str(cfg.output), report=str(cfg.report))
        return result

    def _partial_result(
        self,
        status: str,
        commands: Sequence[PipelineCommand],
        quant_path: Path,
        started: float,
        inspection: dict[str, Any] | None,
    ) -> PipelineResult:
        finished = time.time()
        return PipelineResult(
            status=status,
            config=self.config.to_public_dict(),
            commands=[cmd.to_dict() for cmd in commands],
            output=str(self.config.output),
            report=str(self.config.report),
            quant_path=str(quant_path),
            raw_nunchaku=str(self.config.raw_nunchaku) if self.config.route == "nunchaku-bridge" else None,
            split_dir=str(self.config.split_dir) if self.config.route == "nunchaku-bridge" else None,
            output_bytes=self.config.output.stat().st_size if self.config.output.exists() else None,
            inspection=inspection,
            started_at=started,
            finished_at=finished,
            duration_sec=round(finished - started, 3),
        )

    def _planned_commands(self, *, include_ptq: bool) -> list[PipelineCommand]:
        cfg = self.config
        commands: list[PipelineCommand] = []
        quant_path = cfg.quant_path or Path(f"{cfg.runs_root}/<{cfg.ptq_output_dirname}>/run-<timestamp>/model")
        if include_ptq:
            generated_configs = [
                str(GENERATED_SEARCH_CONFIG_ROOT / "search.base.yaml"),
                str(GENERATED_SEARCH_CONFIG_ROOT / "search.gptq-bf16.yaml"),
                str(GENERATED_SEARCH_CONFIG_ROOT / f"search.{cfg.search_strength}.yaml"),
            ]
            commands.append(self._materialize_command())
            commands.append(self._ptq_command(generated_configs=generated_configs, override_path=cfg.ptq_override_path))
        if cfg.route == "deepcompressor-import":
            commands.append(self._deepcompressor_import_command(quant_path))
        else:
            commands.extend(self._nunchaku_bridge_commands(quant_path))
        return commands

    def _base_env(self) -> dict[str, str]:
        cfg = self.config
        return {
            "PYTHONPATH": _prepend_path(str(cfg.deepcompressor_root), os.environ.get("PYTHONPATH", "")),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            "QWEN_IMAGE_EDIT_2511_MODEL_PATH": cfg.model_id,
            "QWEN_IMAGE_EDIT_2511_SEARCH_CALIB_PATH": str(cfg.calibration_path),
        }

    def _materialize_command(self) -> PipelineCommand:
        cfg = self.config
        return PipelineCommand(
            label="materialize_search_configs",
            cwd=str(cfg.deepcompressor_root),
            env=self._base_env(),
            args=[
                *cfg.python_command(),
                "examples/diffusion/scripts/qwen_image_edit_2511_configs.py",
                "search",
                "launch",
                "--candidate",
                cfg.search_strength,
            ],
        )

    def _ptq_command(self, *, generated_configs: Sequence[str | Path], override_path: str | Path) -> PipelineCommand:
        cfg = self.config
        config_paths = [MODEL_CFG, INT4_CFG, *[str(item) for item in generated_configs], str(override_path)]
        env = self._base_env()
        env["CUDA_VISIBLE_DEVICES"] = cfg.gpus
        return PipelineCommand(
            label="deepcompressor_ptq",
            cwd=str(cfg.deepcompressor_root),
            env=env,
            args=[
                *cfg.python_command(),
                "-m",
                "deepcompressor.app.diffusion.ptq",
                *config_paths,
                "--save-model",
                "default",
                "--skip-gen",
                "true",
                "--skip-eval",
                "true",
            ],
        )

    def _deepcompressor_import_command(self, quant_path: Path) -> PipelineCommand:
        cfg = self.config
        env = {"PYTHONPATH": _prepend_path(str(_package_src_root()), os.environ.get("PYTHONPATH", ""))}
        args = [
            *cfg.python_command(),
            "-m",
            "comfy_quants.cli.main",
            "export-int4",
            "--format",
            "svdquant_w4a4",
            "--source-format",
            "deepcompressor-qwen-image-edit",
            "--source",
            str(quant_path),
            "--out",
            str(cfg.output),
            "--device",
            "auto",
            "--json",
        ]
        if cfg.hash_output:
            args.append("--hash-output")
        return PipelineCommand(label="deepcompressor_import_export", cwd=str(Path.cwd()), env=env, args=args)

    def _nunchaku_bridge_commands(self, quant_path: Path) -> list[PipelineCommand]:
        cfg = self.config
        commands: list[PipelineCommand] = []
        commands.append(
            PipelineCommand(
                label="nunchaku_convert",
                cwd=str(cfg.deepcompressor_root),
                env={"PYTHONPATH": _prepend_path(str(cfg.deepcompressor_root), os.environ.get("PYTHONPATH", ""))},
                args=[
                    *cfg.python_command(),
                    "-m",
                    "deepcompressor.backend.nunchaku.convert",
                    "--quant-path",
                    str(quant_path),
                    "--output-root",
                    str(cfg.export_root),
                    "--model-name",
                    cfg.export_name,
                    "--model-path",
                    cfg.model_id,
                ],
            )
        )
        commands.append(
            PipelineCommand(
                label="nunchaku_merge",
                cwd=str(cfg.deepcompressor_root),
                env={"PYTHONPATH": _prepend_path([str(cfg.nunchaku_root), str(cfg.deepcompressor_root)], os.environ.get("PYTHONPATH", ""))},
                args=[
                    *cfg.python_command(),
                    "-m",
                    "nunchaku.merge_safetensors",
                    "-i",
                    str(cfg.split_dir),
                    "-m",
                    cfg.nunchaku_model_class,
                    "-o",
                    str(cfg.raw_nunchaku),
                ],
            )
        )
        awq_args = ["--awq-group-size", str(cfg.awq_group_size)]
        if cfg.no_awq_modulation:
            awq_args.append("--no-awq-modulation")
        assert cfg.base_checkpoint is not None
        commands.append(
            PipelineCommand(
                label="kitchen_tilepack_convert",
                cwd=str(cfg.deepcompressor_root),
                env={
                    "NUNCHAKU_REPO_DIR": str(cfg.nunchaku_root),
                    "PYTHONPATH": _prepend_path([str(cfg.nunchaku_root), str(cfg.deepcompressor_root)], os.environ.get("PYTHONPATH", "")),
                },
                args=[
                    *cfg.python_command(),
                    "-m",
                    "examples.diffusion.scripts.convert_kitchen_native",
                    "--raw-nunchaku",
                    str(cfg.raw_nunchaku),
                    "--base-comfy",
                    str(cfg.base_checkpoint),
                    "--output",
                    str(cfg.output),
                    *awq_args,
                ],
            )
        )
        return commands

    def _write_ptq_override(self) -> Path:
        cfg = self.config
        payload: dict[str, Any] = {
            "output": {"dirname": cfg.ptq_output_dirname},
            "quant": {
                "calib": {
                    "path": str(cfg.calibration_path),
                    "num_samples": cfg.calibration_samples,
                }
            },
        }
        write_yaml(cfg.ptq_override_path, payload)
        return cfg.ptq_override_path

    def _run_materialize(self, command: PipelineCommand) -> list[str]:
        proc = self._run_command(command, capture_stdout=True)
        paths = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if len(paths) < 3:
            raise PayloadWriteError(f"DeepCompressor config materializer returned too few YAML paths: {paths}")
        return paths

    def _run_command(self, command: PipelineCommand, *, capture_stdout: bool = False) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(command.env)
        try:
            return subprocess.run(
                command.args,
                cwd=command.cwd,
                env=env,
                check=True,
                text=True,
                stdout=subprocess.PIPE if capture_stdout else None,
                stderr=None,
            )
        except subprocess.CalledProcessError as exc:
            raise PayloadWriteError(
                f"external command failed: label={command.label} exit_code={exc.returncode} command={command.shell_display()}"
            ) from exc

    def _should_skip_command(self, command: PipelineCommand) -> bool:
        cfg = self.config
        if not cfg.reuse:
            return False
        if command.label == "nunchaku_convert":
            return (cfg.split_dir / "transformer_blocks.safetensors").exists() and (cfg.split_dir / "unquantized_layers.safetensors").exists()
        if command.label == "nunchaku_merge":
            return cfg.raw_nunchaku.exists()
        if command.label == "kitchen_tilepack_convert":
            return cfg.output.exists()
        return False

    @staticmethod
    def _emit(progress: ProgressCallback | None, **event: Any) -> None:
        if progress is not None:
            progress(event)


def plan_qwen_image_edit_2511_int4_tilepack_pipeline(
    config: QwenImageEditInt4PipelineConfig,
) -> PipelineResult:
    """Return and write a dry-run plan for the one-step INT4 tile-pack flow."""

    return QwenImageEditInt4TilepackPipeline(config).plan()


def run_qwen_image_edit_2511_int4_tilepack_pipeline(
    config: QwenImageEditInt4PipelineConfig,
    *,
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    """Run DeepCompressor PTQ/search and export a single INT4 tile-pack file."""

    return QwenImageEditInt4TilepackPipeline(config).run(progress=progress)


def _default_export_root(output: Path) -> Path:
    return output.parent / f"{output.stem}.work"


def _default_ptq_output_dirname(*, search_strength: str, calibration_samples: int, calibration_path_was_default: bool) -> str:
    base = f"qwen-image-edit-2511-search-{search_strength}"
    if calibration_samples == DEFAULT_CALIBRATION_SAMPLES and calibration_path_was_default:
        return base
    suffix = f"s{calibration_samples}"
    if not calibration_path_was_default:
        suffix = f"customcalib-{suffix}"
    return f"{base}-{suffix}"


def _local_path(path: str | Path | None) -> Path:
    if path is None:
        raise ConfigurationError("internal error: expected a path, got None")
    return Path(os.path.expandvars(str(path))).expanduser()


def _package_src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepend_path(paths: str | Sequence[str], existing: str) -> str:
    if isinstance(paths, str):
        items = [paths]
    else:
        items = list(paths)
    if existing:
        items.append(existing)
    return os.pathsep.join(item for item in items if item)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ConfigurationError(f"missing {label}: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ConfigurationError(f"missing {label}: {path}")


def _validate_quant_path(path: Path) -> None:
    _require_dir(path, "DeepCompressor PTQ artifact directory")
    for filename in ("model.pt", "scale.pt", "smooth.pt", "branch.pt"):
        _require_file(path / filename, f"DeepCompressor PTQ {filename}")


def _find_latest_model_run(runs_root: Path, job_name: str) -> Path:
    candidates: list[tuple[float, Path]] = []
    if not runs_root.exists():
        raise PayloadWriteError(f"DeepCompressor runs root does not exist: {runs_root}")
    for model_file in runs_root.rglob("model.pt"):
        if model_file.parent.name != "model":
            continue
        run_dir = model_file.parent.parent
        parent_name = run_dir.parent.name
        if parent_name not in (job_name, f"{job_name}.RUNNING"):
            continue
        if run_dir.name.endswith(".RUNNING") or run_dir.name.endswith(".ERROR"):
            continue
        candidates.append((model_file.stat().st_mtime, run_dir))
    if not candidates:
        raise PayloadWriteError(f"PTQ finished but no model.pt was found for job {job_name} under {runs_root}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
