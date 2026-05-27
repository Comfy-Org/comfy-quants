#!/usr/bin/env python3
"""Capture Qwen-Image-Edit INT4 calibration activations with an optional diffusers runtime.

This is a development harness.  It imports diffusers only when executed and is
not part of the package runtime.  The base library remains responsible only for
static capture plans, reducer manifests, and safetensors writers.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _repo_src_path() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


_src = _repo_src_path()
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from comfy_quants.backends.activation_capture.materialize import write_int4_activation_case_safetensors
from comfy_quants.calibration.datasets import load_calibration_cases
from comfy_quants.utils.jsonio import read_json, write_json


class CaptureHarnessError(RuntimeError):
    """Raised when the dev activation capture harness cannot continue."""


@dataclass
class CapturedTensorRecord:
    tensor: str
    layer: str
    shape: list[int]
    dtype: str
    rows: int
    call_count: int


@dataclass
class CaptureCaseReport:
    case_id: str
    status: str
    image: str
    prompt_chars: int
    output_file: str
    tensor_count: int
    missing_tensor_count: int
    rows_per_layer_cap: int
    captured: list[CapturedTensorRecord] = field(default_factory=list)
    peak_cuda_allocated_bytes: int | None = None
    peak_cuda_reserved_bytes: int | None = None


@dataclass
class CaptureRunReport:
    status: str
    model_root: str
    plan: str
    records: str
    out_dir: str
    pipeline_class: str
    device: str
    model_dtype: str
    storage_dtype: str
    num_inference_steps: int
    selected_target_count: int
    case_count: int
    written_case_count: int
    rows_per_layer_cap: int
    rows_per_call_cap: int
    schema_version: str = "qwen_image_edit_int4_activation_capture_run.v1"
    cases: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dev environment dependent
        raise CaptureHarnessError("torch is required for this dev capture harness") from exc
    return torch


def _load_pil_image():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dev environment dependent
        raise CaptureHarnessError("Pillow is required to load calibration images") from exc
    return Image


def _dtype_from_name(name: str, *, torch: Any, device: Any):
    value = str(name).lower()
    if value == "auto":
        if getattr(device, "type", str(device)) == "cuda":
            return torch.bfloat16
        return torch.float32
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise CaptureHarnessError(f"unsupported dtype {name!r}")


def _resolve_device(name: str):
    torch = _load_torch()
    requested = str(name or "auto")
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise CaptureHarnessError(f"CUDA device requested but unavailable: {requested}")
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return device


def _pipeline_class_from_model_index(model_root: Path) -> str | None:
    model_index = model_root / "model_index.json"
    if not model_index.is_file():
        return None
    data = json.loads(model_index.read_text(encoding="utf-8"))
    value = data.get("_class_name") if isinstance(data, dict) else None
    return value if isinstance(value, str) and value else None


def _load_pipeline_class(class_name: str):
    requested = class_name
    if requested == "auto":
        raise CaptureHarnessError("internal error: auto pipeline class was not resolved")
    try:
        from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline
    except ImportError as exc:  # pragma: no cover - dev environment dependent
        raise CaptureHarnessError(
            "diffusers with Qwen-Image pipeline support is required for this dev harness"
        ) from exc
    mapping = {
        "QwenImageEditPipeline": QwenImageEditPipeline,
        "QwenImageEditPlusPipeline": QwenImageEditPlusPipeline,
        "edit": QwenImageEditPipeline,
        "edit_plus": QwenImageEditPlusPipeline,
    }
    if requested not in mapping:
        valid = ", ".join(sorted(mapping))
        raise CaptureHarnessError(f"unsupported pipeline class {requested!r}; expected one of: {valid}")
    return mapping[requested]


def _load_cases(records: Path, *, image_root: Path | None, limit: int | None, case_ids: set[str] | None):
    cases = load_calibration_cases(records, image_root=image_root, limit=limit)
    if case_ids:
        cases = [case for case in cases if case.case_id in case_ids]
    if not cases:
        raise CaptureHarnessError("no calibration cases selected")
    return cases


def _resolve_case_image(case: Any, *, fallback_image: Path | None):
    Image = _load_pil_image()
    image_value = getattr(case, "image", None)
    if image_value:
        image_path = Path(image_value).expanduser()
        if image_path.is_file():
            return str(image_path), Image.open(image_path).convert("RGB")
    if fallback_image is not None:
        if not fallback_image.is_file():
            raise CaptureHarnessError(f"fallback image does not exist: {fallback_image}")
        return str(fallback_image), Image.open(fallback_image).convert("RGB")
    raise CaptureHarnessError(f"case {case.case_id!r} does not reference an existing image; pass --image-root or --fallback-image")


def _flatten_channel_last(tensor: Any, *, channel_dim: int):
    torch = _load_torch()
    if not torch.is_tensor(tensor):
        raise CaptureHarnessError("hook input is not a torch tensor")
    if int(tensor.ndim) == 0:
        raise CaptureHarnessError("hook input must have at least one dimension")
    dim = int(channel_dim)
    if dim < 0:
        dim += int(tensor.ndim)
    if dim < 0 or dim >= int(tensor.ndim):
        raise CaptureHarnessError(f"channel_dim {channel_dim} is out of range for shape {tuple(tensor.shape)}")
    value = tensor.detach()
    if dim != int(value.ndim) - 1:
        value = value.movedim(dim, -1)
    return value.reshape(-1, int(value.shape[-1]))


def _evenly_spaced_indices(row_count: int, take: int, *, device: Any):
    torch = _load_torch()
    if take >= row_count:
        return None
    if take <= 0:
        raise CaptureHarnessError("sample size must be positive")
    if take == 1:
        return torch.zeros((1,), device=device, dtype=torch.long)
    return torch.linspace(0, row_count - 1, steps=take, device=device).round().to(dtype=torch.long)


def _first_nonfinite_tensor(tensors: dict[str, Any]) -> dict[str, Any] | None:
    torch = _load_torch()
    for name in sorted(tensors):
        tensor = tensors[name]
        if not torch.is_tensor(tensor):
            continue
        finite = torch.isfinite(tensor)
        if bool(finite.all().item()):
            continue
        bad = ~finite
        return {
            "tensor": name,
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "nonfinite_count": int(bad.sum().item()),
            "element_count": int(tensor.numel()),
        }
    return None


class ActivationCollector:
    def __init__(
        self,
        *,
        targets_by_source: dict[str, dict[str, Any]],
        storage_dtype: Any,
        max_rows_per_layer: int,
        max_rows_per_call: int,
    ) -> None:
        self.targets_by_source = targets_by_source
        self.storage_dtype = storage_dtype
        self.max_rows_per_layer = int(max_rows_per_layer)
        self.max_rows_per_call = int(max_rows_per_call)
        self.enabled = False
        self._chunks: dict[str, list[Any]] = {}
        self._rows: dict[str, int] = {}
        self._calls: dict[str, int] = {}

    def reset(self) -> None:
        self._chunks.clear()
        self._rows.clear()
        self._calls.clear()

    def hook_for(self, source_prefix: str):
        target = self.targets_by_source[source_prefix]
        tensor_name = str(target["capture_tensor_name"])
        layer_name = str(target["output_prefix"])
        channel_dim = int(target.get("channel_dim", -1))
        expected_channels = int(target["input_channels"])

        def _hook(_module, inputs, _output):
            if not self.enabled:
                return
            if not inputs:
                return
            rows = _flatten_channel_last(inputs[0], channel_dim=channel_dim)
            if int(rows.shape[1]) != expected_channels:
                raise CaptureHarnessError(
                    f"captured {tensor_name!r} has {int(rows.shape[1])} channels, expected {expected_channels}"
                )
            existing = int(self._rows.get(tensor_name, 0))
            if self.max_rows_per_layer > 0 and existing >= self.max_rows_per_layer:
                return
            take = int(rows.shape[0])
            if self.max_rows_per_call > 0:
                take = min(take, self.max_rows_per_call)
            if self.max_rows_per_layer > 0:
                take = min(take, self.max_rows_per_layer - existing)
            if take <= 0:
                return
            index = _evenly_spaced_indices(int(rows.shape[0]), take, device=rows.device)
            if index is not None:
                rows = rows.index_select(0, index)
            rows = rows.to(device="cpu", dtype=self.storage_dtype).contiguous()
            self._chunks.setdefault(tensor_name, []).append(rows)
            self._rows[tensor_name] = existing + int(rows.shape[0])
            self._calls[tensor_name] = int(self._calls.get(tensor_name, 0)) + 1

        _hook.__name__ = f"capture_{layer_name.replace('.', '_')}"
        return _hook

    def tensors(self) -> dict[str, Any]:
        torch = _load_torch()
        out = {}
        for name, chunks in self._chunks.items():
            if not chunks:
                continue
            out[name] = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0).contiguous()
        return out

    def records(self) -> list[CapturedTensorRecord]:
        result: list[CapturedTensorRecord] = []
        tensors = self.tensors()
        target_by_tensor = {str(t["capture_tensor_name"]): t for t in self.targets_by_source.values()}
        for name in sorted(tensors):
            tensor = tensors[name]
            target = target_by_tensor[name]
            result.append(
                CapturedTensorRecord(
                    tensor=name,
                    layer=str(target["output_prefix"]),
                    shape=[int(dim) for dim in tensor.shape],
                    dtype=str(tensor.dtype).replace("torch.", ""),
                    rows=int(tensor.reshape(-1, int(tensor.shape[-1])).shape[0]),
                    call_count=int(self._calls.get(name, 0)),
                )
            )
        return result


def _register_hooks(transformer: Any, plan: dict[str, Any], collector: ActivationCollector, *, allow_missing_modules: bool):
    modules = dict(transformer.named_modules())
    handles = []
    missing = []
    for target in plan["targets"]:
        source = str(target["source_prefix"])
        module = modules.get(source)
        if module is None:
            missing.append(source)
            continue
        handles.append(module.register_forward_hook(collector.hook_for(source)))
    if missing and not allow_missing_modules:
        first = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise CaptureHarnessError(f"transformer is missing planned modules: {first}{suffix}")
    return handles, missing


def _run_case(
    *,
    pipe: Any,
    case: Any,
    plan_path: Path,
    plan: dict[str, Any],
    out_dir: Path,
    collector: ActivationCollector,
    image_root: Path | None,
    fallback_image: Path | None,
    num_inference_steps: int,
    height: int | None,
    width: int | None,
    true_cfg_scale: float,
    negative_prompt: str | None,
    guidance_scale: float | None,
    seed: int,
    max_sequence_length: int,
) -> CaptureCaseReport:
    torch = _load_torch()
    image_label, image = _resolve_case_image(case, fallback_image=fallback_image)
    prompt = str(case.prompt)
    generator = None
    device = getattr(pipe, "_execution_device", None)
    if seed >= 0:
        generator_device = "cuda" if device is not None and getattr(device, "type", str(device)) == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    collector.reset()
    collector.enabled = True
    try:
        with torch.inference_mode():
            pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=float(true_cfg_scale),
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=int(num_inference_steps),
                generator=generator,
                output_type="latent",
                return_dict=True,
                max_sequence_length=int(max_sequence_length),
            )
    finally:
        collector.enabled = False

    tensors = collector.tensors()
    nonfinite = _first_nonfinite_tensor(tensors)
    if nonfinite is not None:
        raise CaptureHarnessError(
            "captured activations contain NaN or Inf values; "
            f"first_bad={json.dumps(nonfinite, ensure_ascii=False, sort_keys=True)}. "
            "Increase --num-inference-steps or check the external pipeline scheduler configuration."
        )
    write_report = write_int4_activation_case_safetensors(
        plan=plan,
        case_id=case.case_id,
        tensors=tensors,
        out_dir=out_dir,
        allow_missing=False,
    )
    peak_alloc = peak_reserved = None
    if torch.cuda.is_available():
        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    return CaptureCaseReport(
        case_id=case.case_id,
        status="captured",
        image=image_label,
        prompt_chars=len(prompt),
        output_file=write_report.output_file,
        tensor_count=write_report.tensor_count,
        missing_tensor_count=write_report.missing_tensor_count,
        rows_per_layer_cap=int(collector.max_rows_per_layer),
        captured=collector.records(),
        peak_cuda_allocated_bytes=peak_alloc,
        peak_cuda_reserved_bytes=peak_reserved,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, help="Local diffusers Qwen-Image-Edit model directory")
    parser.add_argument("--plan", required=True, help="capture_plan.json from comfy-quants calib plan-int4-capture")
    parser.add_argument("--records", help="Calibration records JSONL; defaults to records_path from the plan")
    parser.add_argument("--image-root", help="Base directory for relative record image paths")
    parser.add_argument("--fallback-image", help="Optional image used when a selected record image is missing")
    parser.add_argument("--out-dir", help="Capture-run directory; defaults to the plan directory")
    parser.add_argument("--report", help="Output JSON report path; defaults to <out-dir>/activation_capture_run_report.json")
    parser.add_argument("--pipeline-class", default="auto", help="auto, QwenImageEditPipeline, QwenImageEditPlusPipeline, edit, or edit_plus")
    parser.add_argument("--device", default="auto", help="auto uses cuda:0 when available")
    parser.add_argument("--model-dtype", default="auto", help="auto, bfloat16, float16, or float32")
    parser.add_argument("--storage-dtype", default="bfloat16", help="bfloat16, float16, or float32 for saved activations")
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-rows-per-layer", type=int, default=128, help="0 disables per-layer row cap")
    parser.add_argument("--max-rows-per-call", type=int, default=128, help="0 disables per-hook-call row cap")
    parser.add_argument("--limit", type=int, help="Maximum number of records to process")
    parser.add_argument("--case-id", action="append", help="Only capture the given case id; may be repeated")
    parser.add_argument("--allow-missing-modules", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate plan/records/images without loading diffusers")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_root = Path(args.model_root).expanduser()
    plan_path = Path(args.plan).expanduser()
    if not model_root.is_dir():
        raise CaptureHarnessError(f"model root does not exist: {model_root}")
    plan = read_json(plan_path)
    if not isinstance(plan, dict) or plan.get("schema_version") != "int4_activation_capture_plan.v1":
        raise CaptureHarnessError(f"unsupported capture plan: {plan_path}")
    targets = plan.get("targets")
    if not isinstance(targets, list) or not targets:
        raise CaptureHarnessError(f"capture plan has no targets: {plan_path}")

    records_path = Path(args.records or str(plan.get("records_path", ""))).expanduser()
    if not records_path.is_file():
        raise CaptureHarnessError(f"records file does not exist: {records_path}")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else plan_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root).expanduser() if args.image_root else None
    fallback_image = Path(args.fallback_image).expanduser() if args.fallback_image else None
    case_ids = set(args.case_id or []) or None
    cases = _load_cases(records_path, image_root=image_root, limit=args.limit, case_ids=case_ids)

    pipeline_class_name = args.pipeline_class
    if pipeline_class_name == "auto":
        pipeline_class_name = _pipeline_class_from_model_index(model_root) or "QwenImageEditPipeline"

    if args.dry_run:
        result = {
            "status": "dry_run_ok",
            "model_root": str(model_root),
            "plan": str(plan_path),
            "records": str(records_path),
            "out_dir": str(out_dir),
            "pipeline_class": pipeline_class_name,
            "selected_target_count": len(targets),
            "case_count": len(cases),
            "schema_version": "qwen_image_edit_int4_activation_capture_dry_run.v1",
        }
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    torch = _load_torch()
    device = _resolve_device(args.device)
    model_dtype = _dtype_from_name(args.model_dtype, torch=torch, device=device)
    storage_dtype = _dtype_from_name(args.storage_dtype, torch=torch, device=device)
    PipelineClass = _load_pipeline_class(pipeline_class_name)

    pipe = PipelineClass.from_pretrained(str(model_root), torch_dtype=model_dtype, local_files_only=True)
    pipe.to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    targets_by_source = {str(target["source_prefix"]): target for target in targets}
    collector = ActivationCollector(
        targets_by_source=targets_by_source,
        storage_dtype=storage_dtype,
        max_rows_per_layer=int(args.max_rows_per_layer),
        max_rows_per_call=int(args.max_rows_per_call),
    )

    case_reports: list[dict[str, Any]] = []
    with ExitStack() as stack:
        handles, missing_modules = _register_hooks(
            pipe.transformer,
            plan,
            collector,
            allow_missing_modules=bool(args.allow_missing_modules),
        )
        for handle in handles:
            stack.callback(handle.remove)
        if missing_modules:
            print(f"warning: skipped {len(missing_modules)} missing modules", file=sys.stderr)
        for index, case in enumerate(cases, start=1):
            print(f"capture case {index}/{len(cases)}: {case.case_id}", file=sys.stderr)
            case_report = _run_case(
                pipe=pipe,
                case=case,
                plan_path=plan_path,
                plan=plan,
                out_dir=out_dir,
                collector=collector,
                image_root=image_root,
                fallback_image=fallback_image,
                num_inference_steps=int(args.num_inference_steps),
                height=args.height,
                width=args.width,
                true_cfg_scale=float(args.true_cfg_scale),
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                seed=int(args.seed) + index - 1 if int(args.seed) >= 0 else -1,
                max_sequence_length=int(args.max_sequence_length),
            )
            case_reports.append(asdict(case_report))
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = CaptureRunReport(
        status="ok",
        model_root=str(model_root),
        plan=str(plan_path),
        records=str(records_path),
        out_dir=str(out_dir),
        pipeline_class=pipeline_class_name,
        device=str(device),
        model_dtype=str(model_dtype).replace("torch.", ""),
        storage_dtype=str(storage_dtype).replace("torch.", ""),
        num_inference_steps=int(args.num_inference_steps),
        selected_target_count=len(targets),
        case_count=len(cases),
        written_case_count=len(case_reports),
        rows_per_layer_cap=int(args.max_rows_per_layer),
        rows_per_call_cap=int(args.max_rows_per_call),
        cases=case_reports,
    )
    report_path = Path(args.report).expanduser() if args.report else out_dir / "activation_capture_run_report.json"
    write_json(report_path, report.to_dict())
    payload = report.to_dict()
    payload["report"] = str(report_path)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - dev script entry point
    try:
        raise SystemExit(main())
    except CaptureHarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
