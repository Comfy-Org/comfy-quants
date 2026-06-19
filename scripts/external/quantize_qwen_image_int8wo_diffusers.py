#!/usr/bin/env python
"""Reproduce a diffusers + torchao INT8 weight-only (``int8wo``) checkpoint.

This mirrors how ``dimitribarbot/Qwen-Image-Edit-int8wo`` was produced:

1. load a diffusers transformer (e.g. the ``transformer`` subfolder of
   ``Qwen/Qwen-Image-Edit``);
2. apply torchao ``Int8WeightOnlyConfig`` to every ``nn.Linear`` via diffusers'
   ``TorchAoConfig("int8wo")`` (symmetric per-output-channel int8:
   ``scale = amax / 127.5``, ``zero_point = 0``, weight ``int8 [out, in]``,
   activations stay high precision);
3. ``save_pretrained(..., safe_serialization=False)`` so the resulting torchao
   ``AffineQuantizedTensor`` subclasses are **pickled into ``.bin`` shards** —
   they cannot be stored in ``.safetensors`` (``save_file`` raises
   "invalid python storage" on a tensor subclass).

The output is a standard diffusers model folder:

    config.json                                  # carries the quantization_config
    diffusion_pytorch_model-0000N-of-0000M.bin   # pickled torchao subclasses
    diffusion_pytorch_model.bin.index.json

Load it back with ``AutoModel.from_pretrained(out, use_safetensors=False)``:
diffusers reads ``quantization_config``, unpickles the subclasses, and torchao
auto-dequantizes each weight to the compute dtype at runtime (it runs, but
slower than bf16).

NOTE: this is an *external-toolchain* wrapper (diffusers + torchao), kept under
``scripts/external/`` per the repo's layering rules. It is intentionally NOT part
of the ``comfy_quants`` library, which is safetensors-first and must not import
diffusers/torchao. The produced artifact is a diffusers/torchao model loaded by
``diffusers.from_pretrained`` (or a diffusers-based ComfyUI loader node) — not a
ComfyUI-native ``comfy_quant`` safetensors checkpoint.

Requirements:
    pip install "diffusers>=0.33" "torchao>=0.14" accelerate transformers
    plus access to the base model weights (e.g. ``Qwen/Qwen-Image-Edit``).

Example:
    python scripts/external/quantize_qwen_image_int8wo_diffusers.py \
        --model Qwen/Qwen-Image-Edit \
        --subfolder transformer \
        --out runs/qwen-image-edit-int8wo \
        --dtype bfloat16 \
        --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="quantize_qwen_image_int8wo_diffusers",
        description="diffusers + torchao int8wo quantizer (reproduces dimitribarbot/Qwen-Image-Edit-int8wo)",
    )
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit",
                        help="Base model repo id or local path (default: Qwen/Qwen-Image-Edit)")
    parser.add_argument("--subfolder", default="transformer",
                        help="Subfolder holding the diffusers transformer; pass '' for the repo root")
    parser.add_argument("--out", required=True, help="Output diffusers model directory")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Compute dtype the high-precision activations / dequant run in")
    parser.add_argument("--quant-type", default="int8wo",
                        help="torchao quant type passed to diffusers TorchAoConfig (default: int8wo)")
    parser.add_argument("--modules-to-not-convert", default=None,
                        help="Comma-separated module name substrings to keep unquantized (default: none)")
    parser.add_argument("--device-map", default=None,
                        help="Optional device_map for from_pretrained, e.g. 'auto' or 'cpu' (big models)")
    parser.add_argument("--base-bin", action="store_true",
                        help="Set if the BASE model stores weights as .bin instead of .safetensors")
    parser.add_argument("--verify", action="store_true",
                        help="Reload the saved folder and report a sample quantized weight")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without loading/quantizing")
    return parser.parse_args(argv)


def _plan(args: argparse.Namespace) -> dict:
    mods = [m.strip() for m in args.modules_to_not_convert.split(",")] if args.modules_to_not_convert else None
    return {
        "model": args.model,
        "subfolder": args.subfolder or None,
        "out": str(Path(args.out)),
        "dtype": args.dtype,
        "quant_type": args.quant_type,
        "modules_to_not_convert": mods,
        "device_map": args.device_map,
        "base_use_safetensors": not args.base_bin,
        "serialization": "pickled .bin (safe_serialization=False)",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = _plan(args)
    print("== plan ==")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("dry-run: not loading the model.")
        return 0

    # Lazy, guarded imports so --help / --dry-run work without the heavy deps.
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("torch is required: pip install torch") from exc
    try:
        import torchao  # noqa: F401  (diffusers TorchAoConfig needs it installed)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("torchao is required: pip install 'torchao>=0.14'") from exc
    try:
        from diffusers import AutoModel
        try:
            from diffusers import TorchAoConfig
        except ImportError:  # older export location
            from diffusers.quantizers.quantization_config import TorchAoConfig
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("diffusers is required: pip install 'diffusers>=0.33' accelerate transformers") from exc

    dtype = getattr(torch, args.dtype)
    quant_config = _build_quant_config(TorchAoConfig, args.quant_type, plan["modules_to_not_convert"])

    from_kwargs: dict = {"quantization_config": quant_config, "torch_dtype": dtype}
    if plan["subfolder"]:
        from_kwargs["subfolder"] = plan["subfolder"]
    if args.device_map:
        from_kwargs["device_map"] = args.device_map
    if args.base_bin:
        from_kwargs["use_safetensors"] = False

    print(f"\n== loading + quantizing {args.model!r} (this downloads/loads the full transformer) ==")
    model = AutoModel.from_pretrained(args.model, **from_kwargs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"== saving to {out} (safe_serialization=False -> pickled .bin) ==")
    # torchao subclasses cannot go into safetensors; must pickle.
    model.save_pretrained(out, safe_serialization=False)

    _summarize(out)
    if args.verify:
        _verify(out, AutoModel, dtype)
    return 0


def _build_quant_config(TorchAoConfig, quant_type: str, modules_to_not_convert):
    """Build a diffusers ``TorchAoConfig`` across torchao/diffusers versions.

    diffusers <= 0.36 accepts a string alias (e.g. ``"int8wo"``) and serializes
    it verbatim into ``config.json`` (this is what the reference repo shows).
    diffusers >= 0.37 / torchao >= 0.x require an ``AOBaseConfig`` instance, and
    serialize ``quant_type`` as a config dict (``{"_type": "Int8WeightOnlyConfig",
    ...}``, ``granularity=PerRow`` = per-output-channel). Both are functionally
    identical int8 weight-only; only the serialized representation differs by
    version. We try the string first, then fall back to the AOBaseConfig.
    """
    try:
        return TorchAoConfig(quant_type, modules_to_not_convert=modules_to_not_convert)
    except TypeError:
        try:
            from torchao.quantization import (
                Float8WeightOnlyConfig,
                Int4WeightOnlyConfig,
                Int8WeightOnlyConfig,
            )
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("torchao config classes unavailable; upgrade torchao") from exc
        mapping = {
            "int8wo": Int8WeightOnlyConfig,
            "int4wo": Int4WeightOnlyConfig,
            "fp8wo": Float8WeightOnlyConfig,
        }
        if quant_type not in mapping:
            raise SystemExit(
                f"quant-type {quant_type!r} needs an AOBaseConfig mapping for this diffusers/torchao version; "
                f"known: {sorted(mapping)}"
            )
        return TorchAoConfig(mapping[quant_type](), modules_to_not_convert=modules_to_not_convert)


def _summarize(out: Path) -> None:
    print("\n== output ==")
    files = sorted(p.name for p in out.iterdir())
    print("files:", files)
    cfg_path = out / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        print("_class_name:", cfg.get("_class_name"))
        print("quantization_config:", json.dumps(cfg.get("quantization_config"), ensure_ascii=False))
    index = out / "diffusion_pytorch_model.bin.index.json"
    if index.is_file():
        meta = json.loads(index.read_text(encoding="utf-8")).get("metadata", {})
        total = meta.get("total_size")
        if total:
            print(f"total_size: {total} bytes ({total / 1e9:.2f} GB)")


def _verify(out: Path, AutoModel, dtype) -> None:
    print("\n== verify: reload (use_safetensors=False) ==")
    reloaded = AutoModel.from_pretrained(out, use_safetensors=False, torch_dtype=dtype)
    sample = None
    for name, module in reloaded.named_modules():
        w = getattr(module, "weight", None)
        if w is not None and type(w).__name__ != "Parameter" or (w is not None and "AffineQuant" in type(getattr(w, "data", w)).__name__):
            sample = (name, w)
            break
    if sample is None:
        # Fall back to the first weight, quantized or not.
        for name, p in reloaded.named_parameters():
            if name.endswith(".weight"):
                sample = (name, p)
                break
    if sample is not None:
        name, w = sample
        data = getattr(w, "data", w)
        print(f"sample weight: {name} -> {type(data).__name__}")
        try:
            deq = data.dequantize() if hasattr(data, "dequantize") else data.to(dtype)
            print(f"dequant dtype: {deq.dtype}, shape: {tuple(deq.shape)}")
        except Exception as exc:  # pragma: no cover
            print("dequant check skipped:", exc)


if __name__ == "__main__":
    sys.exit(main())
