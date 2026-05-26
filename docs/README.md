# Comfy Quants documentation

Comfy Quants exports quantized checkpoints for ComfyUI-compatible loaders. The
user guides below are organized by what you want to produce: a checkpoint, a
format reference, or command-line help.

## Start here

| Goal | Page |
| --- | --- |
| Pick a quantization flow | [`quantization/README.md`](quantization/README.md) |
| Look up command syntax | [`cli.md`](cli.md) |
| Understand repository architecture | [`architecture.md`](architecture.md) |
| Check tensor/storage formats | [`formats/`](formats/) |

## Quantization guides

- [`quantization/fp8.md`](quantization/fp8.md) — export Qwen FP8 E4M3/E5M2 checkpoints.
- [`quantization/qwen_image_edit_2511_int4.md`](quantization/qwen_image_edit_2511_int4.md) — one-step Qwen-Image-Edit-2511 INT4 tile-pack export.
- [`quantization/native_int4.md`](quantization/native_int4.md) — built-in INT4 solver for development and advanced use.
- [`quantization/int4_tools.md`](quantization/int4_tools.md) — inspect and repack existing INT4 artifacts.

## Format references

- [`formats/fp8.md`](formats/fp8.md) — FP8 E4M3/E5M2 checkpoint metadata and dtype mapping.
- [`formats/svdquant_w4a4_kitchen_tilepack.md`](formats/svdquant_w4a4_kitchen_tilepack.md) — SVDQuant W4A4 kitchen tile-pack tensor contract.
- [`formats/awq_w4a16.md`](formats/awq_w4a16.md) — AWQ W4A16 modulation tensor contract.

## ComfyUI integration model

Run quantization with this package, then load the exported artifact with a
compatible ComfyUI setup. Downstream custom-node projects can depend on
`comfy-quants` and call its CLI or Python API when they need an in-UI workflow.
