# Comfy Quants documentation

This documentation is organized by ownership. Each concept has one canonical page;
other pages link to it instead of redefining it.

## Structure

| Area | Canonical location | Owns |
| --- | --- | --- |
| Project boundary and extension model | [`architecture.md`](architecture.md) | package layout, dependency boundary, extension rules |
| CLI command reference | [`cli.md`](cli.md) | command names, required flags, common examples |
| Quantization user flows | [`quantization/`](quantization/) | which command to run for each input/output goal |
| Artifact storage formats | [`formats/`](formats/) | tensor names, shapes, metadata, packing rules |

## Quantization guides

- [`quantization/fp8.md`](quantization/fp8.md) — export Qwen FP8 E4M3/E5M2 checkpoints.
- [`quantization/qwen_image_edit_2511_int4.md`](quantization/qwen_image_edit_2511_int4.md) — one-step Qwen-Image-Edit-2511 INT4 tile-pack export.
- [`quantization/native_int4.md`](quantization/native_int4.md) — package-native INT4 solver for development and advanced use.
- [`quantization/int4_tools.md`](quantization/int4_tools.md) — inspect and repack existing INT4 artifacts.

## Format references

- [`formats/fp8.md`](formats/fp8.md) — FP8 E4M3/E5M2 checkpoint metadata and dtype mapping.
- [`formats/svdquant_w4a4_kitchen_tilepack.md`](formats/svdquant_w4a4_kitchen_tilepack.md) — SVDQuant W4A4 kitchen tile-pack tensor contract.
- [`formats/awq_w4a16.md`](formats/awq_w4a16.md) — AWQ W4A16 modulation tensor contract.

## Public package boundary

`comfy_quants` produces ComfyUI-loadable artifacts. It does not import, vendor,
embed, launch, or configure ComfyUI. External quantization or conversion tools are
used only through explicit command-line/subprocess boundaries.
