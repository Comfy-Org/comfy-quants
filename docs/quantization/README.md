# Quantization guides

Use this directory to choose the command for the artifact you want to produce.
Format-level tensor definitions live under [`../formats/`](../formats/) and are
not duplicated here.

## Entrypoint matrix

| Goal | Command | Inputs | Output | Guide |
| --- | --- | --- | --- | --- |
| Export a Qwen FP8 checkpoint | `quantize`, `export-model` | FP8 config; local transformer safetensors; CUDA device | FP8 ComfyUI checkpoint | [`fp8.md`](fp8.md) |
| Export Qwen-Image-Edit-2511 INT4 tile-pack | `qwen-image-edit-2511-int4` | Qwen model dir; BF16 scaffold; DeepCompressor checkout; Nunchaku checkout; calibration set | single `svdquant_w4a4` tile-pack safetensors | [`qwen_image_edit_2511_int4.md`](qwen_image_edit_2511_int4.md) |
| Run the native INT4 solver | `quantize-int4` | dense checkpoint; optional activation stats; optional GPTQ Hessians | native `svdquant_w4a4` tile-pack | [`native_int4.md`](native_int4.md) |
| Inspect or repack INT4 artifacts | `inspect-int4`, `export-int4` | existing tile-pack or imported PTQ artifact directory | JSON report or repacked tile-pack | [`int4_tools.md`](int4_tools.md) |

## Route boundaries

- `qwen-image-edit-2511-int4` is the production Qwen-Image-Edit-2511 INT4 export flow.
- `quantize-int4` is the package-native solver for advanced users and algorithm work.
- `export-int4` repacks already-quantized artifacts; it does not run calibration, search, or GPTQ.
- `inspect-int4` validates static artifact structure; image quality validation is a separate inference run.
