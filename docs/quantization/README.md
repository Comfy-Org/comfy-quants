# Quantization guides

Start with the artifact you want to create. Format-level tensor definitions live
under [`../formats/`](../formats/).

## Choose an entrypoint

| I want to... | Use | Main inputs | Output | Guide |
| --- | --- | --- | --- | --- |
| Export a Qwen FP8 checkpoint | `comfy-quants export-model` | FP8 config; local transformer checkpoint; CUDA device | FP8 `.safetensors` checkpoint | [`fp8.md`](fp8.md) |
| Produce Qwen-Image-Edit-2511 INT4 for ComfyUI | `comfy-quants qwen-image-edit-2511-int4` | Qwen model; BF16 base checkpoint; DeepCompressor; Nunchaku; calibration set | single `svdquant_w4a4` tile-pack `.safetensors` | [`qwen_image_edit_2511_int4.md`](qwen_image_edit_2511_int4.md) |
| Work on the built-in INT4 solver | `comfy-quants quantize-int4` | dense checkpoint; optional activation stats; optional GPTQ Hessians | `svdquant_w4a4` tile-pack artifact | [`native_int4.md`](native_int4.md) |
| Inspect or repack INT4 artifacts | `comfy-quants inspect-int4`, `comfy-quants export-int4` | existing tile-pack or imported PTQ artifact directory | JSON report or repacked tile-pack | [`int4_tools.md`](int4_tools.md) |

## Recommended starting points

- For Qwen-Image-Edit-2511 INT4, start with
  [`qwen_image_edit_2511_int4.md`](qwen_image_edit_2511_int4.md). It is the
  complete one-command export path for the `svdquant_w4a4` tile-pack artifact.
- For FP8 E4M3 or E5M2 exports, start with [`fp8.md`](fp8.md).
- Use [`int4_tools.md`](int4_tools.md) when you already have an INT4 artifact and
  only need to inspect or repack it.

## Validation flow

1. Export the checkpoint with the selected command.
2. Run the relevant inspector, for example `comfy-quants inspect-int4` for INT4
   tile-pack artifacts.
3. Load the exported `.safetensors` file in a compatible ComfyUI setup and run an
   inference workflow to validate the generated image.
