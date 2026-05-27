# Quantization guides

Choose the output format first. Each format section then lists the supported
model-family flows for that format. Tensor/storage definitions live under
[`../formats/`](../formats/).

## Format matrix

| Output format | Workflow guide | Format reference | Model-family flows |
| --- | --- | --- | --- |
| FP8 E4M3 / E5M2 checkpoint | [`fp8.md`](fp8.md) | [`../formats/fp8.md`](../formats/fp8.md) | listed in the FP8 guide |
| INT4 SVDQuant W4A4 tile-pack | [`int4.md`](int4.md) | [`../formats/svdquant_w4a4_kitchen_tilepack.md`](../formats/svdquant_w4a4_kitchen_tilepack.md) | listed in the INT4 guide |
| INT4 AWQ W4A16 tensors | [`int4.md`](int4.md) | [`../formats/awq_w4a16.md`](../formats/awq_w4a16.md) | used by supported mixed INT4 bundles |

## FP8

Start with [`fp8.md`](fp8.md) when the target artifact is a full FP8 checkpoint.
The page lists the supported config files and model-family examples for E4M3 and
E5M2 exports.

Command pattern:

```bash
comfy-quants export-model \
  --config /path/to/fp8_config.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out /path/to/model_fp8.safetensors \
  --device cuda:0 \
  --hash-output \
  --json
```

## INT4

Start with [`int4.md`](int4.md) when the target artifact is an INT4 tile-pack
checkpoint. The page links to:

- model-family one-step export guides;
- built-in solver usage;
- artifact inspection and repack tools;
- SVDQuant W4A4 and AWQ W4A16 format references.

For the currently supported Qwen-Image-Edit-2511 one-step flow:

```bash
comfy-quants qwen-image-edit-2511-int4 \
  --model /path/to/Qwen-Image-Edit-2511 \
  --base-checkpoint /path/to/qwen_image_edit_2511_bf16_transformer.safetensors \
  --out /path/to/qwen_image_edit_2511_int4_tilepack.safetensors \
  --deepcompressor-root /path/to/DeepCompressor \
  --nunchaku-root /path/to/nunchaku \
  --calibration-samples 128 \
  --search-strength quality-r64 \
  --gpus 0 \
  --hash-output \
  --json
```

## Validation flow

1. Export the checkpoint with the selected format workflow.
2. Run the relevant inspector, for example `comfy-quants inspect-int4` for INT4
   tile-pack artifacts.
3. Load the exported `.safetensors` file in a compatible ComfyUI setup and run an
   inference workflow to validate the generated image.
