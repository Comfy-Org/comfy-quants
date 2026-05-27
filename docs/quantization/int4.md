# INT4 export

Use this guide when the target artifact is an INT4 checkpoint. The main INT4
artifact produced by this package is a single-file `svdquant_w4a4` kitchen
tile-pack `.safetensors` checkpoint. Supported mixed bundles can also include
AWQ W4A16 tensors for modulation layers.

## INT4 formats

| Format | Role | Reference |
| --- | --- | --- |
| SVDQuant W4A4 kitchen tile-pack | Main tile-packed INT4 linear format | [`../formats/svdquant_w4a4_kitchen_tilepack.md`](../formats/svdquant_w4a4_kitchen_tilepack.md) |
| AWQ W4A16 | Mixed INT4 modulation tensor format in supported bundles | [`../formats/awq_w4a16.md`](../formats/awq_w4a16.md) |

## Model-family flows

| Model family or flow | Command | Calibration/search | Output | Guide |
| --- | --- | --- | --- | --- |
| Qwen-Image-Edit-2511 one-step export | `comfy-quants qwen-image-edit-2511-int4` | DeepCompressor search/PTQ; default 128 calibration samples | single `svdquant_w4a4` tile-pack checkpoint | [`qwen_image_edit_2511_int4.md`](qwen_image_edit_2511_int4.md) |
| Built-in solver flow | `comfy-quants quantize-int4` | optional activation stats and GPTQ Hessians | `svdquant_w4a4` tile-pack artifact | [`native_int4.md`](native_int4.md) |
| Existing artifact tools | `comfy-quants inspect-int4`, `comfy-quants export-int4` | no calibration or search | JSON report or repacked tile-pack checkpoint | [`int4_tools.md`](int4_tools.md) |

## Recommended path

For a full INT4 export, use the model-family one-step guide when one is available.
That path owns calibration/search defaults, model-specific conversion, final
single-file export, and strict inspection.

Use the built-in solver guide when you are developing or evaluating solver logic.
Use the artifact tools guide when you already have an INT4 artifact and only need
to inspect or repack it.

## Quick start: Qwen-Image-Edit-2511

This is the shortest path to produce the `svdquant_w4a4` kitchen tile-pack
checkpoint for Qwen-Image-Edit-2511 with the default 128-sample calibration set
and `quality-r64` search preset:

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

Use [`qwen_image_edit_2511_int4.md`](qwen_image_edit_2511_int4.md) for custom
calibration data, PTQ reuse, dry-run command preview, and output file details.

## Inspect an INT4 checkpoint

```bash
comfy-quants inspect-int4 \
  --artifact /path/to/model_int4_tilepack.safetensors \
  --family <model_family> \
  --format svdquant_w4a4 \
  --json
```

Some model-family guides add stricter inspection flags for their known tensor
counts and naming rules.
