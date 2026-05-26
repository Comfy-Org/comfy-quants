# Qwen-Image-Edit-2511 INT4 tile-pack export

Use `qwen-image-edit-2511-int4` to export Qwen-Image-Edit-2511 into a single
`svdquant_w4a4` kitchen tile-pack `.safetensors` file.

The command coordinates external DeepCompressor and Nunchaku checkouts through
subprocesses, then uses `comfy_quants` to write and inspect the final artifact.
DeepCompressor and Nunchaku are not Python dependencies of this package.

## Inputs

| Input | Argument | Default |
| --- | --- | --- |
| Qwen-Image-Edit-2511 model directory | `--model` | required when PTQ is run |
| BF16 transformer scaffold checkpoint | `--base-checkpoint` | required for the default bridge route |
| DeepCompressor checkout | `--deepcompressor-root` | required unless `--quant-path` is supplied |
| Nunchaku checkout | `--nunchaku-root` | required for the default bridge route |
| Calibration set | `--calibration-path` | `<deepcompressor-root>/datasets/torch.bfloat16/qwen-image-edit-2511/fmeuler50-g4.0/qdiff/s128` |
| Calibration sample count | `--calibration-samples` | `128` |
| Search preset | `--search-strength` | `quality-r64` |
| GPU selection | `--gpus` | `0` |
| Output artifact | `--out` | required |

## Pipeline

```text
model directory
  + BF16 transformer scaffold
  + calibration set
  -> DeepCompressor search/PTQ
  -> Nunchaku split/merge conversion
  -> comfy_quants tile-pack conversion
  -> inspect-int4 structural check
  -> single svdquant_w4a4 .safetensors
```

## Dry run

```bash
comfy-quants qwen-image-edit-2511-int4 \
  --model /path/to/Qwen-Image-Edit-2511 \
  --base-checkpoint /path/to/qwen_image_edit_2511_bf16_transformer.safetensors \
  --out /tmp/qwen_edit_2511_int4_tilepack.safetensors \
  --deepcompressor-root /path/to/DeepCompressor \
  --nunchaku-root /path/to/nunchaku \
  --search-strength quality-r64 \
  --calibration-samples 128 \
  --gpus 0 \
  --dry-run \
  --json
```

## Export

```bash
comfy-quants qwen-image-edit-2511-int4 \
  --model /path/to/Qwen-Image-Edit-2511 \
  --base-checkpoint /path/to/qwen_image_edit_2511_bf16_transformer.safetensors \
  --out runs/qwen-image-edit-2511-int4-tilepack/qwen_edit_2511_quality_r64_128calib_int4_tilepack.safetensors \
  --deepcompressor-root /path/to/DeepCompressor \
  --nunchaku-root /path/to/nunchaku \
  --search-strength quality-r64 \
  --calibration-samples 128 \
  --gpus 0 \
  --hash-output \
  --json
```

Use a custom calibration set by passing `--calibration-path`:

```bash
comfy-quants qwen-image-edit-2511-int4 \
  --model /path/to/Qwen-Image-Edit-2511 \
  --base-checkpoint /path/to/qwen_image_edit_2511_bf16_transformer.safetensors \
  --out runs/qwen-edit-2511/qwen_edit_2511_custom_calib_int4_tilepack.safetensors \
  --deepcompressor-root /path/to/DeepCompressor \
  --nunchaku-root /path/to/nunchaku \
  --calibration-path /absolute/path/to/qdiff/s128 \
  --calibration-samples 128 \
  --search-strength quality-r64 \
  --gpus 0 \
  --hash-output \
  --json
```

## Search presets

`--search-strength` accepts the presets exposed by the CLI, including `fast-*`,
`balanced-*`, `mid-*`, and `quality-*` variants. Use
`comfy-quants qwen-image-edit-2511-int4 --help` for the complete list.

## Inspect output

```bash
comfy-quants inspect-int4 \
  --artifact /absolute/path/to/qwen_edit_2511_int4_tilepack.safetensors \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --strict-qwen-image-edit-2511 \
  --json
```

The inspector checks the static artifact structure. Run a separate target-runtime
inference workflow to validate image output.

## Format reference

- [`../formats/svdquant_w4a4_kitchen_tilepack.md`](../formats/svdquant_w4a4_kitchen_tilepack.md)
- [`../formats/awq_w4a16.md`](../formats/awq_w4a16.md)
