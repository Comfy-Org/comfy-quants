# Qwen-Image-Edit-2511 INT4 export

Use this guide to produce one ComfyUI-loadable INT4 checkpoint for
Qwen-Image-Edit-2511.

The output is a single `.safetensors` file using the `svdquant_w4a4` kitchen
tile-pack layout. The default path runs the full flow: calibration/search, PTQ,
QKV split and conversion, tile-pack export, and structural inspection.

## What you get

| Item | Value |
| --- | --- |
| Output file | single `.safetensors` checkpoint |
| Target format | `svdquant_w4a4` |
| Storage layout | kitchen tile-pack W4A4 |
| Default calibration samples | `128` |
| Default search preset | `quality-r64` |
| Default GPU selector | `0` |
| Report | `<output>.pipeline_report.json` unless `--report` is set |

## Prerequisites

Prepare these local paths before running the command:

| Required path | CLI argument | Notes |
| --- | --- | --- |
| Qwen-Image-Edit-2511 model | `--model` | Hugging Face id or local model directory. |
| BF16 transformer checkpoint | `--base-checkpoint` | Base transformer checkpoint used to assemble the final artifact. |
| DeepCompressor checkout | `--deepcompressor-root` | Provides the Qwen-Image-Edit-2511 search/PTQ implementation. |
| Nunchaku checkout | `--nunchaku-root` | Provides the split/merge helpers used by the bridge export route. |
| Calibration cache or dataset | `--calibration-path` | Optional. If omitted, the default Qwen-Image-Edit-2511 calibration path under the DeepCompressor checkout is used. |

Default calibration path:

```text
<deepcompressor-root>/datasets/torch.bfloat16/qwen-image-edit-2511/fmeuler50-g4.0/qdiff/s128
```

Use a CUDA GPU for the full export. Select devices with `--gpus`; the value is
passed as `CUDA_VISIBLE_DEVICES` for the PTQ step.

## Export with the default calibration set

This command uses the default 128-sample calibration set and the `quality-r64`
search preset:

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

## Export with a custom calibration set

Pass `--calibration-path` when you want to use your own calibration cache or
sample set. Keep `--calibration-samples` aligned with the number of samples you
want the search/PTQ step to use.

```bash
comfy-quants qwen-image-edit-2511-int4 \
  --model /path/to/Qwen-Image-Edit-2511 \
  --base-checkpoint /path/to/qwen_image_edit_2511_bf16_transformer.safetensors \
  --out /path/to/qwen_image_edit_2511_int4_custom_calib.safetensors \
  --deepcompressor-root /path/to/DeepCompressor \
  --nunchaku-root /path/to/nunchaku \
  --calibration-path /path/to/qwen-image-edit-calibration/s128 \
  --calibration-samples 128 \
  --search-strength quality-r64 \
  --gpus 0 \
  --hash-output \
  --json
```

## Reuse an existing PTQ result

If you already have a DeepCompressor PTQ artifact directory, pass it with
`--quant-path`. The command will skip calibration/search/PTQ and only run the
conversion, final tile-pack export, and inspection steps.

```bash
comfy-quants qwen-image-edit-2511-int4 \
  --quant-path /path/to/deepcompressor/run/model \
  --base-checkpoint /path/to/qwen_image_edit_2511_bf16_transformer.safetensors \
  --out /path/to/qwen_image_edit_2511_int4_tilepack.safetensors \
  --deepcompressor-root /path/to/DeepCompressor \
  --nunchaku-root /path/to/nunchaku \
  --gpus 0 \
  --reuse \
  --hash-output \
  --json
```

## Preview the resolved plan

Use `--dry-run` to print the resolved paths, selected preset, calibration path,
and external commands without producing the final checkpoint.

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
  --dry-run \
  --json
```

## Output files

For an output path such as:

```text
/path/to/qwen_image_edit_2511_int4_tilepack.safetensors
```

the command writes:

| File or directory | Description |
| --- | --- |
| `/path/to/qwen_image_edit_2511_int4_tilepack.safetensors` | Final ComfyUI-loadable INT4 checkpoint. |
| `/path/to/qwen_image_edit_2511_int4_tilepack.pipeline_report.json` | Resolved config, executed commands, output hash, timing, and inspection result. |
| `--export-root` directory | Intermediate split/raw artifacts when the bridge route is used. |
| DeepCompressor runs directory | PTQ run output when `--quant-path` is not supplied. |

Set `--report`, `--export-root`, or `--runs-root` if you want those paths in a
specific location.

## Inspect the exported checkpoint

The one-step command runs strict inspection by default. You can also inspect the
artifact directly:

```bash
comfy-quants inspect-int4 \
  --artifact /path/to/qwen_image_edit_2511_int4_tilepack.safetensors \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --strict-qwen-image-edit-2511 \
  --json
```

Inspection verifies the static tensor contract: expected tensor families, QKV
split layout, tile-pack shapes, rank metadata, and Qwen-Image-Edit-2511 counts.
Run a target ComfyUI workflow after inspection to validate final image output.

## Load in ComfyUI

Copy or symlink the exported `.safetensors` file into the model location expected
by your compatible ComfyUI loader or custom node, then select it in the workflow.
The exact folder name is loader-specific; use the folder documented by the loader
that supports `svdquant_w4a4` kitchen tile-pack checkpoints.

## Main options

| Option | Default | Description |
| --- | --- | --- |
| `--model` | `Qwen/Qwen-Image-Edit-2511` | Source model id or local model directory. |
| `--base-checkpoint` | required for default route | BF16 transformer checkpoint used to assemble the final artifact. |
| `--out` | required | Final single-file `.safetensors` output path. |
| `--deepcompressor-root` | explicit path recommended | Local DeepCompressor checkout. |
| `--nunchaku-root` | explicit path recommended | Local Nunchaku checkout. |
| `--calibration-path` | default path under DeepCompressor | Calibration cache or dataset. |
| `--calibration-samples` | `128` | Number of calibration samples used by search/PTQ. |
| `--search-strength` | `quality-r64` | Search preset. Use `--help` for the full preset list. |
| `--gpus` | `0` | GPU ids for `CUDA_VISIBLE_DEVICES`. |
| `--quant-path` | unset | Existing DeepCompressor PTQ artifact directory to reuse. |
| `--reuse` | off | Reuse existing intermediate split/raw/final artifacts when present. |
| `--hash-output` | off | Add SHA256 for the final checkpoint to the report. |
| `--dry-run` | off | Resolve and print the plan without running the export. |
| `--json` | off | Print machine-readable command output. |

## Related pages

- [`../formats/svdquant_w4a4_kitchen_tilepack.md`](../formats/svdquant_w4a4_kitchen_tilepack.md)
- [`../formats/awq_w4a16.md`](../formats/awq_w4a16.md)
- [`int4_tools.md`](int4_tools.md)
