# CLI reference

The public command is `comfy-quants`.

```bash
comfy-quants --help
comfy-quants <command> --help
```

Use this page for command syntax. For complete workflows, start with
[`quantization/`](quantization/). For tensor/storage details, see
[`formats/`](formats/).

## Inspect source weights

```bash
comfy-quants inspect \
  --model Qwen/Qwen-Image-Edit-2511 \
  --family qwen_image_edit \
  --out runs/qwen-edit-2511/inspect \
  --json
```

## FP8 planning and export

Plan selected tensors without writing checkpoint bytes:

```bash
comfy-quants quantize \
  --config configs/qwen_image_edit_2511_fp8_static.yaml \
  --work-dir runs/qwen-edit-2511/fp8-e4m3-plan \
  --dry-run \
  --json
```

Export a full ComfyUI-loadable FP8 checkpoint:

```bash
comfy-quants export-model \
  --config configs/qwen_image_edit_2511_fp8_static.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out runs/qwen-edit-2511/export-fp8-e4m3 \
  --device cuda:0 \
  --hash-output \
  --json
```

Guide: [`quantization/fp8.md`](quantization/fp8.md).

## Qwen-Image-Edit-2511 INT4 one-step export

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

Useful options:

| Option | Meaning |
| --- | --- |
| `--calibration-path` | Custom calibration cache or dataset path. |
| `--calibration-samples` | Calibration sample count. Default: `128`. |
| `--search-strength` | Search preset. Default: `quality-r64`. |
| `--quant-path` | Reuse an existing DeepCompressor PTQ artifact directory. |
| `--reuse` | Reuse existing intermediate artifacts when present. |
| `--dry-run` | Print the resolved plan without running the export. |

Guide: [`quantization/qwen_image_edit_2511_int4.md`](quantization/qwen_image_edit_2511_int4.md).

## Built-in INT4 solver

```bash
comfy-quants quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out runs/qwen-edit-2511/int4-svdquant-w4a4 \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

Main modes:

```text
weight_only_initialization
calibrated_svdquant
svdquant_gptq_experimental
```

Guide: [`quantization/native_int4.md`](quantization/native_int4.md).

## Calibration helpers

These commands create capture plans and reduce captured activation tensors. They
are advanced helpers for solver development.

```bash
comfy-quants calib plan-int4-capture --help
comfy-quants calib materialize-int4-capture --help
comfy-quants calib reduce-int4-activations --help
comfy-quants calib reduce-int4-gptq-hessians --help
```

## INT4 artifact inspection

```bash
comfy-quants inspect-int4 \
  --artifact /path/to/qwen_image_edit_2511_int4_tilepack.safetensors \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --strict-qwen-image-edit-2511 \
  --json
```

Guide: [`quantization/int4_tools.md`](quantization/int4_tools.md).

## INT4 repack/export

```bash
comfy-quants export-int4 \
  --format svdquant_w4a4 \
  --source-format deepcompressor-qwen-image-edit \
  --source /path/to/deepcompressor-ptq-artifacts \
  --out runs/qwen-edit-2511/export-int4 \
  --device cuda:0 \
  --hash-output \
  --json
```

## Generic artifact commands

```bash
comfy-quants validate --help
comfy-quants export --help
comfy-quants jobs --help
comfy-quants resume --help
```
