# FP8 E4M3 / E5M2 export

Use this guide when the target artifact is a full FP8 transformer checkpoint.
The exported file is a ComfyUI-loadable `.safetensors` checkpoint. Format
details are defined in [`../formats/fp8.md`](../formats/fp8.md).

Model-family config files select the tensor contract, source layout, target FP8
dtype, and output naming rules. Start with the format, then choose one of the
supported model-family configs below.

## Supported model-family configs

| Model family | E4M3 config | E5M2 config |
| --- | --- | --- |
| Qwen-Image | `configs/qwen_image_2512_fp8_static.yaml` | `configs/qwen_image_2512_fp8_e5m2_static.yaml` |
| Qwen-Image-Edit-2511 | `configs/qwen_image_edit_2511_fp8_static.yaml` | `configs/qwen_image_edit_2511_fp8_e5m2_static.yaml` |

## Quick start: Qwen-Image-Edit-2511

E4M3:

```bash
comfy-quants export-model \
  --config configs/qwen_image_edit_2511_fp8_static.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out /path/to/qwen_image_edit_2511_fp8_e4m3.safetensors \
  --device cuda:0 \
  --hash-output \
  --json
```

E5M2:

```bash
comfy-quants export-model \
  --config configs/qwen_image_edit_2511_fp8_e5m2_static.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out /path/to/qwen_image_edit_2511_fp8_e5m2.safetensors \
  --device cuda:0 \
  --hash-output \
  --json
```

Use `--device auto` when you want CUDA if available and CPU fallback otherwise.
After export, copy or symlink the `.safetensors` file into the model directory
used by the compatible ComfyUI loader and run an inference workflow.

## Inputs

| Input | Argument | Description |
| --- | --- | --- |
| Config | `--config` | YAML/JSON selecting model family, source type, quantization algorithm, and target dtype. |
| Source checkpoint | `--source` | Local transformer `.safetensors`, safetensors index JSON, or indexed directory. |
| Output path | `--out` | Output `.safetensors` path or output directory. |
| Device | `--device` | Torch device. `auto` uses `cuda:0` when available and falls back to CPU. |

## Plan without writing checkpoint bytes

```bash
comfy-quants quantize \
  --config /path/to/fp8_config.yaml \
  --work-dir runs/fp8-plan \
  --dry-run \
  --json
```

## Export a single checkpoint

E4M3:

```bash
comfy-quants export-model \
  --config /path/to/fp8_e4m3_config.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out runs/export-fp8-e4m3 \
  --device cuda:0 \
  --hash-output \
  --json
```

E5M2:

```bash
comfy-quants export-model \
  --config /path/to/fp8_e5m2_config.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out runs/export-fp8-e5m2 \
  --device cuda:0 \
  --hash-output \
  --json
```

Directory outputs use format-specific filenames:

```text
diffusion_pytorch_model.fp8_e4m3.safetensors
diffusion_pytorch_model.fp8_e5m2.safetensors
```

## Optional selected-payload artifact

`quantize` can write only selected FP8 payload bytes and scales instead of a full
inference checkpoint:

```bash
comfy-quants quantize \
  --config /path/to/fp8_config.yaml \
  --work-dir runs/fp8-static-v0 \
  --device cuda:0 \
  --json
```

Payload layout:

```text
artifact/
├── quant_tensor_index.json
├── payload_report.json
├── tensors/fp8_weights.safetensors
└── scales/fp8_static_scales.safetensors
```
