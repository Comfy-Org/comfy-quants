# FP8 E4M3 / E5M2 export

Use this flow to export Qwen-Image and Qwen-Image-Edit transformer checkpoints as
ComfyUI-loadable FP8 `.safetensors` files. Format details are defined in
[`../formats/fp8.md`](../formats/fp8.md).

## Inputs

| Input | Argument | Description |
| --- | --- | --- |
| Config | `--config` | YAML/JSON selecting model family, source type, quantization algorithm, and target dtype. |
| Source checkpoint | `--source` | Local transformer `.safetensors`, safetensors index JSON, or indexed directory. |
| Output path | `--out` | Output `.safetensors` path or output directory. |
| Device | `--device` | Torch device. `auto` uses `cuda:0` when available and falls back to CPU. |

Example configs:

```text
configs/qwen_image_2512_fp8_static.yaml
configs/qwen_image_2512_fp8_e5m2_static.yaml
configs/qwen_image_edit_2511_fp8_static.yaml
configs/qwen_image_edit_2511_fp8_e5m2_static.yaml
```

## Plan without writing checkpoint bytes

```bash
comfy-quants quantize \
  --config configs/qwen_image_edit_2511_fp8_static.yaml \
  --work-dir runs/qwen-edit-2511/fp8-e4m3-plan \
  --dry-run \
  --json
```

## Export a single checkpoint

E4M3:

```bash
comfy-quants export-model \
  --config configs/qwen_image_edit_2511_fp8_static.yaml \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --out runs/qwen-edit-2511/export-fp8-e4m3 \
  --device cuda:0 \
  --hash-output \
  --json
```

E5M2:

```bash
comfy-quants export-model \
  --config configs/qwen_image_edit_2511_fp8_e5m2_static.yaml \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --out runs/qwen-edit-2511/export-fp8-e5m2 \
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
  --config /absolute/path/to/local-qwen-fp8.yaml \
  --work-dir runs/qwen-image-local/fp8-static-v0 \
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
