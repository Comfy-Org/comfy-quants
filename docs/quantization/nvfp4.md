# NVFP4 (FP4 E2M1 microscaling) export

Use this guide to produce an **NVFP4** checkpoint loadable by **stock ComfyUI's
native** quantized loader. Comfy Quants writes FP4-E2M1 weights (packed 2-per-byte) +
per-block-16 FP8-E4M3 block scales (cuBLAS `to_blocked` swizzle) + a per-tensor FP32
scale + a per-layer `comfy_quant` marker. Format details:
[`../formats/nvfp4.md`](../formats/nvfp4.md).

This is a **native-loader format** (same `comfy_quant` handshake as FP8/MXFP8) and a
**producer-only** flow — the nvfp4 tensor-core matmul lives in ComfyUI/comfy-kitchen.

## Supported model-family configs (v1)

| Model family | Config |
| --- | --- |
| Qwen-Image | `configs/qwen_image_2512_nvfp4.yaml` |
| Qwen-Image-Edit-2511 | `configs/qwen_image_edit_2511_nvfp4.yaml` |
| Qwen-Image-Layered | `configs/qwen_image_layered_nvfp4.yaml` |

## Quick start: Qwen-Image-Layered

```bash
comfy-quants export-model-nvfp4 \
  --config configs/qwen_image_layered_nvfp4.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out /path/to/qwen_image_layered_nvfp4.safetensors \
  --device cuda:0 \
  --hash-output \
  --json
```

Directory outputs use `diffusion_pytorch_model.nvfp4.safetensors`.

## Inputs

| Input | Argument | Description |
| --- | --- | --- |
| Config | `--config` | YAML selecting family, source, and `quant.target_dtype: nvfp4`. |
| Source | `--source` | Local transformer `.safetensors` / index JSON / indexed directory. |
| Output | `--out` | Output `.safetensors` path or directory. |
| Device | `--device` | Torch device; `auto` uses `cuda:0` when available, else CPU. |

Requires `quant.target_dtype: nvfp4` in the config. The block size (16), the two-level
scale, and deterministic E2M1 rounding are fixed by the format.

## Layer selection

NVFP4 reuses the **FP8 default policy** (it is a native-loader format): the selected
set is every transformer-block Linear except norms / `proj_out` / `norm_out` and the
block-0 `img_mod.1` special case — identical to the FP8/MXFP8 flows. `in_features` of
every selected Qwen Linear is a multiple of 16, so all are block-quantizable.

## Loader prerequisite & runtime gate

The artifact loads in **stock ComfyUI** via the native `QUANT_ALGOS["nvfp4"]` path — no
custom node required. The quantized nvfp4 matmul is **Blackwell-gated** (NVIDIA SM ≥ 10
+ comfy_kitchen); on other hardware ComfyUI silently dequantizes the weight to the
compute dtype (loads & correct, no speedup).
