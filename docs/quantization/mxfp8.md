# MXFP8 (OCP microscaling FP8) export

Use this guide to produce an **MXFP8** checkpoint loadable by **stock ComfyUI's
native** quantized loader. Comfy Quants writes `float8_e4m3fn` weights + per-32-element
**E8M0** block scales (in the cuBLAS `to_blocked` swizzle) + a per-layer `comfy_quant`
marker. Format details: [`../formats/mxfp8.md`](../formats/mxfp8.md).

This is the **native-loader sibling of FP8** — same `comfy_quant` handshake, no
downstream custom node required. It is a **producer-only** flow; the mxfp8
tensor-core matmul lives in ComfyUI/comfy-kitchen.

## Supported model-family configs (v1)

| Model family | Config |
| --- | --- |
| Qwen-Image | `configs/qwen_image_2512_mxfp8.yaml` |
| Qwen-Image-Edit-2511 | `configs/qwen_image_edit_2511_mxfp8.yaml` |
| Qwen-Image-Layered | `configs/qwen_image_layered_mxfp8.yaml` |

## Quick start: Qwen-Image-Layered

```bash
comfy-quants export-model-mxfp8 \
  --config configs/qwen_image_layered_mxfp8.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out /path/to/qwen_image_layered_mxfp8.safetensors \
  --device cuda:0 \
  --hash-output \
  --json
```

Directory outputs use `diffusion_pytorch_model.mxfp8.safetensors`.

## Inputs

| Input | Argument | Description |
| --- | --- | --- |
| Config | `--config` | YAML selecting family, source, and `quant.target_dtype: mxfp8`. |
| Source | `--source` | Local transformer `.safetensors` / index JSON / indexed directory. |
| Output | `--out` | Output `.safetensors` path or directory. |
| Device | `--device` | Torch device; `auto` uses `cuda:0` when available, else CPU. |

Requires `quant.target_dtype: mxfp8` in the config. The block size (32) and the E8M0
scale are fixed by the format. Guide: [`../formats/mxfp8.md`](../formats/mxfp8.md).

## Layer selection

MXFP8 reuses the **FP8 default policy** (it is the native-loader sibling): the
selected set is every transformer-block Linear except norms / `proj_out` /
`norm_out` and the block-0 `img_mod.1` special case — identical to the FP8 flow.
`in_features` of every selected Qwen Linear is a multiple of 32, so all are
block-quantizable.

## Loader prerequisite & runtime gate

The artifact loads in **stock ComfyUI** via the native `QUANT_ALGOS["mxfp8"]` path —
no custom node required. The quantized mxfp8 matmul is **Blackwell-gated** (NVIDIA
SM ≥ 10 + torch ≥ 2.10 + comfy_kitchen); on other hardware ComfyUI silently
dequantizes the weight to the compute dtype (loads & correct, no speedup).
