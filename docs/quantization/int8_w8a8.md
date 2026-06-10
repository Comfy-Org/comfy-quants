# INT8 W8A8 (+ ConvRot) export

Use this guide to produce an INT8 **W8A8** prequantized checkpoint for the
**ComfyUI-INT8-Fast** custom node. Comfy Quants writes int8 weights (optionally
ConvRot-rotated) + per-output-channel scales offline; the node quantizes
activations dynamically and runs an `int8×int8` matmul at inference (faster than
bf16). Format details: [`../formats/int8_w8a8.md`](../formats/int8_w8a8.md).

This is a **producer-only** flow — the Triton W8A8 kernel, dynamic activation
quantization, and online activation rotation all live in the downstream node.

## Supported model-family configs (v1)

| Model family | Config |
| --- | --- |
| Qwen-Image | `configs/qwen_image_2512_int8_w8a8.yaml` |
| Qwen-Image-Edit-2511 | `configs/qwen_image_edit_2511_int8_w8a8.yaml` |
| Qwen-Image-Layered | `configs/qwen_image_layered_int8_w8a8.yaml` |

## Quick start: Qwen-Image-Layered

```bash
comfy-quants export-model-w8a8 \
  --config configs/qwen_image_layered_int8_w8a8.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out /path/to/qwen_image_layered_int8_w8a8.safetensors \
  --device cuda:0 \
  --convrot \
  --hash-output \
  --json
```

Use `--no-convrot` for plain row-wise W8A8 (lower quality, slightly faster runtime).
Directory outputs use `diffusion_pytorch_model.int8_w8a8.safetensors`.

## Inputs

| Input | Argument | Description |
| --- | --- | --- |
| Config | `--config` | YAML selecting family, source, and `quant.target_dtype: int8_w8a8`. |
| Source | `--source` | Local transformer `.safetensors` / index JSON / indexed directory. |
| Output | `--out` | Output `.safetensors` path or directory. |
| Device | `--device` | Torch device; `auto` uses `cuda:0` when available, else CPU. |
| ConvRot | `--convrot` / `--no-convrot` | Regular-Hadamard weight rotation (default on). |
| Group size | `--convrot-groupsize` | ConvRot group (power of four; default 256). |

## Layer selection

The configs encode ComfyUI-INT8-Fast's `qwen` exclusion list (`time_text_embed`,
`img_in`, `norm_out`, `proj_out`, `txt_in`) as an exclude-driven policy (empty
`include` ⇒ all quantizable Linear pass the gate). The selected set is every
transformer-block Linear — and, unlike the FP8 flow, `transformer_blocks.0.img_mod.1`
**is** quantized (INT8-Fast does not special-case block 0).

## Loader prerequisite

The artifact loads in **ComfyUI-INT8-Fast** (`Load Diffusion Model INT8 (W8A8)` node
with on-the-fly quantization off), which must be installed in the ComfyUI Python
environment. It is not loaded by stock ComfyUI's native model loader.
