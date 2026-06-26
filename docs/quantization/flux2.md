# FLUX.2 family export (FP8 / MXFP8 / NVFP4, + mixed precision)

Quantize the **FLUX.2** diffusion model (Black Forest Labs MMDiT, ComfyUI
`image_model="flux2"`) to the stock-ComfyUI-native formats. FLUX.2 is a distinct
architecture from FLUX.1 — see [`flux.md`](flux.md) for the latter.

## Supported formats & configs

| Format | Command | Config | Mixed-precision config |
| --- | --- | --- | --- |
| FP8 E4M3 | `export-model` | `configs/flux2_fp8.yaml` | `configs/flux2_fp8_mixed.yaml` |
| FP8 E5M2 | `export-model` | `configs/flux2_fp8_e5m2.yaml` | — |
| MXFP8 | `export-model-mxfp8` | `configs/flux2_mxfp8.yaml` | `configs/flux2_mxfp8_mixed.yaml` |
| NVFP4 | `export-model-nvfp4` | `configs/flux2_nvfp4.yaml` | `configs/flux2_nvfp4_mixed.yaml` |

MXFP8/NVFP4 load on stock ComfyUI (Blackwell SM≥10); FP8 on any GPU.

## Architecture

FLUX.2-dev: hidden=6144 (48 heads × 128), mlp_ratio=3 with **gated MLP** (`mlp.0` out = 2×
hidden), qkv_bias=False, **global modulation** (top-level `double_stream_modulation_img`/
`_txt`, `single_stream_modulation` — no per-block `mod.lin`), **8 double + 48 single**
blocks, out_channels=128. Bare `double_blocks.`/`single_blocks.` keys.

The default (full) policy quantizes **every block Linear** → **160** quantized Linears.
Contract validated 160/160 against the real `Comfy-Org/flux2-dev` header.

## Mixed precision (matches official `flux2_dev_fp8mixed`)

The shipped `flux2_dev_fp8mixed` keeps double-stream **attention** in bf16 and quantizes
only MLP + the fused single-stream linears. Reproduce it with the `*_mixed.yaml` configs, or
programmatically:

```python
adapter.default_policy("fp8_e4m3", mixed=True)   # excludes *_attn.qkv / *_attn.proj
```

This quantizes **128** Linears (vs 160). Trade-off: ~6% larger (attention stays bf16). Note:
on a 32B guidance-distilled model the pixel PSNR-vs-bf16 is dominated by sampling-trajectory
sensitivity, so the mixed recipe's value is **official-recipe alignment + a size/precision
choice**, not a measurable pixel-quality gain.

## Quick start (NVFP4, full)

```bash
comfy-quants export-model-nvfp4 \
  --config configs/flux2_nvfp4.yaml \
  --source /path/to/flux2-dev.safetensors \
  --out /path/to/flux2_nvfp4.safetensors \
  --device cuda:0 --hash-output --json
```

Source is the bf16 ComfyUI single-file (`black-forest-labs/FLUX.2-dev/flux2-dev.safetensors`,
gated). `comfy-quants inspect --family flux2 --model … --out … --json` validates names/shapes.

## Measured (RTX PRO 6000 Blackwell, 1024², 20 steps)

| Format | Model VRAM | Speed vs bf16 | Notes |
| --- | --- | --- | --- |
| FP8 E4M3 | 30.8 GB (−49%) | **1.97×** | native |
| MXFP8 | 33.0 GB | emulated (torch<2.10) | best fidelity (block scaling) |
| NVFP4 | **18.0 GB (−70%)** | **2.07×** | native; 32B is compute-bound → biggest speedup |
