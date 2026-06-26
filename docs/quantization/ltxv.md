# LTX-Video (LTXV) family export (FP8 / MXFP8 / NVFP4)

Quantize the **LTX-Video** diffusion model (Lightricks video DiT, ComfyUI
`image_model="ltxv"`) to the stock-ComfyUI-native formats.

## Supported formats & configs

| Format | Command | Config | Loads in |
| --- | --- | --- | --- |
| FP8 E4M3 | `export-model` | `configs/ltxv_fp8.yaml` | stock ComfyUI (any GPU) |
| FP8 E5M2 | `export-model` | `configs/ltxv_fp8_e5m2.yaml` | stock ComfyUI (any GPU) |
| MXFP8 | `export-model-mxfp8` | `configs/ltxv_mxfp8.yaml` | stock ComfyUI (Blackwell SM≥10) |
| NVFP4 | `export-model-nvfp4` | `configs/ltxv_nvfp4.yaml` | stock ComfyUI (Blackwell SM≥10) |

## Architecture

LTXV-2B (0.9.x): single-stream video DiT, `transformer_blocks.N` each with self-attention
(`attn1`), cross-attention to the text caption (`attn2`), and a GELU feed-forward
(`ff.net.0.proj` / `ff.net.2`) — 10 Linears/block. inner_dim=2048, 32 heads, head_dim=64,
**28 blocks**, cross_attention_dim=2048.

> **Key prefix**: the released ComfyUI single-file (`Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors`)
> stores every diffusion tensor under a **`model.diffusion_model.`** prefix (unlike flux/ideogram
> which are bare). The contract/configs use that prefix. The single-file also bundles the VAE, so
> exporting from it yields a full checkpoint loadable via `CheckpointLoaderSimple`.

The default policy quantizes every block Linear → **280** quantized Linears (attn1/attn2
q/k/v/out + ff). Kept high precision: `patchify_proj`, `adaln_single`, `caption_projection`,
`proj_out`, `scale_shift_table`. Contract validated 280/280 against the real header.

## Quick start (MXFP8)

```bash
comfy-quants export-model-mxfp8 \
  --config configs/ltxv_mxfp8.yaml \
  --source /path/to/ltx-video-2b-v0.9.safetensors \
  --out /path/to/ltxv_mxfp8.safetensors \
  --device cuda:0 --hash-output --json
```

`comfy-quants inspect --family ltxv --model … --out … --json` validates names/shapes.

## Measured (RTX PRO 6000 Blackwell, 512²×25 frames, 20 steps)

| Format | Model VRAM | Speed vs bf16 | Notes |
| --- | --- | --- | --- |
| FP8 E4M3 | 1.88 GB (−49%) | 0.66× | |
| MXFP8 | 1.93 GB | emulated | best fidelity |
| NVFP4 | **1.09 GB (−70%)** | 0.51× | |

> On a small (2B) model at low resolution, quantization **saves VRAM but is slower** — the
> per-layer quant overhead is not amortized on small matmuls (the opposite regime from the
> 12B/32B flux models). Pick the format by model size: small models quantize mainly for VRAM.
