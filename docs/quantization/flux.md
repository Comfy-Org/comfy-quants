# FLUX.1 family export (FP8 / MXFP8 / NVFP4)

Quantize the **FLUX.1** diffusion model (Black Forest Labs MMDiT, ComfyUI
`image_model="flux"`) to the stock-ComfyUI-native formats. Reuses the existing format
export commands — only the model-family adapter is new.

## Supported formats & configs

| Format | Command | Config | Loads in |
| --- | --- | --- | --- |
| FP8 E4M3 | `export-model` | `configs/flux_fp8.yaml` | stock ComfyUI (any GPU) |
| FP8 E5M2 | `export-model` | `configs/flux_fp8_e5m2.yaml` | stock ComfyUI (any GPU) |
| MXFP8 | `export-model-mxfp8` | `configs/flux_mxfp8.yaml` | stock ComfyUI (Blackwell SM≥10) |
| NVFP4 | `export-model-nvfp4` | `configs/flux_nvfp4.yaml` | stock ComfyUI (Blackwell SM≥10) |

> **FLUX.2 is a different architecture** (`image_model="flux2"`) — see [`flux2.md`](flux2.md).

## Architecture

FLUX.1-dev MMDiT: `double_blocks.N` (img+txt streams, 10 Linears each) + `single_blocks.N`
(fused qkv+mlp, 3 Linears each). hidden=3072, 24 heads, **19 double + 38 single** blocks,
mlp_ratio=4. Bare `double_blocks.`/`single_blocks.` keys (no wrapper prefix).

The default policy quantizes **every block Linear** (qkv, proj, mlp.0/2, mod.lin, single
linear1/linear2/modulation.lin) and keeps `img_in`/`txt_in`/`time_in`/`vector_in`/
`guidance_in`/`final_layer` high precision → **304** quantized Linears. All quantized
in-features are multiples of 16 and 32, so MXFP8 (group-32) and NVFP4 (group-16) align with
no padding. Contract validated 304/304 against the real `Comfy-Org/flux1-dev/flux1-dev.safetensors`
header (layer name + shape).

## Quick start (NVFP4)

```bash
comfy-quants export-model-nvfp4 \
  --config configs/flux_nvfp4.yaml \
  --source /path/to/flux1-dev.safetensors \
  --out /path/to/flux_nvfp4.safetensors \
  --device cuda:0 --hash-output --json
```

The source is the bf16 ComfyUI single-file (`Comfy-Org/flux1-dev/flux1-dev.safetensors`).
FP8 uses `export-model`; MXFP8 uses `export-model-mxfp8`.

## Verify

```bash
comfy-quants inspect --family flux --model /path/to/flux1-dev --out /tmp/flux_inspect --json
```

The exporter's strict missing-tensor check surfaces any name mismatch at export time. For
MXFP8/NVFP4, load the artifact on a Blackwell ComfyUI; FP8 loads on any GPU.

## Measured (RTX PRO 6000 Blackwell, 1024², 20 steps)

| Format | Model disk | Speed vs bf16 | Notes |
| --- | --- | --- | --- |
| FP8 E4M3 | −50% | ≈1.0× | native scaled_mm |
| MXFP8 | −48% | emulated (torch<2.10) | best image fidelity |
| NVFP4 | **−71%** | **1.49×** | native nvfp4 GEMM |
