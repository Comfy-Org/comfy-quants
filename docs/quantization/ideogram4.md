# Ideogram 4.0 family export (FP8 / MXFP8 / NVFP4) + format transcoding

Quantize **Ideogram 4.0** (open-weight 9.3B NextDiT/Lumina2 single-stream DiT, ComfyUI
`image_model="ideogram4"`) to the stock-ComfyUI-native formats.

## Supported formats & configs

| Format | Command | Config |
| --- | --- | --- |
| FP8 E4M3 | `export-model` | `configs/ideogram4_fp8.yaml` |
| FP8 E5M2 | `export-model` | `configs/ideogram4_fp8_e5m2.yaml` |
| MXFP8 | `export-model-mxfp8` | `configs/ideogram4_mxfp8.yaml` |
| NVFP4 | `export-model-nvfp4` | `configs/ideogram4_nvfp4.yaml` |

Loading Ideogram 4.0 needs a recent ComfyUI (≥0.26, which adds `comfy/ldm/ideogram4`) plus a
matching `comfy_kitchen` / `comfy_aimdo`. Text encoder is Qwen3-VL-8B; VAE is the FLUX.2 VAE.

## Architecture

emb_dim=4608 (18 heads × 256), 34 layers, intermediate=12288, adaln_dim=512. Each `layers.N`
has fused-qkv self-attention (`attention.qkv` / `attention.o`), a SwiGLU feed-forward
(`feed_forward.w1/w2/w3`), and an `adaln_modulation` Linear → 6 Linears/block. Bare `layers.`
keys. The default policy quantizes every block Linear → **204** quantized Linears; the
embedders / final layer / norms are kept high precision. Contract validated 204/204 against
the real `Comfy-Org/Ideogram-4/.../ideogram4_fp8_scaled.safetensors` header.

## No dense bf16 source → format transcoding

Ideogram publishes **only fp8/nvfp4** (`Comfy-Org/Ideogram-4`); there is no public dense
bf16 checkpoint. To produce a format Ideogram didn't ship (e.g. MXFP8), **transcode** from
the published fp8: dequantize fp8→bf16, then re-quantize via the normal pipeline.

```bash
# 1. fp8 -> approximate dense bf16 (dequant; strips ALL quant sidecars incl. comfy_quant markers)
python models/transcode_ideogram_fp8_to_bf16.py \
  ideogram4_fp8_scaled.safetensors  ideogram4_bf16_from_fp8.safetensors

# 2. bf16 -> mxfp8 via the standard exporter
comfy-quants export-model-mxfp8 \
  --config configs/ideogram4_mxfp8.yaml \
  --source ideogram4_bf16_from_fp8.safetensors \
  --out ideogram4_mxfp8.safetensors --device cuda:0 --json
```

Quality is capped by the fp8 source (already lossy), but the re-quantization is near-lossless
relative to it (measured requant SQNR ≈ 264 dB vs the fp8 source). The transcoded MXFP8 loads
and samples in ComfyUI. This is the same approach as the community `ideogram-4-int8-ConvRot`.

> **Gotcha (fixed in the transcode script)**: the published fp8 quantizes a few layers our
> contract keeps bf16. The dequant must strip those layers' `comfy_quant` markers too —
> otherwise they become orphans (fp8 marker + bf16 weight + no scale) and crash ComfyUI's
> loader.

## Measured (RTX PRO 6000 Blackwell, 1024², 20 steps; fp8 is the reference, no bf16 source)

| Format | Model VRAM | Speed | Notes |
| --- | --- | --- | --- |
| FP8 (official) | 8.85 GB | 2.35 it/s | reference baseline |
| NVFP4 (official) | 5.24 GB | 2.85 it/s | PSNR vs fp8 14.9 dB |
| MXFP8 (transcoded) | 9.40 GB | emulated | loads + samples; requant SQNR ≈264 dB vs fp8 |
