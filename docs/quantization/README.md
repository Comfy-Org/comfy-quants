# Quantization guides

Choose the output format first. Each format section then lists the supported
model-family flows for that format. Tensor/storage definitions live under
[`../formats/`](../formats/).

## Format matrix

| Output format | Workflow guide | Format reference | Model-family flows |
| --- | --- | --- | --- |
| FP8 E4M3 / E5M2 checkpoint | [`fp8.md`](fp8.md) | [`../formats/fp8.md`](../formats/fp8.md) | listed in the FP8 guide |
| INT8 W8A8 (+ConvRot) checkpoint | [`int8_w8a8.md`](int8_w8a8.md) | [`../formats/int8_w8a8.md`](../formats/int8_w8a8.md) | listed in the INT8 W8A8 guide (ComfyUI-INT8-Fast) |
| MXFP8 microscaling checkpoint | [`mxfp8.md`](mxfp8.md) | [`../formats/mxfp8.md`](../formats/mxfp8.md) | listed in the MXFP8 guide (stock ComfyUI, Blackwell) |
| NVFP4 microscaling checkpoint | [`nvfp4.md`](nvfp4.md) | [`../formats/nvfp4.md`](../formats/nvfp4.md) | listed in the NVFP4 guide (stock ComfyUI, Blackwell) |
| INT4 SVDQuant W4A4 tile-pack | [`int4.md`](int4.md) | [`../formats/svdquant_w4a4_kitchen_tilepack.md`](../formats/svdquant_w4a4_kitchen_tilepack.md) | listed in the INT4 guide |
| INT4 AWQ W4A16 tensors | [`int4.md`](int4.md) | [`../formats/awq_w4a16.md`](../formats/awq_w4a16.md) | used by supported mixed INT4 bundles |

## FP8

Start with [`fp8.md`](fp8.md) when the target artifact is a full FP8 checkpoint.
The page lists the supported config files and model-family examples for E4M3 and
E5M2 exports.

Command pattern:

```bash
comfy-quants export-model \
  --config /path/to/fp8_config.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out /path/to/model_fp8.safetensors \
  --device cuda:0 \
  --hash-output \
  --json
```

## INT8 W8A8 (ComfyUI-INT8-Fast)

Start with [`int8_w8a8.md`](int8_w8a8.md) to produce an INT8 W8A8 prequantized
checkpoint (int8 weights + optional ConvRot rotation) for the downstream
ComfyUI-INT8-Fast node, which adds dynamic int8 activations + an int8 matmul at
runtime. Command pattern:

```bash
comfy-quants export-model-w8a8 \
  --config /path/to/int8_w8a8_config.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out /path/to/model_int8_w8a8.safetensors \
  --device cuda:0 \
  --convrot \
  --hash-output \
  --json
```

## MXFP8 (stock ComfyUI native, Blackwell)

Start with [`mxfp8.md`](mxfp8.md) to produce an MXFP8 (OCP microscaling FP8)
checkpoint for **stock ComfyUI's native** loader — `float8_e4m3fn` weights + E8M0
block-32 scales (cuBLAS `to_blocked` swizzle), the same per-layer `comfy_quant`
handshake as FP8. The mxfp8 tensor-core matmul runs on Blackwell (SM ≥ 10); other
hardware silently dequantizes. Command pattern:

```bash
comfy-quants export-model-mxfp8 \
  --config /path/to/mxfp8_config.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out /path/to/model_mxfp8.safetensors \
  --device cuda:0 \
  --hash-output \
  --json
```

## NVFP4 (stock ComfyUI native, Blackwell)

Start with [`nvfp4.md`](nvfp4.md) to produce an NVFP4 (FP4 E2M1 microscaling)
checkpoint for **stock ComfyUI's native** loader — packed FP4-E2M1 weights + per-block-16
FP8-E4M3 block scales (cuBLAS `to_blocked` swizzle) + a per-tensor FP32 scale, the same
per-layer `comfy_quant` handshake as FP8/MXFP8. The nvfp4 tensor-core matmul runs on
Blackwell (SM ≥ 10); other hardware silently dequantizes. Command pattern:

```bash
comfy-quants export-model-nvfp4 \
  --config /path/to/nvfp4_config.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out /path/to/model_nvfp4.safetensors \
  --device cuda:0 \
  --hash-output \
  --json
```

## Model families

Format guides above are family-agnostic. For per-family flows beyond Qwen-Image, see:

- **Anima** (cosmos_predict2 DiT + llm_adapter) — FP8 / MXFP8 / NVFP4:
  [`anima.md`](anima.md).
- **FLUX.1** (MMDiT, double + single blocks) — FP8 / MXFP8 / NVFP4:
  [`flux.md`](flux.md).
- **FLUX.2** (MMDiT, gated MLP + global modulation; mixed-precision recipe) — FP8 / MXFP8 / NVFP4:
  [`flux2.md`](flux2.md).
- **LTX-Video** (Lightricks video DiT; `model.diffusion_model.` prefix) — FP8 / MXFP8 / NVFP4:
  [`ltxv.md`](ltxv.md).
- **Ideogram 4.0** (open-weight NextDiT/Lumina2; fp8→mxfp8 transcoding) — FP8 / MXFP8 / NVFP4:
  [`ideogram4.md`](ideogram4.md).

## INT4

Start with [`int4.md`](int4.md) when the target artifact is an INT4 tile-pack
checkpoint. The page links to:

- model-family one-step export guides;
- built-in solver usage;
- artifact inspection and repack tools;
- SVDQuant W4A4 and AWQ W4A16 format references.

For the currently supported Qwen-Image-Edit-2511 one-step flow:

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

## Validation flow

1. Export the checkpoint with the selected format workflow.
2. Run the relevant inspector, for example `comfy-quants inspect-int4` for INT4
   tile-pack artifacts.
3. Load the exported `.safetensors` file in a compatible ComfyUI setup and run an
   inference workflow to validate the generated image.
