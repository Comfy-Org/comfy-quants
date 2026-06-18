# Anima family export (FP8 / MXFP8 / NVFP4)

Quantize the **Anima** diffusion model (ComfyUI's `cosmos_predict2` DiT + `llm_adapter`)
to the stock-ComfyUI-native formats. Anima is the first non-Qwen family; it reuses the
existing format export commands — only the model-family adapter is new.

## Supported formats & configs (v1)

| Format | Command | Config (2B) | Loads in |
| --- | --- | --- | --- |
| FP8 E4M3 | `export-model` | `configs/anima_2b_fp8.yaml` | stock ComfyUI (any GPU) |
| FP8 E5M2 | `export-model` | `configs/anima_2b_fp8_e5m2.yaml` | stock ComfyUI (any GPU) |
| MXFP8 | `export-model-mxfp8` | `configs/anima_2b_mxfp8.yaml` | stock ComfyUI (Blackwell SM≥10) |
| NVFP4 | `export-model-nvfp4` | `configs/anima_2b_nvfp4.yaml` | stock ComfyUI (Blackwell SM≥10) |

**INT8 W8A8 and INT4 are intentionally not supported for anima** — W8A8 targets the
ComfyUI-INT8-Fast node (which doesn't list anima) and INT4 is the mixed SVDQuant/AWQ path.

## Sizes

One architecture at two cosmos sizes, auto-detected by ComfyUI from `model_channels`
(`= x_embedder.proj.1.weight.shape[0]`):

| Family | model_channels | blocks | heads |
| --- | --- | --- | --- |
| `anima` | 2048 (2B) | 28 | 16 |
| `anima_14b` | 5120 (14B) | 36 | 40 |

Both are registered; the v1 configs target `anima` (2B). For 14B, copy a config and set
`model.family: anima_14b`.

## Quick start (MXFP8, 2B)

```bash
comfy-quants export-model-mxfp8 \
  --config configs/anima_2b_mxfp8.yaml \
  --source /path/to/anima/diffusion_pytorch_model.safetensors \
  --out /path/to/anima_2b_mxfp8.safetensors \
  --device cuda:0 \
  --hash-output \
  --json
```

The source is the released ComfyUI `diffusion_models` checkpoint, whose keys carry the
`net.` prefix (e.g. `net.blocks.0.self_attn.q_proj.weight`) — verified against
`circlestone-labs/Anima` (`split_files/diffusion_models/anima-base-v1.0.safetensors`).
FP8 uses `export-model`; NVFP4 uses `export-model-nvfp4`.

## Layer selection (convert_to_quant `anima` preset)

The adapter's default policy quantizes the **main-DiT transformer blocks** and keeps the
rest high precision:

- **Quantized**: `net.blocks.{2..N-1}.*` fully, plus `net.blocks.1`'s attention + MLP —
  every `self_attn`/`cross_attn` q/k/v/output projection, `mlp.layer1`/`layer2`, and the
  `adaln_modulation_*` Linears.
- **Kept high precision**: block 0 entirely, `net.blocks.1.adaln_modulation_*`,
  `net.final_layer`, `net.t_embedder`/`net.x_embedder`, all of `net.llm_adapter`, every norm.

That is **426** quantized Linears for 2B (28 blocks) / **554** for 14B (36 blocks). All
quantized in-features ∈ {model_channels, 1024, 256} are multiples of 16 and 32, so MXFP8
(group-32) and NVFP4 (group-16) both align with no padding.

## Verify

Before a full export, confirm the contract's `blocks.N...` names/shapes match your
checkpoint:

```bash
comfy-quants inspect --family anima --model /path/to/anima --json
```

The exporter's strict missing-tensor check also surfaces any name mismatch at export time
(the contract is authored from ComfyUI module source). For MXFP8/NVFP4, load the artifact
on a Blackwell ComfyUI to confirm native loading; FP8 loads on any GPU.
