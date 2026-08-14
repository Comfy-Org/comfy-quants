# MiniMax Music 3 NVFP4

This flow produces two stock-ComfyUI-native NVFP4 checkpoints for MiniMax Music 3:
one for the diffusion transformer and one for the autoregressive text/audio encoder.
The DAV remains in its published precision. ComfyUI loads the outputs through its
native `comfy_quant` / `TensorCoreNVFP4Layout` path; no custom node is required.

## Model identity

Use the Music 3 artifacts from `Comfy-Org/MiniMax-Music-3`, pinned to revision
`6444666eb6edfb2c7fcab5f8b81da8b84b4b17b6`:

| Component | Published source | Suggested output |
| --- | --- | --- |
| Diffusion transformer | `diffusion_models/minimax_music3_dit_fp16.safetensors` | `minimax_music3_dit_nvfp4.safetensors` |
| AR text/audio encoder | `text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors` | `minimax_music3_text_encoder_pruned_nvfp4.safetensors` |
| Audio decoder (unchanged) | `vae/minimax_music3_dav.safetensors` | use the published file |

Do not substitute a checkpoint named simply “MiniMax M3.” MiniMax M3 is a separate
multimodal language/vision architecture (`MiniMaxM3SparseForConditionalGeneration`),
not the Music 3 AR+DiT model, and cannot be loaded by the Music 3 nodes.

## Export

```bash
comfy-quants export-model-nvfp4 \
  --config configs/minimax_music3_dit_nvfp4.yaml \
  --source /path/to/minimax_music3_dit_fp16.safetensors \
  --out /path/to/minimax_music3_dit_nvfp4.safetensors \
  --device cuda:0 \
  --hash-output \
  --json

comfy-quants export-model-nvfp4 \
  --config configs/minimax_music3_text_encoder_nvfp4.yaml \
  --source /path/to/minimax_music3_text_encoder_pruned_bf16.safetensors \
  --out /path/to/minimax_music3_text_encoder_pruned_nvfp4.safetensors \
  --device cuda:0 \
  --hash-output \
  --json
```

The exporter is deterministic and can run on CPU, but CUDA is substantially faster.
It copies every unselected tensor through and writes per-layer `comfy_quant` markers,
packed FP4 weights, blocked FP8 scales, and FP32 tensor scales for selected Linears.

## Layer policy

The DiT recipe quantizes four Linear weights per transformer block across 36 blocks
(144 weights): fused QKV, attention output, SwiGLU input, and feed-forward output.

The text-encoder recipe quantizes the same four categories in each of the 36 main AR
blocks and four RVQ depth-decoder blocks (160 weights). Token/code embeddings,
`lm_head_pruned`, RVQ logits heads, RVQ input projection, norms, and tokenizer bytes
remain high precision. This conservative boundary policy avoids quantizing logits and
embeddings before an audio-quality comparison has been completed.

## Runtime requirements and validation

Native NVFP4 matrix multiplication requires an NVIDIA Blackwell GPU (SM 10.0 or
newer), a current CUDA/PyTorch stack, and `comfy-kitchen` with `scaled_mm_nvfp4`.
ComfyUI can load the files on other hardware by dequantizing, but that does not test
the intended fast path.

Before distributing or serving the artifacts:

1. load both NVFP4 files with stock ComfyUI and confirm its startup capabilities list
   `nvfp4` as native rather than emulated;
2. run short and long Music 3 generations with fixed prompts/seeds against the
   published FP16/BF16 or INT8-ConvRot files;
3. compare termination behavior, duration, silence/clipping, and perceptual audio
   quality, then stress consecutive generations for VRAM/offload regressions;
4. record ComfyUI, comfy-kitchen, PyTorch, CUDA, driver, and GPU versions with the
   resulting hashes.

The static contracts and exporter tests validate names, shapes, selection, packing,
and loader metadata. They are not a substitute for Blackwell inference or audio-
quality validation.

## License

The `Comfy-Org` repository is a repack of the original MiniMax weights. Follow the
original MiniMax Music 3 license and acceptable-use terms when using or redistributing
the resulting checkpoint; repository metadata on a repack should not be treated as a
replacement license. In particular, review the original commercial-use attribution,
notice/authorization, and hosted-service safeguard requirements before deployment.
