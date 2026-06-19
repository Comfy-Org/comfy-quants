# NVFP4 (FP4 E2M1 microscaling) checkpoint format

This page defines the **NVFP4** checkpoint format produced by Comfy Quants for
**stock ComfyUI's native** quantized loader. User commands are in
[`../quantization/nvfp4.md`](../quantization/nvfp4.md).

NVFP4 is NVIDIA's 4-bit microscaling format: **FP4-E2M1** elements packed 2-per-byte,
with **two-level scaling** — a per-block-16 FP8-E4M3 scale (cuBLAS `to_blocked`
swizzle) and a per-tensor FP32 scale. ComfyUI loads it through `QUANT_ALGOS["nvfp4"]`
and `TensorCoreNVFP4Layout` — the **same per-layer `comfy_quant` handshake as
FP8/MXFP8**, so NVFP4 is another native-loader format (unlike INT8 W8A8 / INT4).

## Scope: producer only

Comfy Quants produces the checkpoint. The nvfp4 tensor-core matmul + dynamic
activation quantization live in ComfyUI/comfy-kitchen. The whole quantize pipeline
(E2M1 round-half-to-even encode + nibble pack + two-level scale + `to_blocked`
swizzle) is reproduced in **pure torch** (`formats/nvfp4_blocked.py`), so the library
needs no comfy_kitchen dependency.

## Format identifier

| Comfy Quants target | Weight | Block scale | Per-tensor scale | Activations | Marker |
| --- | --- | --- | --- | --- | --- |
| `nvfp4` | `uint8` `[out, in/2]` (FP4-E2M1, 2/byte) | `float8_e4m3fn`, block-16, swizzled | `float32` scalar | dynamic (runtime) | `comfy_quant` |

E2M1 grid (signed): `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`, `F4_E2M1_MAX=6`. Nibble =
`(sign<<3)|(exp<<1)|mantissa`; pack order HIGH=even index, LOW=odd, along in.

## Numeric convention (bit-faithful to comfy-kitchen eager `quantize_nvfp4`)

```text
F4_E2M1_MAX=6.0 ; F8_E4M3_MAX=448.0 ; BLOCK=16
per_tensor   = amax(|W|) / (448*6)                                  # fp32 scalar (weight_scale_2)
block_amax   = |W.reshape(out, in/16, 16)|.amax(-1)                 # [out, in/16]
scaled_fp8   = clamp((block_amax/6) / per_tensor, max=448)          # stored (-> fp8) block scale
total        = per_tensor * float8_round(scaled_fp8)
x_scaled     = clamp(W / total, -6, 6)
weight       = pack_uint4(f32_to_floatx_unpacked(x_scaled, ebits=2, mbits=1))   # uint8 [out, in/2]
weight_scale = to_blocked(scaled_fp8.to(float8_e4m3fn))            # fp8 swizzled
weight_scale_2 = per_tensor                                        # fp32 scalar
```

Deterministic round-to-nearest-even (`f32_to_floatx_unpacked` is the torchao-derived
bit-level encoder). Dequant: `value = e2m1(nibble) · weight_scale · weight_scale_2`.

## Layer side tensors

For each quantized Linear `<layer>` (`in_features` must be a multiple of 16 — true for
the Qwen-Image families):

```text
<layer>.weight          uint8 tensor, shape [out_features, in_features//2]  (FP4-E2M1 packed)
<layer>.weight_scale    float8_e4m3fn tensor, block-16 scales, to_blocked swizzle,
                        shape (128*ceil(out/128), 4*ceil((in/16)/4))
<layer>.weight_scale_2  float32 scalar (0-dim), per-tensor scale = amax(|W|)/(448*6)
<layer>.comfy_quant     uint8 JSON marker: {"format": "nvfp4"}
```

There is **no `<layer>.input_scale`** (activations are runtime-dynamic) and bias is
copied through unquantized.

## Loader handshake

ComfyUI's `MixedPrecisionOps` Linear reads the `comfy_quant` marker, dispatches on
`format == "nvfp4"` to `QUANT_ALGOS["nvfp4"]` (`storage_t=uint8`,
`comfy_tensor_layout=TensorCoreNVFP4Layout`, `group_size=16`,
`parameters={weight_scale, weight_scale_2, input_scale}`), loads `weight_scale_2`
(fp32, per-tensor) and `weight_scale` (block, as `float8_e4m3fn`), and builds a
`QuantizedTensor` with `Params(scale=weight_scale_2, block_scale=weight_scale)`.

## Runtime gate

The nvfp4 tensor-core matmul requires **NVIDIA Blackwell (SM ≥ 10) + comfy_kitchen**
(`supports_nvfp4_compute`). On unsupported hardware ComfyUI **silently dequantizes**
the weight to the compute dtype — the checkpoint still loads and is numerically
correct, just without the quantized-matmul speedup. This is a consumer property, not a
producer choice; the artifact is identical.

## Scope note

The format is reusable across model families; layer selection mirrors the FP8 policy
(see the workflow page). The pure-torch E2M1 encoder + pack + swizzle are bit-for-bit
parity-tested against comfy-kitchen `float_utils` in
`tests/unit/test_external_nvfp4_parity.py` (gated on `COMFY_QUANTS_COMFY_KITCHEN_SOURCE`).
