# MXFP8 (OCP microscaling FP8) checkpoint format

This page defines the **MXFP8** checkpoint format produced by Comfy Quants for
**stock ComfyUI's native** quantized loader. User commands are in
[`../quantization/mxfp8.md`](../quantization/mxfp8.md).

MXFP8 stores `float8_e4m3fn` weights with one **E8M0 power-of-2 block scale per 32
consecutive elements** (along the input dim), the block-scale grid laid out in the
cuBLAS `to_blocked` swizzle. ComfyUI loads it through `QUANT_ALGOS["mxfp8"]` and
`TensorCoreMXFP8Layout` — the **same per-layer `comfy_quant` handshake as the FP8
path**, so MXFP8 is the native-loader sibling of FP8 (unlike INT8 W8A8 / INT4,
which target downstream custom nodes).

## Scope: producer only

Comfy Quants produces the checkpoint. The mxfp8 tensor-core matmul + dynamic
activation quantization live in ComfyUI/comfy-kitchen and are out of scope here.
The whole quantize pipeline (E8M0 block scaling + the `to_blocked` swizzle + the
FP8 encode) is reproduced in **pure torch** (`formats/mxfp8_blocked.py`), so the
library needs no comfy_kitchen dependency.

## Format identifier

| Comfy Quants target | Weight | Weight scale | Activations | Marker |
| --- | --- | --- | --- | --- |
| `mxfp8` | `torch.float8_e4m3fn`, `[out, in]` | E8M0 block-32, `uint8` on disk (swizzled) | dynamic (runtime) | `comfy_quant` |

## Numeric convention (bit-faithful to `comfy/float.py`)

Block size 32 along the input dim; one E8M0 (8-bit power-of-2 exponent, bias 127)
scale per block; deterministic round-to-nearest-even into FP8-E4M3:

```text
F8_E4M3_MAX = 448.0 ; E8M0_BIAS = 127 ; BLOCK = 32
xb       = w.reshape(out, in//32, 32).float()
max_abs  = xb.abs().amax(dim=-1)                                   # [out, in/32]
need     = (max_abs / 448.0).clamp(min=2**-127)
exp      = (ceil(log2(need)).int() + 127).clamp(0, 254)            # E8M0 exponent
e8m0     = exp.to(uint8)
zero     = (max_abs == 0)
sf32     = ((e8m0.int() << 23)).view(float32) ; sf32[zero] = 1.0   # 2^(e-127)
q        = (xb / sf32).reshape(out, in).clamp(-448, 448).to(float8_e4m3fn)
e8m0[zero] = 0
weight        = q                              # float8_e4m3fn [out, in]
weight_scale  = to_blocked(e8m0)               # uint8, swizzled
```

This matches ComfyUI's reference quantizer (`comfy/float.py`
`stochastic_round_quantize_mxfp8_by_block`) except we use deterministic rounding
instead of stochastic. The `to_blocked` swizzle is the cuBLAS d-block-scaling-factor
layout (`comfy/float.py` `to_blocked`).

## Layer side tensors

For each quantized Linear `<layer>` (`in_features` must be a multiple of 32 — true
for the Qwen-Image families):

```text
<layer>.weight        float8_e4m3fn tensor, shape [out_features, in_features]
<layer>.weight_scale  uint8 tensor, E8M0 block scales in the to_blocked swizzle,
                      shape (128*ceil(out/128), 4*ceil((in/32)/4))
<layer>.comfy_quant   uint8 JSON marker: {"format": "mxfp8"}
```

There is **no `<layer>.input_scale`** (activations are quantized dynamically at
runtime) and bias is copied through unquantized.

## Loader handshake

ComfyUI's `MixedPrecisionOps` Linear `_load_from_state_dict` reads the per-layer
`comfy_quant` marker, dispatches on `format == "mxfp8"` to `QUANT_ALGOS["mxfp8"]`
(`storage_t = float8_e4m3fn`, `comfy_tensor_layout = TensorCoreMXFP8Layout`,
`group_size = 32`), loads `weight_scale` as `uint8` then `.view(torch.float8_e8m0fnu)`,
and builds a `QuantizedTensor`. (A top-level `_quantization_metadata` header is the
alternative ComfyUI also accepts; we emit the per-layer marker like the FP8 writer.)

## Runtime gate

The mxfp8 tensor-core matmul requires **NVIDIA Blackwell (SM ≥ 10) + torch ≥ 2.10 +
comfy_kitchen**. On unsupported hardware ComfyUI **silently dequantizes** the weight
to the compute dtype — the checkpoint still loads and is numerically correct, just
without the quantized-matmul speedup. This is a consumer property, not a producer
choice; the artifact is identical.

## Scope note

The format is reusable across model families; layer selection lives in
`model_adapters` / the config `quant.modules` (it mirrors the FP8 selection). The
pure-torch swizzle + E8M0 math is bit-for-bit parity-tested against `comfy/float.py`
in `tests/unit/test_external_mxfp8_parity.py` (gated on `COMFY_QUANTS_COMFYUI_SOURCE`).
