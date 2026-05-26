# AWQ W4A16 format

This page defines the AWQ W4A16 tensor contract used by INT4 model bundles. User
commands are documented in [`../quantization/int4_tools.md`](../quantization/int4_tools.md).

## Identifier

Layer metadata is stored as a uint8 JSON tensor named `<layer>.comfy_quant`:

```json
{
  "format": "awq_w4a16",
  "group_size": 64
}
```

## Tensor family

| Tensor | Shape | Dtype | Required | Meaning |
| --- | --- | --- | --- | --- |
| `weight` | `(N, K/2)` | `int8` | yes | two 4-bit weight values per byte |
| `weight_scale` | `(K/64, N)` | fp16/bf16/fp32 | yes | per-group scale |
| `weight_zero` | `(K/64, N)` | fp16/bf16/fp32 | yes | per-group additive center |
| `bias` | `(N,)` | fp16/bf16/fp32 | no | linear bias |
| `comfy_quant` | `(json_bytes,)` | uint8 | yes | metadata |

## Dequantization convention

Packed weights are unpacked to unsigned 4-bit values in `[0, 15]`. The reference
layout uses centered codes:

```text
dequant_weight[n, k] =
    (uint4_weight[n, k] - 8) * weight_scale[k / group_size, n]
    + weight_zero[k / group_size, n]
```

`weight_zero` is a floating-point group center with shape `(K/64, N)`.

## Intended model usage

For Qwen-Image-Edit INT4 bundles, AWQ W4A16 is intended for modulation linear
layers when the target runtime supports that mixed layout:

```text
transformer_blocks.*.img_mod.1
transformer_blocks.*.txt_mod.1
```

Attention and MLP linears use the SVDQuant W4A4 format described in
[`svdquant_w4a4_kitchen_tilepack.md`](svdquant_w4a4_kitchen_tilepack.md).
