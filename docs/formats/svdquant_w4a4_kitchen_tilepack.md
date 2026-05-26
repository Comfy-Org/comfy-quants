# SVDQuant W4A4 kitchen tile-pack format

This page defines the reusable SVDQuant W4A4 storage contract. Start from
[`../quantization/int4.md`](../quantization/int4.md) for export workflows and
model-family guides.

## Identifiers

| Field | Value |
| --- | --- |
| Quant format | `svdquant_w4a4` |
| Storage layout | `kitchen_tile_packed_w4a4` |
| Weight storage dtype | `int8` bytes containing two signed INT4 values |
| Weight value range | `[-8, 7]` |
| Group size | `64` input features |
| N tile size | `128` output features |
| Interleave | `4` |

Layer metadata is stored as a uint8 JSON tensor named `<layer>.comfy_quant`:

```json
{
  "format": "svdquant_w4a4",
  "layout": "kitchen_tile_packed_w4a4"
}
```

Optional metadata fields include:

```json
{
  "act_unsigned": true,
  "lowrank_branch_input_basis": "raw",
  "proj_down_smooth_folded": true
}
```

## Signed INT4 packing

Two signed INT4 values are stored in one byte:

```text
low nibble  = first value  & 0x0F
high nibble = second value & 0x0F
byte        = low | (high << 4)
```

When unpacking, nibble values `8..15` map back to signed values by subtracting `16`.

## Natural tensor family

A natural-layout SVDQuant linear layer uses:

| Tensor | Shape | Dtype | Required | Meaning |
| --- | --- | --- | --- | --- |
| `weight` | `(N, K/2)` | `int8` | yes | signed INT4 pairs for an `(N, K)` logical matrix |
| `weight_scale` | `(K/64, N)` | fp16/bf16/fp32 | yes | per-group weight scale |
| `smooth_factor` | `(K,)` | fp16/bf16/fp32 | yes | activation smoothing factor |
| `proj_down` | `(K, R)` | fp16/bf16/fp32 | yes | low-rank down projection |
| `proj_up` | `(N, R)` | fp16/bf16/fp32 | yes | low-rank up projection |
| `bias` | `(N,)` | fp16/bf16/fp32 | no | linear bias |
| `comfy_quant` | `(json_bytes,)` | uint8 | yes | metadata |

Where `N = out_features`, `K = in_features`, and `R = low-rank rank`.

## Tile-packed tensor family

| Tensor | Natural shape | Tile-packed shape | Notes |
| --- | --- | --- | --- |
| `weight` | `(N, K/2)` | `(N/128, K/64, 32, 128)` | requires `N % 128 == 0` and `K % 64 == 0` |
| `weight_scale` | `(K/64, N)` | `(N/128, K/64, 128)` | packs the N axis |
| `smooth_factor` | `(K,)` | unchanged | natural layout |
| `proj_down` | `(K, R)` | unchanged | natural layout |
| `proj_up` | `(N, R)` | `(N/128, R, 128)` | packs the N axis |
| `bias` | `(N,)` | unchanged | optional |
| `comfy_quant` | uint8 JSON | uint8 JSON | must include `format` and `layout` |

The fixed `weight` tile tail is:

```text
(128 / 4, 4 * 64 / 2) = (32, 128)
```

## How model adapters use this format

A model adapter maps selected linear layer families to SVDQuant W4A4. Other
layer families may stay high precision or use another format such as AWQ W4A16.
QKV splitting and model-family-specific tensor names are handled by the model
adapter or export bridge.
