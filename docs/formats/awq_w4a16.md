# AWQ W4A16 format

This page defines the AWQ W4A16 tensor contract used by INT4 model bundles. Start
from [`../quantization/int4.md`](../quantization/int4.md) for export workflows
and model-family guides.

## How to produce this format

AWQ W4A16 is produced as part of supported mixed INT4 bundles. For
Qwen-Image-Edit-2511, the one-step INT4 export keeps AWQ modulation enabled by
default and writes the final single-file `svdquant_w4a4` tile-pack checkpoint.

```bash
comfy-quants qwen-image-edit-2511-int4 \
  --model /path/to/Qwen-Image-Edit-2511 \
  --base-checkpoint /path/to/qwen_image_edit_2511_bf16_transformer.safetensors \
  --out /path/to/qwen_image_edit_2511_int4_tilepack.safetensors \
  --deepcompressor-root /path/to/DeepCompressor \
  --nunchaku-root /path/to/nunchaku \
  --calibration-samples 128 \
  --search-strength quality-r64 \
  --awq-group-size 64 \
  --gpus 0 \
  --hash-output \
  --json
```

Use `--no-awq-modulation` only when you intentionally want to disable the AWQ
modulation bridge path for a compatible export route. The final artifact is still
the INT4 tile-pack checkpoint; AWQ W4A16 describes the modulation-layer tensor
family inside that bundle.

Full guide: [`../quantization/qwen_image_edit_2511_int4.md`](../quantization/qwen_image_edit_2511_int4.md)

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

AWQ W4A16 is for model-family-selected linear layers when the target runtime
supports a mixed INT4 bundle. The exact layer patterns belong in the
model-family guide for the selected INT4 workflow.

Other layer families in the same bundle may use the SVDQuant W4A4 format
described in [`svdquant_w4a4_kitchen_tilepack.md`](svdquant_w4a4_kitchen_tilepack.md).
