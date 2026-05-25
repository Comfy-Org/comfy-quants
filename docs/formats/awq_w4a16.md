# AWQ W4A16 Format

This document defines the Comfy Quants artifact contract for AWQ W4A16 tensors used by INT4 model bundles.

The base library may write this checkpoint payload, but it must not import, embed, or launch ComfyUI or comfy-kitchen. Runtime support is expected from the target ComfyUI/comfy-kitchen installation.

## Format identifier

Layer-local checkpoint metadata is stored in the `comfy_quant` uint8 JSON tensor:

```json
{
  "format": "awq_w4a16",
  "group_size": 64
}
```

## Tensor family

| Tensor | Shape | Dtype | Required | Meaning |
| --- | --- | --- | --- | --- |
| `weight` | `(N, K/2)` | `int8` | yes | Two 4-bit weight values per byte. |
| `weight_scale` | `(K/64, N)` | fp16/bf16 | yes | Per-group scale. |
| `weight_zero` | `(K/64, N)` | fp16/bf16/fp32 | yes | Per-group additive floating-point zero/center. |
| `bias` | `(N,)` | fp16/bf16 | no | Linear bias. |
| `comfy_quant` | `(json_bytes,)` | uint8 | yes | Format metadata. |


## Kitchen-native quantization convention

The direct writer currently quantizes each output row independently with
asymmetric unsigned INT4 groups along the K axis. The byte packing is low nibble
first, high nibble second, and the packed `weight` tensor is stored as `int8`
bytes for safetensors compatibility.

The checkpoint/reference dequantization convention follows the target
kitchen-native AWQ W4A16 layout:

```text
dequant_weight[n, k] =
    (uint4_weight[n, k] - 8) * weight_scale[k / group_size, n]
    + weight_zero[k / group_size, n]
```

`uint4_weight` is unpacked to `[0, 15]`, centered to `[-8, 7]` by subtracting
`8`, then multiplied by the per-group scale.  `weight_zero` is an additive
floating-point group center with shape `(K/64, N)`, not an integer zero-point
nibble.  For direct dense quantization the writer fits each row/group min/max
range to those centered codes:

```text
scale = (w_max - w_min) / 15
zero  = w_min + 8 * scale
q     = clamp(round((w - zero) / scale), -8, 7) + 8
```

Constant groups use `scale = 1` and `zero = constant`, which reconstructs the
group at code `8`.  The formula is covered by the package reference helper, but
the overall Qwen mixed runtime still remains non-publishable until full external
load/inference validation is complete.

## Runtime fixture

The CLI can write a deterministic single-layer AWQ W4A16 fixture for external
runtime parity work:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main make-awq-runtime-fixture \
  --out runs/int4-runtime-fixtures/awq-w4a16 \
  --json
```

The fixture contains:

```text
fixture_layer.weight
fixture_layer.weight_scale
fixture_layer.weight_zero
fixture_layer.bias                 optional
fixture_layer.comfy_quant
fixture.input
fixture.expected_output
fixture.dequantized_weight
fixture.quantized_weight_uint4
```

The report is intentionally marked `external_runtime_validation: not_run` and
`publishable_svdquant_gptq: false`.  Passing the local self-check confirms only
the repository's kitchen-native AWQ formula.  It is a deterministic input for a
future external fused-runtime harness, not proof of ComfyUI mixed inference.

## Intended Qwen-Image-Edit usage

For the Qwen-Image-Edit INT4 kitchen-native bundle, AWQ W4A16 is intended for modulation linear layers:

```text
transformer_blocks.*.img_mod.1
transformer_blocks.*.txt_mod.1
```

Attention and MLP layers use SVDQuant W4A4 instead. This mixed bundle shape is intentional: it keeps the reusable formats independent while allowing the Qwen model adapter to select the correct format per layer family.

## Boundary

AWQ layout conversion from an external engine-specific checkpoint is backend/export-bridge work. The reusable format declaration only describes the checkpoint tensor family and metadata. It must not contain ComfyUI imports or hard-coded local checkout paths.
