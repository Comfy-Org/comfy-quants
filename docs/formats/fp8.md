# FP8 checkpoint formats

This page defines the FP8 checkpoint format mapping used by Comfy Quants exports.
User commands are documented in [`../quantization/fp8.md`](../quantization/fp8.md).

## Format identifiers

| Comfy Quants target | Torch dtype | Safetensors dtype | ComfyUI checkpoint metadata |
| --- | --- | --- | --- |
| `fp8_e4m3` | `torch.float8_e4m3fn` | `F8_E4M3` | `float8_e4m3fn` |
| `fp8_e5m2` | `torch.float8_e5m2` | `F8_E5M2` | `float8_e5m2` |

`fp8_e5m2` is not `mxfp8` and is not `fp8_e4m3fn_fast`.

## Layer side tensors

For each quantized weight, the full checkpoint exporter writes the FP8 weight and
small side tensors used by the target loader:

```text
<layer>.weight        FP8 tensor
<layer>.weight_scale  scale tensor
<layer>.input_scale   scale tensor
<layer>.comfy_quant   uint8 JSON metadata tensor
```

The `comfy_quant` metadata identifies the stored FP8 checkpoint format and whether
full-precision matrix multiplication is requested by the target loader.

## Scope

The FP8 format definition is reusable across model families. Model-family layer
selection belongs in `model_adapters/`, and command usage belongs in
`docs/quantization/fp8.md`.
