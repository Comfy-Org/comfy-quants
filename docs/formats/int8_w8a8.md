# INT8 W8A8 (+ ConvRot) checkpoint format

This page defines the INT8 **W8A8** checkpoint format produced by Comfy Quants for
the downstream **ComfyUI-INT8-Fast** custom node. User commands are in
[`../quantization/int8_w8a8.md`](../quantization/int8_w8a8.md).

W8A8 means 8-bit **weights** and 8-bit **activations**: Comfy Quants writes the
int8 weights offline; the activations are quantized *dynamically per token at
runtime* by the downstream node, which then runs a real `int8×int8→int32` matmul
(faster than bf16). Optional **ConvRot** rotates each weight (regular Hadamard,
group 256) before quantization to spread channel outliers (near-GGUF-Q8 quality).

## Scope: producer only

Comfy Quants produces the **prequantized checkpoint**. The runtime — dynamic
activation int8, online activation rotation, the Triton W8A8 kernel — lives in the
ComfyUI-INT8-Fast node and is out of scope here. The artifact is consumed by that
node's `Int8TensorwiseOps` loader (with on-the-fly quantization off), **not** stock
ComfyUI's native `QUANT_ALGOS` path — the same downstream-loader pattern as INT4.

## Format identifier

| Comfy Quants target | Storage | Weight scale | Activations | Marker |
| --- | --- | --- | --- | --- |
| `int8_w8a8` | `torch.int8` | `float32`, per output channel | dynamic int8 (runtime) | `comfy_quant` |

## Numeric convention (bit-faithful to ComfyUI-INT8-Fast)

Symmetric, per-output-channel, signed int8 — and, if ConvRot is on, rotate first:

```text
w = W.float()
if convrot and in_features % group_size == 0:        # group_size = 256
    w = (w.view(out, in//gs, gs) @ H.T).reshape(out, in)   # H = normalized regular Hadamard
scale = clamp(w.abs().amax(dim=1, keepdim=True) / 127, min=1e-30)   # fp32 [out, 1]
q     = round(w / scale).clamp(-128, 127).to(int8)
```

The regular Hadamard `H` is the Kronecker power of `H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`,
normalized by `1/sqrt(size)` (no all-ones column → no row-outlier amplification).

## Layer side tensors

For each quantized Linear `<layer>`:

```text
<layer>.weight        int8 tensor, shape [out_features, in_features]
<layer>.weight_scale  float32 tensor, shape [out_features, 1]   (2D, per-row)
<layer>.comfy_quant   uint8 JSON marker
```

There is **no `<layer>.input_scale`** (activations are runtime-dynamic) and bias is
copied through unquantized. The marker JSON is, in this exact insertion order:

```json
{"convrot": true, "convrot_groupsize": 256, "per_row": true}
```

When a layer's `in_features` is not divisible by the group size, ConvRot is skipped
for that layer and the marker is `{"convrot": false, "per_row": true}`.

## Loader handshake

The downstream `Int8TensorwiseOps._load_from_state_dict` reads `<layer>.weight`
(int8), `<layer>.weight_scale` (2D `[out,1]` → per-row kernel), and the
`comfy_quant` marker (`convrot`/`convrot_groupsize` re-derive the same Hadamard for
the online activation rotation). Activations are quantized dynamically at runtime.

## Scope note

The format is reusable across model families; layer selection lives in
`model_adapters` / the config `quant.modules` (see the workflow page). Bit-for-bit
parity with ComfyUI-INT8-Fast is enforced by
`tests/unit/test_external_int8_fast_convrot_parity.py`.
