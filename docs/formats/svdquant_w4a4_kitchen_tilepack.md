# SVDQuant W4A4 Kitchen Tile-Pack Format

This document defines the Comfy Quants artifact contract for SVDQuant W4A4 tensors stored with the kitchen tile-packed layout.

The base library may create this checkpoint payload, but it must not import, embed, or launch ComfyUI or comfy-kitchen. Runtime support is expected from the target ComfyUI/comfy-kitchen installation.

## Format identifiers

| Field | Value |
| --- | --- |
| Quant format | `svdquant_w4a4` |
| Storage layout | `kitchen_tile_packed_w4a4` |
| Weight storage dtype | `int8` bytes containing two signed INT4 values |
| Weight value range | `[-8, 7]` |
| Group size | `64` input features |
| N tile size | `128` output features |
| Interleave | `4` |

Checkpoint metadata is stored in the layer-local `comfy_quant` uint8 JSON tensor:

```json
{
  "format": "svdquant_w4a4",
  "layout": "kitchen_tile_packed_w4a4",
  "lowrank_branch_input_basis": "post_smoothing",
  "proj_down_smooth_folded": false
}
```

Layers that use unsigned activation quantization add:

```json
{
  "act_unsigned": true
}
```

`lowrank_branch_input_basis` and `proj_down_smooth_folded` are part of the
artifact semantics, not a proof of fused-runtime parity.  Default direct
exports store `proj_down` in the **post-smoothing** basis:

```text
branch = (x / smooth_factor) @ proj_down @ proj_up.T
```

If a target runtime computes the low-rank down projection from raw activations,
the exporter must explicitly fold smoothing into the stored down projection:

```text
proj_down_runtime = proj_down_post_smoothing / smooth_factor[:, None]
lowrank_branch_input_basis = "raw"
proj_down_smooth_folded = true
```

The direct `quantize-int4` CLI exposes this as the explicit experimental
`--lowrank-branch-input-basis raw` option.  It must not be applied silently, and
selecting it is not by itself a fused-runtime parity claim.

## Signed INT4 byte packing

Two signed INT4 values are stored in one `int8` byte:

```text
low nibble  = first value  & 0x0F
high nibble = second value & 0x0F
byte        = low | (high << 4)
```

When unpacking, nibbles `8..15` map back to signed values by subtracting `16`.

## Natural SVDQuant parameter family

A natural-layout SVDQuant linear layer uses:

| Tensor | Shape | Dtype | Required | Meaning |
| --- | --- | --- | --- | --- |
| `weight` | `(N, K/2)` | `int8` | yes | Signed INT4 pairs for an `(N, K)` logical matrix. |
| `weight_scale` | `(K/64, N)` | fp16/bf16 | yes | Per-group weight scale. |
| `smooth_factor` | `(K,)` | fp16/bf16 | yes | Activation smoothing factor. |
| `proj_down` | `(K, R)` | fp16/bf16 | yes | Low-rank down projection. |
| `proj_up` | `(N, R)` | fp16/bf16 | yes | Low-rank up projection. |
| `bias` | `(N,)` | fp16/bf16 | no | Linear bias. |
| `comfy_quant` | `(json_bytes,)` | uint8 | no | Layer quantization metadata. |

Where:

```text
N = out_features
K = in_features
R = low-rank rank, for example 32, 64, 96, or 128
```

## Kitchen tile-packed parameter family

After applying the kitchen tile-pack transform:

| Tensor | Natural shape | Tile-packed shape | Notes |
| --- | --- | --- | --- |
| `weight` | `(N, K/2)` | `(N/128, K/64, 32, 128)` | Requires `N % 128 == 0` and `K % 64 == 0`. |
| `weight_scale` | `(K/64, N)` | `(N/128, K/64, 128)` | Packs the N axis. |
| `smooth_factor` | `(K,)` | unchanged | Stored in natural layout. |
| `proj_down` | `(K, R)` | unchanged | Stored in natural layout. |
| `proj_up` | `(N, R)` | `(N/128, R, 128)` | Packs the N axis. |
| `bias` | `(N,)` | unchanged | Optional. |
| `comfy_quant` | uint8 JSON | uint8 JSON | Must include `format` and `layout`. |

The fixed `weight` tile tail is derived from:

```text
KITCHEN_BLOCK_N / KITCHEN_INTERLEAVE = 128 / 4 = 32
KITCHEN_INTERLEAVE * KITCHEN_GROUP_SIZE / 2 = 4 * 64 / 2 = 128
```

Therefore:

```text
weight tile tail = (32, 128)
```

## Model adapter boundary

This format module does not decide which model layers are quantized.

For Qwen-Image-Edit INT4 bundles, the expected model-adapter policy is:

```text
attention and MLP linear layers -> SVDQuant W4A4 kitchen tile-pack
modulation linear layers        -> AWQ W4A16 when dense tensors are present and the target runtime supports it
remaining tensors               -> high precision copy
```

QKV split and Qwen modulation reordering are model-structure operations and belong in the Qwen model adapter/export bridge, not in the reusable format codec.

## Checkpoint writers and import bridge

The base library includes a layout writer for already-quantized SVDQuant W4A4
state dicts:

```text
backends/int4_kitchen_export.py
```

It performs only these operations:

1. read a local safetensors checkpoint or indexed shard set;
2. find layer prefixes whose `.comfy_quant` JSON declares
   `{"format":"svdquant_w4a4"}`;
3. validate that each detected layer has the required SVDQuant tensor family;
4. tile-pack `weight`, `weight_scale`, and `proj_up`;
5. patch `.comfy_quant` with `layout="kitchen_tile_packed_w4a4"`;
6. copy all other tensors into a single output safetensors checkpoint, including
   non-SVDQuant tensor families such as AWQ W4A16 modulation layers;
7. write an export report.

CLI:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main export-int4 \
  --format svdquant_w4a4 \
  --source-format natural-safetensors \
  --source /path/to/natural_svdquant_w4a4.safetensors \
  --out /path/to/export-dir \
  --device cuda:0 \
  --hash-output \
  --json
```

This writer is not an SVDQuant solver and does not read DeepCompressor,
Nunchaku, ComfyUI, or comfy-kitchen Python modules.

The base library also includes a dependency-free import bridge for Qwen-Image-
Edit PTQ artifacts produced by DeepCompressor-style search jobs:

```text
backends/deepcompressor_import.py
```

The bridge expects a local artifact directory:

```text
model.pt   source tensors
scale.pt   <layer>.weight.scale.0 and optional <layer>.weight.scale.1
smooth.pt  optional activation smoothing factors
branch.pt  low-rank branch tensors required for imported SVDQuant layers
```

The bridge performs:

1. load the local `.pt` state dictionaries with `torch.load`;
2. map Qwen-Image-Edit source linear names through
   `model_adapters/qwen_image_edit_int4.py`;
3. convert DeepCompressor scales to natural `(K/64, N)` `weight_scale`;
4. if `.weight.scale.1` is present and representable at group size 64, combine
   it with `.weight.scale.0` into one effective `weight_scale`;
5. round `weight / weight_scale` and require the signed INT4 range `[-8, 7]`;
6. pack the signed INT4 values into natural `(N, K/2)` bytes;
7. attach `smooth_factor`, `proj_down`, `proj_up`, optional `bias`, and
   `comfy_quant`;
8. hand the natural state dict to `backends/int4_kitchen_export.py`.

DeepCompressor import CLI:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main export-int4 \
  --format svdquant_w4a4 \
  --source-format deepcompressor-qwen-image-edit \
  --source /path/to/deepcompressor-ptq-artifacts \
  --out /path/to/export-dir \
  --device cuda:0 \
  --hash-output \
  --json
```

This bridge is model-adapter work, not reusable format logic. It currently
supports the Qwen-Image-Edit SVDQuant attention/MLP linears. Importing external
engine-specific AWQ W4A16 modulation artifacts remains separate bridge work; the
direct dense-checkpoint writer has its own kitchen-native AWQ modulation
quantizer.

## Direct full-pipeline target

The final INT4 route in this repository is not the DeepCompressor import bridge.
The target route is:

```text
BF16/FP16 Qwen-Image-Edit checkpoint + calibration/edit prompts
  -> Comfy Quants mixed INT4 pipeline
  -> SVDQuant W4A4 attention/MLP tensors in kitchen tile-packed layout
  -> AWQ W4A16 modulation tensors when present
  -> one safetensors checkpoint
```

The direct CLI is:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --out runs/qwen-edit-2511/int4-svdquant-w4a4 \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

`export-int4` remains a repack/import utility for already-quantized artifacts.
It is useful for validating the storage contract, but it is not the final
end-to-end quantization path.
