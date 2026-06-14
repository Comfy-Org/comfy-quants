# CLI reference

The public command is `comfy-quants`.

```bash
comfy-quants --help
comfy-quants <command> --help
```

Use this page for command syntax. For complete workflows, start with
[`quantization/`](quantization/). For tensor/storage details, see
[`formats/`](formats/).

## Inspect source weights

```bash
comfy-quants inspect \
  --model /path/to/model \
  --family <model_family> \
  --out runs/inspect \
  --json
```

## FP8 commands

Plan selected tensors without writing checkpoint bytes:

```bash
comfy-quants quantize \
  --config /path/to/fp8_config.yaml \
  --work-dir runs/fp8-plan \
  --dry-run \
  --json
```

Export a full ComfyUI-loadable FP8 checkpoint:

```bash
comfy-quants export-model \
  --config /path/to/fp8_config.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out runs/export-fp8 \
  --device cuda:0 \
  --hash-output \
  --json
```

Guide: [`quantization/fp8.md`](quantization/fp8.md).

## INT8 W8A8 commands

Export a full INT8 W8A8 (+ optional ConvRot) checkpoint for the ComfyUI-INT8-Fast
node — int8 weights + per-output-channel scales offline; dynamic int8 activations
and the int8 matmul run downstream:

```bash
comfy-quants export-model-w8a8 \
  --config /path/to/int8_w8a8_config.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out runs/export-int8-w8a8 \
  --device cuda:0 \
  --convrot \
  --hash-output \
  --json
```

Requires `quant.target_dtype: int8_w8a8` in the config. Use `--no-convrot` for plain
row-wise W8A8. Guide: [`quantization/int8_w8a8.md`](quantization/int8_w8a8.md).

## MXFP8 commands

Export a full MXFP8 (OCP microscaling FP8) checkpoint for **stock ComfyUI's native**
loader — `float8_e4m3fn` weights + per-32-element E8M0 block scales (cuBLAS
`to_blocked` swizzle); the mxfp8 tensor-core matmul runs on Blackwell downstream:

```bash
comfy-quants export-model-mxfp8 \
  --config /path/to/mxfp8_config.yaml \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out runs/export-mxfp8 \
  --device cuda:0 \
  --hash-output \
  --json
```

Requires `quant.target_dtype: mxfp8` in the config. Guide:
[`quantization/mxfp8.md`](quantization/mxfp8.md).

## INT4 commands

Open the INT4 format guide first, then choose one of the model-family flows
listed there. Format-level commands are:

```bash
comfy-quants quantize-int4 --help
comfy-quants inspect-int4 --help
comfy-quants export-int4 --help
```

Model-family one-step commands are listed in
[`quantization/int4.md`](quantization/int4.md).

## INT4 one-step export pattern

Model-family one-step commands follow this pattern:

```bash
comfy-quants <model-family-int4-command> \
  --model /path/to/model \
  --base-checkpoint /path/to/base_transformer.safetensors \
  --out /path/to/model_int4_tilepack.safetensors \
  --calibration-samples 128 \
  --search-strength quality-r64 \
  --gpus 0 \
  --hash-output \
  --json
```

Some commands require additional tool paths or model-family-specific options. Use
`comfy-quants <command> --help` and the linked model-family guide for the exact
arguments.

## Built-in INT4 solver pattern

```bash
comfy-quants quantize-int4 \
  --family <model_family> \
  --format svdquant_w4a4 \
  --source /path/to/diffusion_pytorch_model.safetensors \
  --out runs/int4-svdquant-w4a4 \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

Main modes:

```text
weight_only_initialization
calibrated_svdquant
svdquant_gptq_experimental
```

Guide: [`quantization/native_int4.md`](quantization/native_int4.md).

## Calibration helpers

These commands create capture plans and reduce captured activation tensors. They
are advanced helpers for solver development.

```bash
comfy-quants calib plan-int4-capture --help
comfy-quants calib materialize-int4-capture --help
comfy-quants calib reduce-int4-activations --help
comfy-quants calib reduce-int4-gptq-hessians --help
```

## INT4 artifact inspection

```bash
comfy-quants inspect-int4 \
  --artifact /path/to/model_int4_tilepack.safetensors \
  --family <model_family> \
  --format svdquant_w4a4 \
  --json
```

Guide: [`quantization/int4_tools.md`](quantization/int4_tools.md).

## INT4 repack/export

```bash
comfy-quants export-int4 \
  --format svdquant_w4a4 \
  --source-format <source_format> \
  --source /path/to/int4-source-artifacts \
  --out runs/export-int4 \
  --device cuda:0 \
  --hash-output \
  --json
```

## Generic artifact commands

```bash
comfy-quants validate --help
comfy-quants export --help
comfy-quants jobs --help
comfy-quants resume --help
```
