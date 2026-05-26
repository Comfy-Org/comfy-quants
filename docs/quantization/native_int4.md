# Native INT4 solver

`quantize-int4` runs the package-native Qwen-Image-Edit INT4 solver from a dense
transformer checkpoint and writes a `svdquant_w4a4` tile-pack artifact.

Use this entrypoint for algorithm development or advanced workflows. For the
Qwen-Image-Edit-2511 DeepCompressor/Nunchaku flow, use
[`qwen_image_edit_2511_int4.md`](qwen_image_edit_2511_int4.md).

## Inputs

| Input | Argument | Description |
| --- | --- | --- |
| Dense transformer checkpoint | `--source` | Input safetensors file, index JSON, or local shard directory. |
| Model family | `--family` | `qwen_image_edit`. |
| Output format | `--format` | `svdquant_w4a4`. |
| Output path | `--out` | Output file or directory. |
| Rank | `--rank` | Low-rank branch rank. |
| Device | `--device` | `auto`, `cuda:0`, or another torch device. |
| Activation stats | `--activation-stats` | Required for calibrated modes. |
| GPTQ Hessian stats | `--gptq-hessian-stats` | Required for GPTQ mode. |

## Modes

| Mode | Required extra inputs | Description |
| --- | --- | --- |
| `weight_only_initialization` | none | Initializes SVDQuant tensors from weights without calibration. |
| `calibrated_svdquant` | `--activation-stats` | Applies activation-stat smoothing and residual-SVD branch initialization. |
| `svdquant_gptq_experimental` | `--activation-stats`, `--gptq-hessian-stats` | Adds the package-native GPTQ solve. |

## Basic export

```bash
comfy-quants quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --out runs/qwen-edit-2511/int4-svdquant-w4a4 \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

Directory output:

```text
diffusion_pytorch_model.svdquant_w4a4.safetensors
quantization_report.json
```

## Calibrated export

```bash
comfy-quants quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --activation-stats /absolute/path/to/int4_activation_stats.json \
  --quantization-mode calibrated_svdquant \
  --out runs/qwen-edit-2511/int4-svdquant-w4a4-calibrated \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

## GPTQ export

```bash
comfy-quants quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --activation-stats /absolute/path/to/int4_activation_stats.json \
  --gptq-hessian-stats /absolute/path/to/int4_gptq_hessian_stats.json \
  --quantization-mode svdquant_gptq_experimental \
  --out runs/qwen-edit-2511/int4-svdquant-w4a4-gptq \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

## Calibration reducer commands

`comfy_quants` can build capture manifests and reduce captured activation tensors.
The actual model forward pass is supplied by the user's runtime environment.

```bash
comfy-quants calib plan-int4-capture --help
comfy-quants calib materialize-int4-capture --help
comfy-quants calib reduce-int4-activations --help
comfy-quants calib reduce-int4-gptq-hessians --help
```

## Format reference

- [`../formats/svdquant_w4a4_kitchen_tilepack.md`](../formats/svdquant_w4a4_kitchen_tilepack.md)
- [`../formats/awq_w4a16.md`](../formats/awq_w4a16.md)
