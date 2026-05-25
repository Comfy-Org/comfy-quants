# Comfy Quants CLI

The supported base-library interface is CLI + Python SDK. The CLI is a standalone offline quantization entrypoint. It targets artifacts that ComfyUI can use, but it must not run, import, configure, or embed ComfyUI.

Supported FP8 targets in this repository:

| Comfy Quants `target_dtype` | ComfyUI checkpoint format | Torch dtype | safetensors dtype |
| --- | --- | --- | --- |
| `fp8_e4m3` | `float8_e4m3fn` | `torch.float8_e4m3fn` | `F8_E4M3` |
| `fp8_e5m2` | `float8_e5m2` | `torch.float8_e5m2` | `F8_E5M2` |

`fp8_e5m2` is the ComfyUI `float8_e5m2` path. It is not `mxfp8`.

## Doctor

```bash
PYTHONPATH=src python -m comfy_quants.cli.main doctor
PYTHONPATH=src python -m comfy_quants.cli.main doctor --json
```

`doctor --json` reports:

- registered adapters / formats / algorithms / backends;
- detected Python/package/GPU environment;
- static artifact contract index for ComfyUI-compatible exports.

Expected contract fields look like:

```json
{
  "artifact_contracts": {
    "schema_version": "artifact_contract_index.v1",
    "artifact_target": "comfyui",
    "contract_source": "comfy_quants",
    "contract_mode": "static_adapter_contract"
  },
  "available_formats": ["awq_w4a16", "fp8_e4m3", "fp8_e5m2", "svdquant_w4a4"]
}
```

## Inspect Qwen-Image

```bash
PYTHONPATH=src python -m comfy_quants.cli.main inspect \
  --model Qwen/Qwen-Image-2512 \
  --family qwen_image \
  --revision <pin-this-commit> \
  --out runs/qwen-image-2512/inspect \
  --json
```

## Inspect Qwen-Image-Edit

```bash
PYTHONPATH=src python -m comfy_quants.cli.main inspect \
  --model Qwen/Qwen-Image-Edit-2511 \
  --family qwen_image_edit \
  --revision <pin-this-commit> \
  --out runs/qwen-edit-2511/inspect \
  --json
```

## Build calibration set manifest

Text-to-image:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib build \
  --family qwen_image \
  --prompt-set examples/prompts_t2i.jsonl \
  --out runs/qwen-image-2512/calib-v0
```

Image edit:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib build \
  --family qwen_image_edit \
  --edit-set examples/edit_cases.jsonl \
  --image-root data/calib_images \
  --edit-types text_edit,appearance_preserve \
  --out runs/qwen-edit-2511/calib-v0
```

Normalize the manifest into explicit records that a future runtime capture
backend can consume:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib records \
  --manifest runs/qwen-edit-2511/calib-v0/calibration_manifest.json \
  --out runs/qwen-edit-2511/calib-v0/records \
  --json
```

Plan the calibrated INT4 activation targets from the dense checkpoint and the
normalized records:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib plan-int4-capture \
  --family qwen_image_edit \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --records runs/qwen-edit-2511/calib-v0/records/calibration_records.jsonl \
  --out runs/qwen-edit-2511/calib-v0/capture-plan \
  --json
```

This writes:

```text
runs/qwen-edit-2511/calib-v0/capture-plan/
├── capture_plan.json
├── activation_samples.template.jsonl
└── capture_report.json
```

`capture_plan.json` contains the selected Qwen-Image-Edit INT4 linear inputs,
their source weight shapes, and accepted activation-stats key aliases. This is a
plan-only command: it does not run model forward passes, does not write
activation tensors, and does not import ComfyUI.

Materialize a reducer-ready sample manifest from the plan and records:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib materialize-int4-capture \
  --plan runs/qwen-edit-2511/calib-v0/capture-plan/capture_plan.json \
  --out runs/qwen-edit-2511/calib-v0/capture-run \
  --json
```

This writes:

```text
runs/qwen-edit-2511/calib-v0/capture-run/
├── activation_samples.jsonl
├── activation_tensors/
│   └── <case-id>.safetensors   # written later by external capture code
└── capture_materialization_report.json
```

`materialize-int4-capture` writes only the JSONL references and report. It does
not run forward passes and does not create the safetensors activation files.
The manifest rows point at `activation_tensors/<case-id>.safetensors` and use
the planned tensor name for each selected linear input.

The base library also provides the runtime-independent reduction stage for
captured activation tensors. External capture code can consume
`capture_plan.json`, run its own model runtime, and dump safetensors files that
match the materialized manifest. A typical row is:

```json
{"case_id":"case-1","layer":"transformer_blocks.0.attn.to_q","file":"activation_tensors/case-1.safetensors","tensor":"transformer_blocks.0.attn.to_q.input","channel_dim":-1}
```

For Python capture implementations, the helper
`write_int4_activation_case_safetensors(...)` validates tensor names and channel
counts from `capture_plan.json` before writing one case's safetensors file.
This helper is still runtime-independent; it does not import a model runtime or
ComfyUI.

Then reduce those dumps into the `int4_activation_stats.json` consumed by
`quantize-int4 --quantization-mode calibrated_svdquant` for smoothing:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib reduce-int4-activations \
  --samples runs/qwen-edit-2511/calib-v0/capture-run/activation_samples.jsonl \
  --input-root runs/qwen-edit-2511/calib-v0/capture-run \
  --out runs/qwen-edit-2511/calib-v0/int4-stats \
  --channel-dim -1 \
  --json
```

Reduce the same activation dumps into the per-layer Hessian manifest needed by
the GPTQ solve:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib reduce-int4-gptq-hessians \
  --samples runs/qwen-edit-2511/calib-v0/capture-run/activation_samples.jsonl \
  --input-root runs/qwen-edit-2511/calib-v0/capture-run \
  --out runs/qwen-edit-2511/calib-v0/int4-gptq-hessians \
  --channel-dim -1 \
  --hessian-block-size 512 \
  --device cuda:0 \
  --json
```

This writes `int4_gptq_hessian_stats.json` and a `gptq_hessians/` directory.
The Hessian reducer reads only local safetensors activation dumps; it does not
import a model runtime or ComfyUI. The `svdquant_gptq_experimental` checkpoint
mode consumes this manifest after activation smoothing, transforms the raw-input
Hessian into the post-smoothing basis, and feeds the repo-native GPTQ layer core.
`calibrated_svdquant` intentionally remains the RTN milestone mode.

## Plan FP8 quantization

E4M3 dry run:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize \
  --config configs/qwen_image_2512_fp8_static.yaml \
  --work-dir runs/qwen-image-2512/fp8-e4m3-static-v0 \
  --dry-run

PYTHONPATH=src python -m comfy_quants.cli.main quantize \
  --config configs/qwen_image_edit_2511_fp8_static.yaml \
  --work-dir runs/qwen-edit-2511/fp8-e4m3-static-v0 \
  --dry-run
```

E5M2 dry run:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize \
  --config configs/qwen_image_2512_fp8_e5m2_static.yaml \
  --work-dir runs/qwen-image-2512/fp8-e5m2-static-v0 \
  --dry-run

PYTHONPATH=src python -m comfy_quants.cli.main quantize \
  --config configs/qwen_image_edit_2511_fp8_e5m2_static.yaml \
  --work-dir runs/qwen-edit-2511/fp8-e5m2-static-v0 \
  --dry-run
```

Dry-run planning writes:

```text
<work-dir>/
├── plan.json
├── model_graph.json
├── model_inspection.json
└── artifact/
    ├── manifest.json
    └── quant_tensor_index.json
```

`quant_tensor_index.json` lists selected FP8 weight tensors, storage dtype, source dtype, rounding, scale declaration, scale axis, and module metadata. Qwen-Image and Qwen-Image-Edit currently select 839 tensors with the default policy for both `fp8_e4m3` and `fp8_e5m2`.

Dry-run jobs do not write tensor payload bytes.

## Write selected FP8 payload bytes

The non-dry-run `quantize` path writes selected tensor payload files from a local safetensors source. Accepted source shapes are:

- a single `.safetensors` file;
- a `.safetensors.index.json` file with local shard references;
- a directory containing `diffusion_pytorch_model.safetensors.index.json` and its referenced shard files.

The CLI does not download model weights and it does not call or import ComfyUI.

Config shape:

```yaml
project:
  name: qwen-local-fp8-e5m2
model:
  family: qwen_image
  model_id: /absolute/path/to/qwen-transformer
  source: local
  dtype: bf16
quant:
  algorithm: fp8_static
  target_dtype: fp8_e5m2   # or fp8_e4m3
  scale:
    granularity: per_channel
    axis: out_features
    method: amax
  rounding: nearest_even
  modules:
    include:
      - transformer_blocks.0.attn.to_q
    exclude: []
artifact:
  compatibility_target: L2
```

Command:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize \
  --config local-qwen-fp8.yaml \
  --work-dir runs/qwen-local/fp8-static-v0 \
  --device cuda:0 \
  --json
```

The writer emits:

```text
<work-dir>/artifact/
├── manifest.json
├── quant_tensor_index.json
├── payload_report.json
├── tensors/
│   └── fp8_weights.safetensors
└── scales/
    └── fp8_static_scales.safetensors
```

`payload_report.json` records the source checkpoint, source layout, selected source shard counts, tensor count, output file paths, byte sizes, and hashes. The manifest records `artifact_state: payload_written` and `tensor_payload_state: written`.

If `--dry-run` is omitted while `model.source` is not `local`, the CLI returns a configuration error. If a selected tensor key is missing from the checkpoint, or the checkpoint tensor shape does not match the static contract, the job fails without writing a silent partial artifact.

## Export a single ComfyUI-loadable diffusion checkpoint

`export-model` writes a single safetensors file intended for ComfyUI's diffusion model loader/save path. It does not invoke ComfyUI.

E4M3:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main export-model \
  --config configs/qwen_image_edit_2511_fp8_static.yaml \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --out runs/qwen-edit-2511/export-e4m3 \
  --device cuda:0 \
  --hash-output \
  --json
```

E5M2:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main export-model \
  --config configs/qwen_image_edit_2511_fp8_e5m2_static.yaml \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --out runs/qwen-edit-2511/export-e5m2 \
  --device cuda:0 \
  --hash-output \
  --json
```

When `--out` is a directory, the exporter writes:

```text
<out>/diffusion_pytorch_model.fp8_e4m3.safetensors
# or
<out>/diffusion_pytorch_model.fp8_e5m2.safetensors
<out>/export_report.json
```

Each selected layer is exported as:

```text
<layer>.weight        -> torch.float8_e4m3fn or torch.float8_e5m2
<layer>.weight_scale  -> torch.float32 scalar
<layer>.input_scale   -> torch.float32 scalar 1.0
<layer>.comfy_quant   -> uint8 JSON bytes
```

The `.comfy_quant` JSON payload is one of:

```json
{"format":"float8_e4m3fn","full_precision_matrix_mult":true}
{"format":"float8_e5m2","full_precision_matrix_mult":true}
```

Qwen-Image-Edit-2511 export also keeps the `__index_timestep_zero__` compatibility marker.

## Quantize directly to an INT4 kitchen tile-packed checkpoint

`quantize-int4` is the direct INT4 route. It starts from a dense local
Qwen-Image-Edit safetensors checkpoint and writes one ComfyUI-targeted
`svdquant_w4a4` kitchen tile-packed safetensors file. For full Qwen checkpoints
this file is a mixed INT4 bundle: attention/MLP linears use SVDQuant W4A4, while
`transformer_blocks.*.img_mod.1` and `transformer_blocks.*.txt_mod.1` modulation
linears are emitted as AWQ W4A16 when those dense tensors are present. The base
library does not import ComfyUI, DeepCompressor, Nunchaku, or comfy-kitchen.

Current bootstrap command:

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

When `--out` is a directory, it writes:

```text
<out>/diffusion_pytorch_model.svdquant_w4a4.safetensors
<out>/quantization_report.json
```

The public quantization modes are:

```text
weight_only_initialization   dense weights -> group-64 signed INT4, identity smoothing, zero branch
calibrated_svdquant          activation stats -> smoothing -> residual-SVD branch -> group-64 signed INT4 RTN
svdquant_gptq_experimental   activation stats + GPTQ Hessian manifest -> smoothing -> Hessian basis transform -> branch subtraction -> grouped signed-INT4 GPTQ
```

`weight_only_initialization` is a calibration-free tensor bootstrap. It selects
Qwen-Image-Edit attention/MLP linears from the committed static adapter contract,
quantizes dense weights to signed INT4 with group size 64, emits identity
smoothing and zero low-rank branch tensors, then invokes the kitchen tile-pack
writer. It is not a quality target.

`calibrated_svdquant` consumes a per-layer activation-stats JSON file, solves
non-identity smoothing, emits a low-rank residual SVD branch, and still uses
round-to-nearest for the SVDQuant W4 weights. It also emits AWQ W4A16 modulation
tensors when `img_mod.1` / `txt_mod.1` weights are present. This is an explicit
RTN milestone, not the final GPTQ solver path.

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --activation-stats /absolute/path/to/int4_activation_stats.json \
  --quantization-mode calibrated_svdquant \
  --out runs/qwen-edit-2511/int4-calibrated \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

`svdquant_gptq_experimental` is the first repo-native path where smoothing is
followed by GPTQ. It consumes the activation-stats file plus a GPTQ Hessian
manifest produced from the same captured layer-input dumps, treats the Hessian
as raw-input-basis, transforms it with the solved smoothing divisor, subtracts
the initialized low-rank branch, then calls the repo-native grouped signed-INT4
GPTQ core for SVDQuant attention/MLP layers. The default branch calibration is
`weight_residual`. Passing `--lowrank-calibration output_error` also requires
`--activation-samples`; that path fits the branch from captured activation/output
error before the same GPTQ solve. Both modes still report non-publishable
runtime state because external mixed-runtime full inference is not validated.

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --activation-stats /absolute/path/to/int4_activation_stats.json \
  --gptq-hessian-stats /absolute/path/to/int4_gptq_hessian_stats.json \
  --quantization-mode svdquant_gptq_experimental \
  --out runs/qwen-edit-2511/int4-gptq-experimental \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

Output-error low-rank calibration adds the captured activation sample manifest:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --activation-stats /absolute/path/to/int4_activation_stats.json \
  --gptq-hessian-stats /absolute/path/to/int4_gptq_hessian_stats.json \
  --activation-samples /absolute/path/to/activation_samples.jsonl \
  --quantization-mode svdquant_gptq_experimental \
  --lowrank-calibration output_error \
  --out runs/qwen-edit-2511/int4-gptq-output-error \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

Low-rank branch basis is explicit in the artifact metadata. The default is
`--lowrank-branch-input-basis raw`, which stores `proj_down` for the target
Kitchen/Nunchaku contract:

```text
branch = x @ proj_down @ proj_up.T
```

For internal/reference experiments that need the post-smoothing dense-math
basis, use `--lowrank-branch-input-basis post_smoothing`. For the normal target
runtime path, keep the default raw basis:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --activation-stats /absolute/path/to/int4_activation_stats.json \
  --gptq-hessian-stats /absolute/path/to/int4_gptq_hessian_stats.json \
  --quantization-mode svdquant_gptq_experimental \
  --lowrank-branch-input-basis raw \
  --out runs/qwen-edit-2511/int4-gptq-raw-branch \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

That option writes `proj_down_post_smoothing / smooth_factor[:, None]` and
reports `lowrank_branch_input_basis: raw` plus `proj_down_smooth_folded: true`.
It is not a proof that an external fused runtime has accepted the file; the
`publishable_svdquant_gptq` gate remains false until full mixed-runtime load and
image inference are validated.

### Make an INT4 runtime fixture

Use `make-int4-runtime-fixture` to create a deterministic single-layer golden
case before comparing this repository's artifact contract against an external
fused runtime:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main make-int4-runtime-fixture \
  --out runs/int4-runtime-fixtures/svdquant-w4a4-raw \
  --activation-signedness signed \
  --lowrank-branch-input-basis raw \
  --seed 1234 \
  --json
```

The command writes:

```text
<out>/svdquant_w4a4_runtime_fixture.safetensors
<out>/runtime_fixture_report.json
```

The safetensors file contains one kitchen tile-packed layer under
`fixture_layer.*` and oracle tensors under `fixture.*`, including
`fixture.input`, activation W4 codes/scales/dequantized values, and
`fixture.expected_output`. The report records
`runtime_reference_state: repo_runtime_like_activation_w4_branch_oracle_runtime_unverified`,
`external_runtime_validation: not_run`, and `publishable_svdquant_gptq: false`.
It also includes `external_harness_contract`, which names the raw forward input
tensor, expected output tensor, required layer tensors, the external output
tensor name `runtime.output`, and the validation command to run after an
external harness saves that output.
Passing this local self-check is only a layer-level precondition for external
runtime parity; it is not full ComfyUI load or image inference validation.

Use `make-awq-runtime-fixture` for the companion AWQ W4A16 modulation-layer
contract:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main make-awq-runtime-fixture \
  --out runs/int4-runtime-fixtures/awq-w4a16 \
  --seed 2234 \
  --json
```

The command writes:

```text
<out>/awq_w4a16_runtime_fixture.safetensors
<out>/runtime_fixture_report.json
```

The safetensors file contains one kitchen-native AWQ layer under
`fixture_layer.*` plus `fixture.input`, `fixture.expected_output`,
`fixture.dequantized_weight`, and `fixture.quantized_weight_uint4`.  The report
records `runtime_reference_state: kitchen_awq_w4a16_reference_math_runtime_unverified`,
`external_runtime_validation: not_run`, and `publishable_svdquant_gptq: false`.
Its `external_harness_contract` uses the same `runtime.output` handoff name, so
SVDQuant and AWQ fixtures can share `validate-runtime-fixture-output`.
This fixture validates only the repository's AWQ checkpoint formula and creates
a deterministic handoff case for an external runtime harness; it does not prove
that Qwen mixed SVDQuant+AWQ inference is loadable.

### Validate an external runtime fixture output

Use `validate-runtime-fixture-output` after an external runtime harness has
loaded one fixture and saved its output tensor to a safetensors file. The command
compares that external tensor against the fixture oracle and writes a JSON
report:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main validate-runtime-fixture-output \
  --fixture runs/int4-runtime-fixtures/svdquant-w4a4-raw/svdquant_w4a4_runtime_fixture.safetensors \
  --output runs/external-runtime-output/svdquant_w4a4_output.safetensors \
  --expected-tensor fixture.expected_output \
  --actual-tensor runtime.output \
  --atol 1e-4 \
  --rtol 1e-4 \
  --out runs/runtime-validation/svdquant-w4a4-raw \
  --json
```

The external output file is expected to contain `runtime.output` by default.
The report is written to:

```text
<out>/runtime_fixture_output_validation_report.json
```

Exit code `0` means the single tensor comparison passed. Exit code `2` means
the comparison failed, a tensor was missing, shapes differed, or the inputs were
invalid. A passing report records:

```text
validation_scope: single_layer_runtime_fixture_output_only
external_runtime_validation: single_layer_fixture_output_passed
publishable_svdquant_gptq: false
```

This is deliberately only a single-layer parity gate. It does not validate full
Qwen-Image/Edit model load, mixed SVDQuant W4A4 plus AWQ W4A16 dispatch,
ComfyUI node/runtime registration, full PNG inference, or publishable
SVDQuant+GPTQ checkpoint status.

### Aggregate INT4 runtime readiness gates

Use `validate-int4-runtime-readiness` to collect the individual runtime parity
reports into one checklist. This prevents a passed single-layer SVDQuant/AWQ
comparison from being mistaken for a publishable mixed Qwen checkpoint:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main validate-int4-runtime-readiness \
  --svdquant-report runs/runtime-validation/svdquant-w4a4-raw/runtime_fixture_output_validation_report.json \
  --awq-report runs/runtime-validation/awq-w4a16/runtime_fixture_output_validation_report.json \
  --out runs/runtime-readiness/qwen-edit-2511-int4 \
  --json
```

The command writes:

```text
<out>/int4_runtime_readiness_report.json
```

With only the two single-layer fixture reports, the readiness status remains:

```text
status: blocked
publishable_svdquant_gptq: false
missing_gates:
  - mixed_svdquant_w4a4_awq_w4a16_dispatch
  - full_qwen_image_edit_png_inference
```

The optional future reports are:

```bash
--mixed-dispatch-report /path/to/mixed_dispatch_report.json
--full-inference-report /path/to/full_inference_report.json
```

Those external reports must explicitly carry their validation scope:

```text
mixed_svdquant_w4a4_awq_w4a16_dispatch
full_qwen_image_edit_png_inference
```

Even when every gate is reported as passed, this readiness report keeps
`publishable_svdquant_gptq: false` and sets
`publishable_candidate_after_manual_review: true`; promotion to a publishable
artifact remains an explicit manual release decision.

Minimal activation-stats JSON shape:

```json
{
  "schema_version": "int4_activation_stats.v1",
  "layers": {
    "transformer_blocks.0.attn.to_q": {
      "input_amax": [1.0, 1.2],
      "input_rms": [0.25, 0.3],
      "sample_count": 8,
      "element_count": 4096
    }
  }
}
```

`input_amax` must contain one value per linear input channel. `input_rms` is
optional and is preserved by the stats helpers for later calibration-quality
work; the current calibrated writer uses `input_amax` for smoothing.

The companion GPTQ Hessian manifest produced by
`calib reduce-int4-gptq-hessians` has this shape:

```json
{
  "schema_version": "int4_gptq_hessian_stats.v1",
  "normalization": "two_over_row_count",
  "layers": {
    "transformer_blocks.0.attn.to_q": {
      "file_path": "gptq_hessians/transformer_blocks.0.attn.to_q.safetensors",
      "tensor_name": "hessian",
      "channel_count": 3072,
      "sample_count": 8,
      "row_count": 65536,
      "normalization_count": 65536
    }
  }
}
```

Those Hessians are consumed only by `svdquant_gptq_experimental`. They are not
used by `calibrated_svdquant`, whose report intentionally remains the RTN state
with `gptq_state: not_implemented`.

For a full Qwen checkpoint with modulation tensors, the GPTQ experimental mixed
artifact reports:

```text
algorithm_state: experimental_svdquant_gptq_awq_runtime_unverified
publishable_svdquant_gptq: false
gptq_state: layer_core_integrated
runtime_contract_state: static_artifact_contract_only
mixed_quantization_state: experimental_svdquant_w4a4_awq_w4a16_runtime_unverified
```

If a minimal test checkpoint has no modulation tensors, the mixed state falls
back to `svdquant_only_awq_modulation_not_implemented`. For
`calibrated_svdquant`, `algorithm_state` remains
`experimental_smooth_rtn_svd_no_gptq` by design.

See `int4_svdquant_runtime_contract.md` for the runtime and validation gates
that must pass before this can be advertised as a full SVDQuant+GPTQ model.

Dry-run planning:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --out runs/qwen-edit-2511/int4-plan \
  --dry-run \
  --json
```

For `--quantization-mode calibrated_svdquant`, dry-run also validates the
activation-stats file against the selected dense checkpoint linears:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --activation-stats /absolute/path/to/int4_activation_stats.json \
  --quantization-mode calibrated_svdquant \
  --out runs/qwen-edit-2511/int4-calibrated-plan \
  --dry-run \
  --json
```

The generated `quantization_report.json` includes
`activation_stats_coverage`. A valid report has `state: valid`; missing stats
keys or `input_amax` vectors with the wrong channel length produce
`dry_run_validation_failed` and a non-zero CLI exit before any checkpoint bytes
are written.

## Export an INT4 kitchen tile-packed checkpoint

`export-int4` writes a single INT4 safetensors checkpoint whose imported
SVDQuant W4A4 layers use the kitchen tile-packed storage contract documented in
`docs/formats/svdquant_w4a4_kitchen_tilepack.md`.

The command has two source modes:

| `--source-format` | `--source` path | Purpose |
| --- | --- | --- |
| `natural-safetensors` | `.safetensors`, safetensors index JSON, or local safetensors shard directory | Repack an already-natural SVDQuant W4A4 tensor family into kitchen tile-packed storage. |
| `deepcompressor-qwen-image-edit` | PTQ artifact directory | Import Qwen-Image-Edit DeepCompressor PTQ artifacts, build a natural SVDQuant W4A4 state dict, then reuse the same kitchen tile-pack writer. |

The base library does not import DeepCompressor, Nunchaku, ComfyUI, or
comfy-kitchen when using either mode. The DeepCompressor mode reads local
PyTorch artifact files directly and converts them into this repository's static
checkpoint contract.

The current supported format selector is:

```text
--format svdquant_w4a4
```

Natural safetensors source:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main export-int4 \
  --format svdquant_w4a4 \
  --source-format natural-safetensors \
  --source /absolute/path/to/natural_svdquant_w4a4.safetensors \
  --out runs/qwen-edit-2511/export-int4 \
  --device cuda:0 \
  --hash-output \
  --json
```

DeepCompressor PTQ artifact source:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main export-int4 \
  --format svdquant_w4a4 \
  --source-format deepcompressor-qwen-image-edit \
  --source /absolute/path/to/deepcompressor-ptq-artifacts \
  --out runs/qwen-edit-2511/export-int4 \
  --device cuda:0 \
  --hash-output \
  --json
```

The DeepCompressor source directory must contain:

```text
model.pt   required: source weights and unquantized copied tensors
scale.pt   required: <layer>.weight.scale.0 and optional .weight.scale.1
smooth.pt  optional: activation smoothing factors; missing entries use ones
branch.pt  required for every imported SVDQuant layer: low-rank a.weight/b.weight
```

For Qwen-Image-Edit the import bridge currently handles the SVDQuant attention
and MLP linears. It normalizes source variants such as
`img_mlp.net.2.linear` to the static output prefix `img_mlp.net.2`, applies the
known `attn.to_add_out` smooth-factor alias, marks unsigned-activation MLP down
projections, and leaves unrelated tensors as high-precision copies. AWQ W4A16
modulation import is a separate future bridge.

When `--out` is a directory, the exporter writes:

```text
<out>/diffusion_pytorch_model.svdquant_w4a4.safetensors
<out>/export_report.json
```

Each detected SVDQuant layer is identified by a `.comfy_quant` JSON tensor with
`{"format":"svdquant_w4a4"}` and is exported as:

```text
<layer>.weight        natural (N, K/2) int8 signed-INT4 pairs
                      -> tile-packed (N/128, K/64, 32, 128) int8
<layer>.weight_scale  natural (K/64, N)
                      -> tile-packed (N/128, K/64, 128)
<layer>.proj_up       natural (N, R)
                      -> tile-packed (N/128, R, 128)
<layer>.smooth_factor unchanged shape
<layer>.proj_down     unchanged shape
<layer>.bias          unchanged shape when present
<layer>.comfy_quant   patched JSON with
                      {"format":"svdquant_w4a4","layout":"kitchen_tile_packed_w4a4"}
```

All other tensors are copied into the output checkpoint. The writer is useful
for validating the committed storage contract while the target consumer support
is still being finalized.

## Validate / export artifact metadata

```bash
PYTHONPATH=src python -m comfy_quants.cli.main validate \
  --artifact runs/qwen-image-2512/fp8-static-v0/artifact \
  --baseline Qwen/Qwen-Image-2512 \
  --smoke-set examples/prompts_t2i.jsonl \
  --out runs/qwen-image-2512/fp8-static-v0/validation

PYTHONPATH=src python -m comfy_quants.cli.main export \
  --artifact runs/qwen-image-2512/fp8-static-v0/artifact \
  --format safetensors_quant \
  --out runs/qwen-image-2512/fp8-static-v0/export
```

## Compatibility alias

The previous package name still works as a transition shim:

```bash
PYTHONPATH=src python -m genquant.cli.main doctor
```

Do not add new docs under the old name.
