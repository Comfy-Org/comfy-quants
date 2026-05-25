# INT4 Full Pipeline Target

## Final target

The INT4 target for this repository is an end-to-end, ComfyUI-targeted pipeline:

```text
BF16/FP16 Qwen-Image-Edit diffusion checkpoint
  + calibration/edit prompt set
  -> activation/stat collection
  -> smoothing solve
  -> low-rank branch solve
  -> GPTQ/Hessian W4 group-64 weight quantization
  -> AWQ W4A16 Qwen modulation branch
  -> kitchen tile-pack layout transform
  -> one ComfyUI-compatible safetensors checkpoint
```

The primary artifact is a single large file:

```text
<out>/diffusion_pytorch_model.svdquant_w4a4.safetensors
<out>/quantization_report.json
```

This base library must not import, embed, or depend on ComfyUI or
comfy-kitchen at runtime. DeepCompressor and Nunchaku are allowed only as
optional development/oracle dependencies for parity checks. Development may
inspect those projects to
write static contracts, but committed code owns its contracts and writers.

## Command surface

Direct dense-checkpoint-to-tile-pack entry point:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --calibration /absolute/path/to/calibration.jsonl \
  --out runs/qwen-edit-2511/int4-svdquant-w4a4 \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

Initial calibrated solver mode with precomputed activation statistics:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
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

Experimental GPTQ solver mode with precomputed activation statistics and GPTQ
Hessians:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --activation-stats /absolute/path/to/int4_activation_stats.json \
  --gptq-hessian-stats /absolute/path/to/int4_gptq_hessian_stats.json \
  --quantization-mode svdquant_gptq_experimental \
  --out runs/qwen-edit-2511/int4-svdquant-w4a4-gptq-experimental \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

Experimental GPTQ solver mode with output-error low-rank calibration:

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
  --out runs/qwen-edit-2511/int4-svdquant-w4a4-gptq-output-error \
  --rank 64 \
  --device cuda:0 \
  --hash-output \
  --json
```

Dry-run planning without writing a checkpoint:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --out runs/qwen-edit-2511/int4-svdquant-w4a4-plan \
  --dry-run \
  --json
```

Calibrated dry-run with stats coverage validation:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --activation-stats /absolute/path/to/int4_activation_stats.json \
  --quantization-mode calibrated_svdquant \
  --out runs/qwen-edit-2511/int4-svdquant-w4a4-calibrated-plan \
  --dry-run \
  --json
```

In calibrated mode, dry-run opens only the selected weight metadata from the
local safetensors source, loads the activation-stats JSON, and validates:

- every selected linear resolves to one stats key;
- each stats vector length matches the dense weight input-channel count;
- malformed stats files are reported as a coverage load error.

The report writes a structured `activation_stats_coverage` block:

```json
{
  "state": "valid",
  "selected_layer_count": 1,
  "loaded_layer_count": 1,
  "matched_layer_count": 1,
  "missing_layer_count": 0,
  "shape_mismatch_count": 0
}
```

If coverage is invalid, the CLI writes the report, returns
`dry_run_validation_failed`, and exits non-zero. This gives a cheap preflight
before launching the full GPU quantization/export pass.

## Current implementation boundary

`quantize-int4` writes the final artifact shape directly. It selects
Qwen-Image-Edit attention/MLP linears from the static adapter, builds natural
SVDQuant tensors, then reuses the kitchen tile-pack writer.

The default bootstrap mode is reported as:

```text
quantization_mode: weight_only_initialization
pipeline_kind: direct_quantize_to_kitchen_tilepack
```

The initial calibrated mode is reported as:

```text
quantization_mode: calibrated_svdquant
activation_stats_state: loaded
pipeline_kind: direct_quantize_to_kitchen_tilepack
```

It consumes a JSON activation-stats file, solves a deterministic per-channel
smoothing factor, quantizes the smoothed dense weight with signed W4 group-64
round-to-nearest scales, emits an SVD residual branch as `proj_down` /
`proj_up`, and emits AWQ W4A16 tensors for Qwen modulation linears when those
weights are present.

This mode is intentionally reported as experimental:

```text
algorithm_state: experimental_smooth_rtn_svd_no_gptq
publishable_svdquant_gptq: false
gptq_state: not_implemented
runtime_contract_state: static_artifact_contract_only
mixed_quantization_state: svdquant_only_awq_modulation_not_implemented
```

The final SVDQuant target is mixed quantization: SVDQuant W4A4 for attention/MLP
linears plus AWQ W4A16 for Qwen modulation linears.  The current pipeline has:

- `calibrated_svdquant`: smoothing + residual-SVD + RTN for SVDQuant layers;
- `svdquant_gptq_experimental`: smoothing + raw-Hessian-to-post-smoothing basis
  transform + residual-branch subtraction + repo-native GPTQ for SVDQuant
  layers. By default the residual branch is initialized from the weight-space
  residual; `--lowrank-calibration output_error` instead fits the low-rank
  branch from captured activation/output error before the same GPTQ step;
- AWQ W4A16 modulation quantization for `img_mod.1` / `txt_mod.1` when those
  tensors exist in the dense source checkpoint.

For a full Qwen checkpoint with modulation tensors, GPTQ experimental reports:

```text
algorithm_state: experimental_svdquant_gptq_awq_runtime_unverified
publishable_svdquant_gptq: false
gptq_state: layer_core_integrated
runtime_contract_state: static_artifact_contract_only
runtime_reference_state: repo_runtime_like_activation_w4_branch_oracle_runtime_unverified
lowrank_branch_input_basis: raw
proj_down_smooth_folded: true
lowrank_calibration: weight_residual
mixed_quantization_state: experimental_svdquant_w4a4_awq_w4a16_runtime_unverified
```

This is still not publishable because the default full CLI export uses
weight-residual branch initialization, while the optional output-error low-rank
calibration path and the external mixed runtime full-inference path remain
unverified. The
repository has a local SVDQuant runtime-like layer oracle for activation W4 and
explicit low-rank branch basis checks. `make-int4-runtime-fixture` writes a
deterministic single-layer safetensors golden case for external fused-runtime
comparison, but that fixture is still not a substitute for external kernel
parity or full
ComfyUI inference.  See `int4_svdquant_runtime_contract.md` for the runtime
contract gates.

The emitted `proj_down` basis is deliberately explicit.  By default, direct
exports store the raw-input Kitchen/Nunchaku basis:

```text
branch = x @ proj_down_runtime @ proj_up.T
proj_down_runtime = proj_down_post_smoothing / smooth_factor[:, None]
```

and report `lowrank_branch_input_basis: raw` and
`proj_down_smooth_folded: true`.  The retained `post_smoothing` option is
intentionally not the default; selecting either basis is still only an artifact
layout choice until the external fused-runtime path is closed.

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

The runtime-independent activation reduction stage is implemented. A capture
plan can be generated from a dense checkpoint and normalized calibration
records:

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
capture_plan.json
activation_samples.template.jsonl
capture_report.json
```

The plan lists each selected linear input tensor, its expected input/output
channel counts from the source checkpoint, and the stats keys accepted by the
calibrated writer. This stage is plan-only. It does not run Qwen-Image-Edit
forward passes, does not write activation tensors, and does not import ComfyUI.

The plan can be materialized into a reducer-ready sample manifest:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib materialize-int4-capture \
  --plan runs/qwen-edit-2511/calib-v0/capture-plan/capture_plan.json \
  --out runs/qwen-edit-2511/calib-v0/capture-run \
  --json
```

This writes `activation_samples.jsonl` and
`capture_materialization_report.json`. It does not run forward passes and does
not write activation safetensors. A materialized row looks like:

```json
{"case_id":"case-1","layer":"transformer_blocks.0.attn.to_q","file":"activation_tensors/case-1.safetensors","tensor":"transformer_blocks.0.attn.to_q.input","channel_dim":-1}
```

External capture code is responsible for executing the model and filling the
referenced safetensors files under `capture-run/activation_tensors/`. The helper
`write_int4_activation_case_safetensors(...)` can be used by such capture code
to validate the captured tensor names and channel counts against the plan before
writing one case file.

Then this repository reduces those dumps into the stats file used for smoothing:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib reduce-int4-activations \
  --samples runs/qwen-edit-2511/calib-v0/capture-run/activation_samples.jsonl \
  --input-root runs/qwen-edit-2511/calib-v0/capture-run \
  --out runs/qwen-edit-2511/calib-v0/int4-stats \
  --channel-dim -1 \
  --json
```

The same captured layer-input dumps can now be reduced into per-layer GPTQ
Hessian artifacts:

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

This writes:

```text
runs/qwen-edit-2511/calib-v0/int4-gptq-hessians/
├── int4_gptq_hessian_stats.json
└── gptq_hessians/
    └── <layer>.safetensors
```

The Hessian manifest stores relative tensor file paths and one normalized
`hessian` tensor per layer. `quantize-int4 --quantization-mode
svdquant_gptq_experimental` reads this manifest, treats it as raw-input-basis,
applies the smoothing-basis transform, and calls the repo-native GPTQ layer
helper after branch handling. With `--lowrank-calibration output_error`, the
same command also reads `--activation-samples` and validates sample coverage and
channel shape before fitting the low-rank branch from output error.

The remaining large work item is the real Qwen-Image-Edit runtime capture
executor that runs forward passes and writes those local activation dumps
without adding ComfyUI as a runtime dependency.

## Module ownership

```text
algorithms/int4_svdquant/
  config.py           pipeline options
  calibration.py      activation sample references and stats reduction
  hessian.py          activation-dump to GPTQ Hessian artifact reducer
  layer_selection.py  Qwen layer-selection glue over static adapters
  stats.py            activation-stat data contract and JSON helpers
  smoothing.py        deterministic per-channel smoothing solve
  lowrank.py          residual SVD branch solve
  gptq.py             runtime-independent Hessian builder and GPTQ W4 solve
  weight_quant.py     signed W4 group-64 RTN/GPTQ SVDQuant layer helpers
  runtime_reference.py activation-W4 and branch-basis layer oracle
  runtime_fixture.py  deterministic SVDQuant W4A4 external-parity handoff fixture

algorithms/awq_w4a16/
  weight_quant.py     kitchen-native AWQ W4A16 modulation weight quantization
  qwen_modulation.py  optional Qwen modulation reorder helper for bridge paths
  runtime_fixture.py  deterministic AWQ W4A16 external-parity handoff fixture

backends/int4_full_pipeline_export.py
  safetensors source reading, direct dense-to-natural SVDQuant conversion,
  call into the kitchen tile-pack writer, report generation

backends/int4_kitchen_export.py
  reusable natural-SVDQuant-to-kitchen-tile-pack safetensors writer

backends/activation_capture/qwen_image_edit.py
  plan-only Qwen-Image-Edit INT4 activation target generation

backends/activation_capture/materialize.py
  runtime-independent sample-manifest materialization and case safetensors writer

formats/kitchen_tilepack.py
  storage layout transform only; no model-family logic

model_adapters/qwen_image_edit_int4.py
  Qwen-Image-Edit static INT4 layer suffix mapping, activation flags, and AWQ
  modulation prefix selection
```

This keeps the format module model-agnostic and prevents a super-class or a
single large model/format matrix file.

## Difference from `export-int4`

`export-int4` remains an import/repack utility:

```text
natural SVDQuant safetensors or local DeepCompressor-style PTQ files
  -> kitchen tile-pack writer
```

It is useful for validation and artifact migration, but it is not the final INT4
route. The final route is `quantize-int4`, which starts from a dense checkpoint
and writes the kitchen tile-packed file directly.
