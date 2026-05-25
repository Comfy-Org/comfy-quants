# INT4 SVDQuant Runtime Contract

This document records the implementation boundary for the INT4 workstream.  It
is intentionally explicit because a checkpoint that has the right tensor names
and shapes is not automatically a publishable Qwen-Image-Edit SVDQuant model.

## Current implementation state

The current direct `quantize-int4` writer can produce a single safetensors file
with the `svdquant_w4a4` kitchen tile-packed artifact layout.  For
Qwen-Image-Edit sources that contain modulation weights, that single file is a
mixed INT4 artifact: attention/MLP linears are SVDQuant W4A4 and
`img_mod.1` / `txt_mod.1` linears are AWQ W4A16.  It has three public CLI
modes:

- `weight_only_initialization`
  - groupwise signed INT4 round-to-nearest weight quantization;
  - identity `smooth_factor`;
  - zero `proj_down` / `proj_up` low-rank branch;
  - no calibration and no GPTQ.
- `calibrated_svdquant`
  - consumes committed per-layer activation statistics;
  - solves a SmoothQuant-style per-input-channel smoothing factor;
  - quantizes the smoothed weights with groupwise signed INT4
    round-to-nearest;
  - solves an SVD residual branch;
  - emits AWQ W4A16 modulation tensors when modulation weights are present;
  - **does not run GPTQ**.
- `svdquant_gptq_experimental`
  - consumes committed per-layer activation statistics;
  - consumes a GPTQ Hessian manifest produced from the same captured layer-input
    dumps;
  - solves the smoothing factor first;
  - transforms raw-input Hessians into the post-smoothing input basis;
  - initializes/subtracts the low-rank residual branch;
  - runs the repo-native grouped signed-INT4 GPTQ layer core for SVDQuant
    attention/MLP weights;
  - emits AWQ W4A16 modulation tensors when modulation weights are present;
  - remains runtime-unverified and therefore non-publishable.

Reports and CLI JSON expose mode-specific state.  Examples:

```text
algorithm_state = weight_only_initialization_no_calibration_no_gptq
algorithm_state = experimental_smooth_rtn_svd_no_gptq
algorithm_state = experimental_svdquant_gptq_no_awq_runtime_unverified
algorithm_state = experimental_svdquant_gptq_awq_runtime_unverified
publishable_svdquant_gptq = false
gptq_state = not_implemented
gptq_state = layer_core_integrated
runtime_contract_state = static_artifact_contract_only
runtime_reference_state = repo_runtime_like_activation_w4_branch_oracle_runtime_unverified
lowrank_branch_input_basis = raw
proj_down_smooth_folded = true
mixed_quantization_state = svdquant_only_awq_modulation_not_implemented
mixed_quantization_state = experimental_svdquant_w4a4_awq_w4a16_runtime_unverified
```

Do not interpret any current INT4 mode as the final quality/runtime claim.
`calibrated_svdquant` is a layout-producing and calibration-plumbing milestone.
`svdquant_gptq_experimental` does perform smoothing followed by GPTQ for the
SVDQuant layers, but its low-rank branch and downstream mixed runtime path are
not yet validated as a complete SVDQuant+GPTQ runtime.

The library now also contains a runtime-independent GPTQ/Hessian core for one
linear layer.  That core can:

- build a per-layer Hessian from captured input activations;
- apply the SVDQuant smoothing divisor to those activations before Hessian
  construction;
- run grouped signed-INT4 GPTQ with damping, importance ordering, block error
  propagation, dead-column handling, and RTN fallback;
- initialize a weight-space low-rank branch and subtract that branch before the
  GPTQ solve in the layer helper;
- alternatively fit a layer-level output-error low-rank branch from activation
  samples before branch-effective-weight subtraction and GPTQ;
- emit the same natural-layout `weight` and `weight_scale` tensors consumed by
  the existing SVDQuant tile-pack writer.

The library also has a runtime-independent Hessian artifact reducer:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib reduce-int4-gptq-hessians \
  --samples runs/qwen-edit-2511/calib-v0/capture-run/activation_samples.jsonl \
  --input-root runs/qwen-edit-2511/calib-v0/capture-run \
  --out runs/qwen-edit-2511/calib-v0/int4-gptq-hessians \
  --device cuda:0 \
  --json
```

It writes `int4_gptq_hessian_stats.json` plus per-layer safetensors files under
`gptq_hessians/`, with each `hessian` tensor normalized by `2 / row_count`.
`svdquant_gptq_experimental` consumes this manifest.  The Hessian is treated as
raw-input-basis by the pipeline and transformed with the solved smoothing
divisor before GPTQ.  `calibrated_svdquant` intentionally remains the
non-GPTQ/RTN milestone mode, so its report fields still correctly say
`gptq_state = not_implemented`.

The output-error helper is still not a replacement for end-to-end validation.
The full-checkpoint CLI now has two explicit low-rank calibration modes:
`weight_residual` remains the default, while `output_error` is available through
`--lowrank-calibration output_error` together with `--activation-samples`.  That
optional path is wired into `svdquant_gptq_experimental` and covered by unit
tests for manifest coverage and full export, but it still needs target fused
runtime parity, AWQ modulation parity, and full mixed image inference validation
before it can become the publishable SVDQuant+GPTQ claim.

## Repo-native reference math

The package now includes small runtime-independent reference helpers:

```text
comfy_quants.algorithms.int4_svdquant.reference_svdquant_w4a4_linear
comfy_quants.algorithms.int4_svdquant.reference_svdquant_w4a4_linear_runtime
comfy_quants.algorithms.int4_svdquant.quantize_activation_w4_signed
comfy_quants.algorithms.int4_svdquant.quantize_activation_w4_unsigned
comfy_quants.algorithms.awq_w4a16.reference_awq_w4a16_linear
```

`reference_svdquant_w4a4_linear` executes the current dense/static tensor
contract with plain PyTorch:

```text
SVDQuant:
y = (x / smooth_factor)
    @ (dequant(weight, weight_scale) + proj_up @ proj_down.T).T
    + bias

AWQ:
dense_weight = (uint4_weight - 8) * weight_scale + weight_zero
y = x @ dense_weight.T + bias
```

`reference_svdquant_w4a4_linear_runtime` is a stricter layer oracle for the
mixed SVDQuant path.  It models the main branch as dynamic activation W4 plus
W4 weight dequantization:

```text
if act_unsigned:
  x_main = (x + 0.171875) / smooth_factor
else:
  x_main = x / smooth_factor
qx, ascales = activation_w4_quantize(x_main)
main = dequant(qx, ascales) @ dequant(weight, weight_scale).T
```

The activation W4 helpers use natural row-major scales with shape
`inputs.shape[:-1] + (K/G,)`.  The signed SVDQuant quantizer emits signed codes
in `[-7, 7]` with `absmax / 7`; the byte storage codec still accepts the full
signed nibble range `[-8, 7]` for compatibility.  The unsigned oracle uses
`absmax / 15` and unsigned codes in `[0, 15]`.  Zero-activation groups use scale
`1`.  For `act_unsigned=True`, the target Kitchen/Nunchaku contract applies the
GELU unsigned offset (`0.171875`) before smoothing and unsigned activation W4 on
the main path only.

The runtime-like SVDQuant helper also makes the low-rank branch basis explicit:

```text
raw basis:
  branch = x @ proj_down_runtime @ proj_up.T

post_smoothing basis:
  branch = (x / smooth_factor) @ proj_down_post_smoothing @ proj_up.T
```

This is important because the target fused runtime computes the branch from raw
activations.  Tests cover the equivalence needed when a solver first produces a
post-smoothing-basis branch:

```text
proj_down_runtime = proj_down_post_smoothing / smooth_factor[:, None]
```

By default, `quantize-int4` emits the raw-input basis and marks it in layer
metadata:

```text
lowrank_branch_input_basis = raw
proj_down_smooth_folded = true
```

The CLI retains `--lowrank-branch-input-basis post_smoothing` for controlled
internal/reference experiments.  That option stores the dense-math basis and
marks `proj_down_smooth_folded = false`; it is not the default target runtime
contract.  Changing the artifact branch basis does not by itself prove the
external fused runtime path, so the checkpoint remains runtime-unverified.

The SVDQuant helpers accept either natural tensors or the kitchen tile-packed
checkpoint tensors for `weight`, `weight_scale`, and `proj_up`.  These helpers
are deliberately not bindings to ComfyUI, comfy-kitchen, DeepCompressor, or
Nunchaku.  They are local reference targets for unit tests and future oracle
comparisons.  They do **not** close the external runtime parity gate: exact
activation W4 dtype behavior, signed/unsigned dispatch, fused SVDQuant kernel
accumulation, exact AWQ dispatch registration, and full mixed-runtime inference
still need to be checked against a compatible external runtime branch.

The AWQ side has the same boundary.  `make-awq-runtime-fixture` writes a
deterministic kitchen-native AWQ W4A16 single-layer safetensors file plus
`fixture.input`, `fixture.expected_output`, and dequantization oracle tensors.
It is useful as a handoff case for a future external AWQ dispatch comparison,
but its report remains `external_runtime_validation = not_run` and
`publishable_svdquant_gptq = false`.

## Target Qwen mixed quantization contract

The target Qwen-Image-Edit INT4 checkpoint is mixed quantization:

1. Attention and MLP linear layers use SVDQuant W4A4.
2. Qwen modulation linears use AWQ W4A16.
3. QKV tensors may need model-adapter split/merge rules before the storage
   layout is written.
4. The final deliverable remains a single safetensors file that a compatible
   ComfyUI runtime can load.

The base library is allowed to study upstream model definitions and conversion
scripts during development, but the released library must not import, embed,
vendor, or launch ComfyUI or comfy-kitchen. DeepCompressor and Nunchaku may be
installed as optional development/oracle dependencies for parity checks, but
they are not required production dependencies for artifact export.

## Target quantization order

The publishable W4A4 path is expected to follow this order:

```text
optional rotation
-> activation shifting / smoothing
-> low-rank branch calibration and branch-effective-weight subtraction
-> weight quantizer state calibration
-> GPTQ with layer input activations / Hessian
-> activation W4 runtime quantization contract
-> Qwen mixed-format export and tile packing
```

The important correction is that smoothing is not the end of the solve.  After
smoothing and low-rank handling, the W4 weight tensor must be quantized through
the GPTQ/Hessian path.  A simple per-group amax scale plus round-to-nearest is a
fallback or initialization strategy, not the final solver.

## Static SVDQuant artifact layout

Natural per-linear tensors:

```text
weight        (N, K/2) int8 bytes containing signed INT4 pairs
weight_scale  (K/64, N) fp16/bf16/fp32
smooth_factor (K,) fp16/bf16/fp32
proj_down     (K, R) fp16/bf16/fp32
proj_up       (N, R) fp16/bf16/fp32
bias          (N,) optional
comfy_quant   uint8 JSON tensor
```

Kitchen tile-packed per-linear tensors:

```text
weight        (N/128, K/64, 32, 128) int8
weight_scale  (N/128, K/64, 128)
smooth_factor (K,)
proj_down     (K, R)
proj_up       (N/128, R, 128)
bias          (N,) optional
comfy_quant   {"format": "svdquant_w4a4", "layout": "kitchen_tile_packed_w4a4", ...}
```

The default direct writer's `comfy_quant` payload includes:

```json
{
  "format": "svdquant_w4a4",
  "layout": "kitchen_tile_packed_w4a4",
  "lowrank_branch_input_basis": "raw",
  "proj_down_smooth_folded": true
}
```

The format module owns only these storage transforms and validators.  Model
selection, QKV splitting, and Qwen-specific suffix policy belong in model
adapters or import/export bridges.

## Runtime semantics that still need an oracle

The following points must be fully matched against an actual compatible runtime
before declaring the INT4 output publishable:

- external parity for the local activation W4 dynamic quantization oracle;
- parity for unsigned activation handling on selected MLP output projections,
  including the main-path-only `x + 0.171875` offset;
- whether runtime applies `smooth_factor` as input divide, weight multiply, or a
  fused equivalent;
- raw-input `proj_down` branch parity against the target fused runtime;
- the exact fused formula and accumulation dtype behavior for `proj_down` /
  `proj_up`;
- the AWQ W4A16 modulation dispatch path in the target ComfyUI/comfy-kitchen
  branch;
- exact accepted `comfy_quant` metadata keys for the target runtime branch;
- whether the target ComfyUI/comfy-kitchen branch has `svdquant_w4a4` and
  `awq_w4a16` registered in its mixed-precision dispatch.

Current local ComfyUI-style mixed-precision code inspected during development
does not expose a complete `svdquant_w4a4` / `awq_w4a16` runtime branch in the
mainline registry.  Until that branch is available or imported as a development
oracle, this library should describe its current INT4 output as a static
artifact contract only.  See `int4_runtime_oracle_notes.md` for the external
runtime observations that currently inform the static contract.

## Validation gates before publishable INT4

The project must pass these gates before `publishable_svdquant_gptq` can become
`true`:

1. **Layer runtime emulation parity**: a local, runtime-independent reference
   function exists for one SVDQuant layer and one AWQ modulation layer, and its
   assumptions are compared against the target runtime branch.
2. **GPTQ unit tests**: Hessian construction, damping, permutation, block error
   propagation, dead-column handling, and RTN fallback are covered on small
   deterministic tensors.
3. **Hessian artifact tests**: captured layer-input dumps reduce to portable
   per-layer Hessian manifests and safetensors tensors without importing a
   model runtime.
4. **Pipeline tests**: in `svdquant_gptq_experimental`, smoothing plus low-rank
   branch subtraction feeds GPTQ, not round-to-nearest; in
   `calibrated_svdquant`, RTN remains explicitly reported.
5. **Mixed Qwen export tests**: SVDQuant W4A4 and AWQ W4A16 layers coexist in one
   checkpoint with correct Qwen adapter mapping.
6. **Artifact schema validation**: tensor names, shapes, dtypes, and metadata are
   verified without importing any model runtime.
7. **Optional development oracle comparison**: compare layer outputs or exported
   artifacts against DeepCompressor / comfy-kitchen / Nunchaku checkouts when
   available. DeepCompressor/Nunchaku can be installed through the optional
   `int4-oracle` extra or a local editable checkout; they must not become
   mandatory production dependencies.
8. **External load and full inference**: only after the runtime branch is
   confirmed, load the produced single-file checkpoint in a separate ComfyUI
   checkout and run a full image generation smoke test.

`validate-int4-runtime-readiness` aggregates the current runtime reports into
one conservative checklist. It requires SVDQuant and AWQ single-layer reports
from `validate-runtime-fixture-output`, plus separate mixed-dispatch and
full-inference reports. The readiness report intentionally leaves
`publishable_svdquant_gptq = false`; if every gate passes it only marks the
artifact as a candidate for manual publishable review.

## Dependency boundary

This repository is a quantization/export library for ComfyUI-compatible
artifacts.  Its production code may depend on general libraries such as PyTorch
and safetensors, but not on ComfyUI or comfy-kitchen runtime packages.
DeepCompressor and Nunchaku are allowed as optional dev/oracle dependencies for
parity and import tests; exported artifacts must still be produced by this
package's own writer.  The library's responsibility is to write artifacts that a
compatible ComfyUI runtime can load; it should not carry or execute that runtime
itself.
