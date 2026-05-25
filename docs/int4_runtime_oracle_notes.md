# INT4 Runtime Oracle Notes

This document records external runtime facts used during development. ComfyUI
and comfy-kitchen are references only: production code in this repository must
not import, embed, or launch them. DeepCompressor and Nunchaku may be installed
as optional development/oracle dependencies for parity checks, but they are not
mandatory production dependencies.

## Current oracle status

- The base library has a static artifact writer and runtime-independent PyTorch
  reference math.
- The base library now has a runtime-like SVDQuant layer oracle for activation
  W4 quantization plus an explicit low-rank branch input basis. This is still a
  local oracle, not external fused-runtime validation.
- The mixed Qwen target remains non-publishable until a compatible external
  runtime branch can load the single safetensors file and run full image
  inference.
- `publishable_svdquant_gptq` must stay `false` while runtime parity is open.

## Qwen mixed INT4 layout

The target checkpoint is mixed quantization:

```text
attention / MLP linears  -> svdquant_w4a4
img_mod.1 / txt_mod.1   -> awq_w4a16
remaining tensors        -> copied high precision tensors
```

SVDQuant is not smoothing-only. The intended W4A4 solve order is:

```text
activation smoothing / shift handling
-> low-rank branch handling
-> subtract branch-effective weight from the smoothed dense weight
-> transform raw-input Hessian into the post-smoothing basis
-> grouped signed-INT4 GPTQ
-> runtime activation W4 and fused low-rank branch parity checks
```

The current full-checkpoint CLI path implements the smoothing-basis Hessian
transform and a grouped signed-INT4 GPTQ layer core. It still defaults to
initializing the low-rank branch from a weight-space residual. The optional
`--lowrank-calibration output_error` path is wired into `quantize-int4` and
requires activation samples; it still has not closed the external mixed-runtime
parity gates.

## Local SVDQuant runtime-like oracle

The local oracle APIs are:

```text
comfy_quants.algorithms.int4_svdquant.quantize_activation_w4_signed
comfy_quants.algorithms.int4_svdquant.quantize_activation_w4_unsigned
comfy_quants.algorithms.int4_svdquant.reference_svdquant_w4a4_linear_runtime
```

The signed activation helper uses per-row/per-group natural scales:

```text
scale = absmax(x_group) / 7
q = clamp(round(x_group / scale), -7, 7)
```

The unsigned helper uses:

```text
scale = absmax(x_group) / 15
q = clamp(round(x_group / scale), 0, 15)
```

Zero groups use scale `1`.  The SVDQuant signed quantizer emits `[-7, 7]`; the
byte storage codec still supports the full signed nibble range `[-8, 7]`.  For
`act_unsigned=True`, the target contract shifts the main path by `0.171875`
before smoothing and unsigned activation W4. The low-rank branch remains raw
`x`.

The runtime-like SVDQuant linear helper separates the main branch and low-rank
branch:

```text
if act_unsigned:
  x_main = (x + 0.171875) / smooth_factor
else:
  x_main = x / smooth_factor
main = dequant(activation_w4(x_main)) @ dequant(weight, weight_scale).T

raw low-rank basis:
  branch = x @ proj_down_runtime @ proj_up.T

post-smoothing low-rank basis:
  branch = (x / smooth_factor) @ proj_down_post_smoothing @ proj_up.T
```

The local tests cover the smooth-folding equivalence:

```text
proj_down_runtime = proj_down_post_smoothing / smooth_factor[:, None]
```

Default direct exports keep `proj_down` in the post-smoothing basis and mark
that contract in metadata:

```text
runtime_reference_state = repo_runtime_like_activation_w4_branch_oracle_runtime_unverified
lowrank_branch_input_basis = post_smoothing
proj_down_smooth_folded = false
```

The CLI can explicitly request a raw-input branch basis with
`--lowrank-branch-input-basis raw`; this applies the fold above and records
`lowrank_branch_input_basis = raw` / `proj_down_smooth_folded = true`.  This is
not the default writer behavior and still requires external fused-runtime
parity before it can be treated as publishable.

This oracle improves layer-level reasoning, but it does not prove that the
target external runtime uses the same signedness, scale storage, shift order,
packing, dtype, or fused low-rank accumulation.

### Runtime fixture generator

The repository also provides a CLI-first fixture generator for external parity
work:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main make-int4-runtime-fixture \
  --out runs/int4-runtime-fixtures/svdquant-w4a4-raw \
  --activation-signedness signed \
  --lowrank-branch-input-basis raw \
  --json
```

It writes a single-layer `svdquant_w4a4` kitchen tile-packed safetensors file
and `runtime_fixture_report.json`. The layer tensors live under
`fixture_layer.*`; oracle tensors live under `fixture.*`. The fixture includes
both the stored branch basis output and a post-smoothing-basis equivalent output
so raw-basis smooth folding can be checked before calling an external runtime.
The report also includes `external_harness_contract`, which fixes the single
layer handoff names:

```text
forward_input_tensor = fixture.input
expected_output_tensor = fixture.expected_output
external_output_tensor = runtime.output
validation_command = validate-runtime-fixture-output
```

The report deliberately keeps:

```text
external_runtime_validation = not_run
publishable_svdquant_gptq = false
```

This fixture is a handoff artifact for an external runtime harness. It does not
import ComfyUI, does not exercise a fused kernel, and does not validate full
Qwen image inference by itself.

### Runtime fixture output validation

Once an external runtime harness has executed a fixture, save its output to a
safetensors file containing `runtime.output` and compare it with the fixture
oracle:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main validate-runtime-fixture-output \
  --fixture runs/int4-runtime-fixtures/svdquant-w4a4-post/svdquant_w4a4_runtime_fixture.safetensors \
  --output runs/external-runtime-output/svdquant_w4a4_output.safetensors \
  --out runs/runtime-validation/svdquant-w4a4-post \
  --json
```

The validator is runtime-independent: it only reads safetensors files and does
not import ComfyUI or comfy-kitchen. When optional DeepCompressor/Nunchaku
dev-oracle packages are installed, those checks remain outside the production
export path. A pass means
only that one external output tensor matched `fixture.expected_output` within
the requested tolerance. The report keeps:

```text
validation_scope = single_layer_runtime_fixture_output_only
external_runtime_validation = single_layer_fixture_output_passed
publishable_svdquant_gptq = false
```

It is not evidence for full Qwen-Image/Edit load, mixed SVDQuant plus AWQ
dispatch, node registration, image inference quality, or a publishable
checkpoint.

### Runtime readiness gate aggregation

After collecting individual validation reports, aggregate them with:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main validate-int4-runtime-readiness \
  --svdquant-report runs/runtime-validation/svdquant-w4a4-raw/runtime_fixture_output_validation_report.json \
  --awq-report runs/runtime-validation/awq-w4a16/runtime_fixture_output_validation_report.json \
  --out runs/runtime-readiness/qwen-edit-2511-int4 \
  --json
```

The readiness report has four required gates:

```text
svdquant_w4a4_single_layer_runtime_parity
awq_w4a16_single_layer_runtime_parity
mixed_svdquant_w4a4_awq_w4a16_dispatch
full_qwen_image_edit_png_inference
```

The first two gates consume reports from `validate-runtime-fixture-output`.
The last two gates are external-runtime reports that must carry these validation
scopes:

```text
mixed_svdquant_w4a4_awq_w4a16_dispatch
full_qwen_image_edit_png_inference
```

The readiness report is intentionally conservative: it always keeps
`publishable_svdquant_gptq = false`. If all gates pass, it only marks
`publishable_candidate_after_manual_review = true`.

## AWQ W4A16 modulation oracle

A development comfy-kitchen branch for AWQ modulation defines kitchen-native
AWQ W4A16 tensors in these relative files:

```text
comfy_kitchen/tensor/awq_w4a16.py
comfy_kitchen/backends/eager/awq.py
comfy_kitchen/backends/cuda/ops/awq_w4a16.cu
```

The layout is:

```text
weight / qweight       (N, K/2)     int8 packed uint4, low nibble first
weight_scale / wscales (K/G, N)     fp scale
weight_zero / wzeros   (K/G, N)     fp additive zero/center
```

The dequantization formula is:

```text
W[n, k] = (uint4_weight[n, k] - 8) * weight_scale[k / G, n]
          + weight_zero[k / G, n]
y = x @ W.T + bias
```

This is the formula implemented by this repository's AWQ writer and reference
helper. `weight_zero` is therefore not an integer nibble zero-point; it is an
additive floating-point group center.

### Runtime fixture generator

The repository also provides a deterministic AWQ handoff fixture:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main make-awq-runtime-fixture \
  --out runs/int4-runtime-fixtures/awq-w4a16 \
  --json
```

It writes one kitchen-native AWQ W4A16 layer under `fixture_layer.*` and oracle
tensors under `fixture.*`, including `fixture.input`,
`fixture.expected_output`, `fixture.dequantized_weight`, and
`fixture.quantized_weight_uint4`.  Its report deliberately keeps:

```text
external_runtime_validation = not_run
publishable_svdquant_gptq = false
```

This mirrors the SVDQuant fixture pattern for AWQ modulation parity work.  It
is still local reference math, not a fused AWQ kernel check and not full mixed
Qwen image inference.

## DeepCompressor / Nunchaku conversion clues

A development DeepCompressor conversion script describes two transformations:

```text
examples/diffusion/scripts/convert_kitchen_native.py
```

- SVDQuant W4A4 is described as layout repacking from Nunchaku-style parameters
  into kitchen-native row-major/tile-packed tensors, including QKV split rules.
- AWQ W4A16 modulation is described as conversion from a Nunchaku/TRT-LLM-style
  packed layout into the kitchen-native `(N, K/2)` int8 layout, with an AWQ zero
  adjustment.

The public Nunchaku linear module gives useful runtime-path clues:

```text
nunchaku/models/linear.py
```

- `SVDQW4A4Linear.quantize(...)` quantizes activations and computes the
  low-rank branch activation in one runtime step.
- `SVDQW4A4Linear.forward_quant(...)` calls a fused W4A4 GEMM with activation
  scales, weight scales, low-rank activation, low-rank up projection, optional
  bias, and the selected activation signedness.
- `AWQW4A16Linear` stores `qweight`, `wscales`, and `wzeros` with a different
  engine-specific packed layout from the kitchen-native AWQ checkpoint layout.

The Nunchaku fused W4A4 activation kernel currently provides a strong clue for
the SVDQuant branch basis: the low-rank down projection is applied to the loaded
raw input accumulator before the activation epilogue divides by
`smooth_factor` for main-branch W4 quantization. In other words, that reference
runtime behaves like:

```text
branch = raw_x @ proj_down_runtime @ proj_up.T
main   = quantize_w4(raw_x / smooth_factor) @ qweight.T
```

For a branch solved in the repository's post-smoothing basis, the runtime-facing
raw-basis tensor is therefore:

```text
proj_down_runtime = proj_down_post_smoothing / smooth_factor[:, None]
```

The repository can emit that folded raw-basis contract with
`--lowrank-branch-input-basis raw`, but this observation is still a development
oracle. It must be confirmed against the exact target ComfyUI / kitchen fused
runtime before enabling any publishable gate.

These facts are used only to shape repository-native contracts and tests. They
are not production dependencies.

## Open runtime gaps

The following are not fully closed:

1. External parity for the local SVDQuant activation W4 formula and dtype
   behavior.
2. Fused-runtime parity for signed/unsigned activation dispatch, including
   the `act_unsigned` main-path-only `x + 0.171875` offset.
3. Exact fused low-rank branch application and accumulation dtype.
4. Confirmation that the full target ComfyUI / kitchen runtime produces the
   expected image with the raw-input `proj_down` basis.
5. Output-error low-rank branch calibration is wired into the full-checkpoint
   CLI, but it still needs external fused-runtime parity with the exact target
   activation/branch math.
6. Exact accepted `comfy_quant` metadata keys in the target mixed runtime.
7. Availability of a ComfyUI branch that registers both `svdquant_w4a4` and
   `awq_w4a16` in the mixed-precision dispatch path.
8. Full external load and image-inference validation for the final single-file
   Qwen-Image-Edit checkpoint.

Until those gaps are closed, current INT4 artifacts are static contract outputs,
not a publishable full-runtime parity claim.
