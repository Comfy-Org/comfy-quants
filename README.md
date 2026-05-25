# Comfy Quants

Comfy Quants is a **ComfyUI-targeted, CLI-first offline quantization sub-library**.

Initial production scope:

- **Model families**: Qwen-Image and Qwen-Image-Edit.
- **Quantization formats**: `fp8_e4m3` / `fp8_e5m2` for Qwen diffusion-model checkpoints, plus the `svdquant_w4a4` INT4 kitchen tile-pack checkpoint contract.
- **Default route**: static FP8 planning plus local safetensors payload/checkpoint export.
- **INT4 route**: Qwen-Image-Edit direct `quantize-int4` dense-checkpoint-to-`svdquant_w4a4` kitchen tile-pack single-file export; `export-int4` remains an import/repack bridge for validation artifacts.
- **Interface**: CLI + Python SDK first.
- **Release boundary**: no UI/UX, no Web UI, no ComfyUI runtime dependency, no embedded ComfyUI checkout.
- **Production GPU baseline**: NVIDIA RTX PRO 6000 Blackwell class 96GB; default VRAM safety budget is 88GB.

`fp8_e5m2` maps to ComfyUI checkpoint quant format `float8_e5m2`, torch dtype `torch.float8_e5m2`, and safetensors dtype `F8_E5M2`. It is not `mxfp8` and it is not `fp8_e4m3fn_fast`.

## Documentation / quantization index

Start from this index when choosing a quantization route. This repository root is
the publish target for `Comfy-Org/comfy-quants`; implementation code lives under
`src/comfy_quants`. Do not create or use a nested `comfy-quants/` staging
directory.

| Route / topic | Use this when | Doc |
| --- | --- | --- |
| **FP8 E4M3 / E5M2 static quantization** | Quantize Qwen-Image or Qwen-Image-Edit diffusion checkpoints to ComfyUI FP8 checkpoint formats. | [`docs/cli.md`](docs/cli.md) |
| **Qwen-Image-Edit-2511 INT4 comfy-quants integrated route** | Directly export a dense Qwen-Image-Edit-2511 transformer checkpoint to `svdquant_w4a4` ComfyUI/comfy-kitchen tile-pack safetensors. This is the current recommended INT4 operation guide. | [`docs/qwen_image_edit_2511_int4_comfy_quants.md`](docs/qwen_image_edit_2511_int4_comfy_quants.md) |
| **INT4 full pipeline target** | Read the implementation boundary and end-to-end target for activation capture, smoothing, low-rank branch, GPTQ, AWQ modulation, and tile-pack export. | [`docs/int4_full_pipeline.md`](docs/int4_full_pipeline.md) |
| **SVDQuant W4A4 tile-pack format** | Inspect the `svdquant_w4a4` / `kitchen_tile_packed_w4a4` storage contract. | [`docs/formats/svdquant_w4a4_kitchen_tilepack.md`](docs/formats/svdquant_w4a4_kitchen_tilepack.md) |
| **AWQ W4A16 modulation format** | Inspect the Qwen `img_mod.1` / `txt_mod.1` AWQ W4A16 modulation tensor contract. | [`docs/formats/awq_w4a16.md`](docs/formats/awq_w4a16.md) |
| **INT4 runtime contract / parity gates** | Check what still must be validated before treating INT4 artifacts as runtime-publishable. | [`docs/int4_svdquant_runtime_contract.md`](docs/int4_svdquant_runtime_contract.md) |
| **INT4 oracle notes** | Development notes for oracle/runtime comparison work. | [`docs/int4_runtime_oracle_notes.md`](docs/int4_runtime_oracle_notes.md) |
| **Architecture** | Repository layout, package boundaries, and adapter/format/backend responsibilities. | [`docs/architecture.md`](docs/architecture.md) |

Quick route summary:

```text
FP8:
  dense checkpoint + config
    -> comfy_quants quantize / export-model
    -> diffusion_pytorch_model.fp8_e4m3.safetensors
    -> diffusion_pytorch_model.fp8_e5m2.safetensors

INT4:
  Qwen-Image-Edit-2511 dense transformer checkpoint
    -> comfy_quants quantize-int4
    -> diffusion_pytorch_model.svdquant_w4a4.safetensors
```

## ComfyUI target, artifact contract only

This library exists so exported quantized model artifacts are usable by ComfyUI. That does **not** mean the quantizer should run ComfyUI or depend on ComfyUI.

The rule is:

1. During development, review the relevant ComfyUI model definitions so the model adapter contract matches what ComfyUI expects.
2. Commit the resulting static adapter/format contract into this repository.
3. When the library/CLI runs, use only Comfy Quants code to inspect, plan, quantize, validate, and export.
4. The exported artifact must be ComfyUI-compatible.

Hard release boundary:

- Do not vendor/copy ComfyUI into this repo or package.
- Do not import or call `comfy.*` from library/CLI code.
- Do not launch a ComfyUI server, workflow engine, UI, or custom node to interpret model formats.
- Do not accept a ComfyUI checkout as quantization configuration.
- Do not use ComfyUI as a hidden parser fallback.

In short: **development review of ComfyUI is allowed; package coupling to ComfyUI is not.**

The separate ComfyUI custom-node adapter should live in its own package; it may import ComfyUI because it is not part of this base library boundary.

## Decoupled quantization architecture

Comfy Quants separates responsibilities so one quantization format can serve multiple model families:

```text
model_adapters/   model-family contract and module selection
formats/          reusable tensor/storage format declarations, e.g. fp8_e4m3, fp8_e5m2
algorithms/       quantization procedure, e.g. fp8_static scale solving/planning
backends/         export backend boundary, e.g. safetensors/torch reference
registry/         Comfy Kitchen-style discovery for adapters/formats/algorithms/backends
```

Example ownership:

- `qwen_image` and `qwen_image_edit` adapters load committed static Qwen contracts and decide which Qwen components are quantizable and which stay high precision.
- `fp8_e4m3` and `fp8_e5m2` are reusable format specs; they are not owned by Qwen and can be reused by additional ComfyUI model families.
- `svdquant_w4a4` is a reusable INT4 tensor/storage format; Qwen-specific layer mapping lives in model adapters and INT4 pipeline/import backends, not in the format codec.
- `fp8_static` is the algorithm route that consumes a model graph plus a quant policy and emits a plan/artifact metadata.
- FP8 runtime metadata shared by multiple writers lives in `formats/fp8_common.py`; it is deliberately small and format-focused, not a model/format matrix.

This avoids a super-class/super-file design: adding a new model family should add a small adapter contract; adding a new quantized storage should add a reusable format spec; adding a new procedure should add an algorithm module.

## Comfy Kitchen influence

The package is named `comfy_quants`, matching the Comfy sub-library style used by `comfy_kitchen`.

It follows the same broad pattern:

- public package `__init__.py`;
- built-in imports for registration side effects;
- central process-local registry;
- backend/format/algorithm/model-adapter modules with clear ownership;
- core library separated from UI/custom-node integration.

## Quick start from source

```bash
PYTHONPATH=src python -m comfy_quants.cli.main doctor
PYTHONPATH=src python -m comfy_quants.cli.main doctor --json

PYTHONPATH=src python -m comfy_quants.cli.main inspect \
  --model Qwen/Qwen-Image-2512 \
  --family qwen_image \
  --out runs/qwen-image-2512/inspect

PYTHONPATH=src python -m comfy_quants.cli.main quantize \
  --config configs/qwen_image_2512_fp8_static.yaml \
  --work-dir runs/qwen-image-2512/fp8-e4m3-static-v0 \
  --dry-run

PYTHONPATH=src python -m comfy_quants.cli.main quantize \
  --config configs/qwen_image_2512_fp8_e5m2_static.yaml \
  --work-dir runs/qwen-image-2512/fp8-e5m2-static-v0 \
  --dry-run

PYTHONPATH=src python -m pytest -q
```

Legacy bootstrap command compatibility is retained for now:

```bash
PYTHONPATH=src python -m genquant.cli.main doctor
```

…but new docs and production usage should use `comfy_quants` / `comfy-quants`.

### Optional INT4 oracle dependencies

The production library still exports artifacts with its own writer and does not
embed ComfyUI or comfy-kitchen. For INT4 parity work, it is acceptable to install
DeepCompressor and Nunchaku as development/oracle dependencies:

```bash
python -m pip install -e '.[int4-oracle]'
# Or, for local maintained checkouts during oracle debugging:
python -m pip install -e /path/to/deepcompressor -e /path/to/nunchaku
```

These dependencies are for tests, import bridges, and contract comparison only;
passing those checks still does not make an INT4 artifact publishable until the
external ComfyUI full-inference PNG is correct.

## Current implementation status

Current implementation provides static Qwen contracts, reusable FP8 E4M3/E5M2 format declarations, a local safetensors payload writer, and a full inference checkpoint exporter for selected Qwen diffusion-model tensors:

- Qwen-Image and Qwen-Image-Edit inspection use committed contracts under `model_adapters/qwen_contracts/`.
- `fp8_static` dry-run creates a quantization plan and a populated tensor index. The default Qwen-Image and Qwen-Image-Edit policies currently select 839 weight tensors for both `fp8_e4m3` and `fp8_e5m2`.
- Torch reference FP8 helpers provide scale solving, payload creation, and dequant roundtrip checks for isolated tensors.
- Non-dry-run `quantize` accepts `model.source: local` with `model.model_id` pointing to a single `.safetensors` file, a safetensors index JSON, or a directory containing a safetensors index and local shards. It writes selected FP8 weight payload bytes plus FP32 scales according to `artifact/quant_tensor_index.json`.
- `export-model` writes a ComfyUI-loadable single `.safetensors` diffusion-model checkpoint. Selected weights are stored as torch FP8 tensors and each quantized layer receives `weight_scale`, `input_scale`, and `comfy_quant` side tensors.
- `quantize-int4` is the direct INT4 pipeline entry point. It starts from a dense Qwen-Image-Edit safetensors checkpoint and writes a single `svdquant_w4a4` kitchen tile-packed safetensors checkpoint plus `quantization_report.json`. The target Qwen bundle is mixed quantization: attention/MLP linears use SVDQuant W4A4, while `img_mod.1` / `txt_mod.1` modulation linears are emitted as AWQ W4A16 when those tensors are present. The public modes are:
  - `weight_only_initialization`: calibration-free SVDQuant tensor bootstrap with identity smoothing and a zero branch;
  - `calibrated_svdquant`: activation-stats smoothing plus residual-SVD branch, followed by groupwise RTN for SVDQuant weights;
  - `svdquant_gptq_experimental`: activation stats plus raw-input GPTQ Hessian manifest, smoothing, Hessian basis transform, residual-branch subtraction, and the repo-native GPTQ layer core for SVDQuant weights.
  Direct exports now default to the raw low-rank branch basis (`lowrank_branch_input_basis: raw`, `proj_down_smooth_folded: true`) because the target Kitchen/Nunchaku SVDQuant runtime computes the branch from raw activations. The retained `--lowrank-branch-input-basis post_smoothing` option is for internal/reference experiments only. For `act_unsigned` layers, the target contract applies the GELU unsigned offset (`x + 0.171875`) only on the main activation-W4 path; the low-rank branch still uses raw `x`. The GPTQ mode defaults to `--lowrank-calibration weight_residual`; the experimental `--lowrank-calibration output_error` path is wired into the full CLI and requires the activation sample manifest in addition to activation stats and Hessians. All INT4 modes still report `publishable_svdquant_gptq: false` until DeepCompressor/Nunchaku/comfy-kitchen layer parity and external full-inference image validation are closed.
- `make-int4-runtime-fixture` writes a deterministic single-layer `svdquant_w4a4` kitchen tile-packed safetensors fixture plus oracle input/output tensors and `runtime_fixture_report.json`. `make-awq-runtime-fixture` does the same for one kitchen-native AWQ W4A16 modulation-style layer. Both fixture reports include an `external_harness_contract` naming `fixture.input`, `fixture.expected_output`, required layer tensors, and the expected external handoff tensor `runtime.output`. `validate-runtime-fixture-output` compares that saved `runtime.output` tensor with the fixture oracle and writes `runtime_fixture_output_validation_report.json`. `validate-int4-runtime-readiness` aggregates SVDQuant, AWQ, mixed-dispatch, and full-inference validation reports into a conservative gate checklist. These commands are parity handoff tools only; they do not import ComfyUI and keep `publishable_svdquant_gptq: false`.
- `calib plan-int4-capture` writes a static Qwen-Image-Edit activation-capture plan and sample-manifest template for the calibrated INT4 path. `calib materialize-int4-capture` turns that plan plus calibration records into a reducer-ready activation sample manifest. `calib reduce-int4-activations` produces smoothing stats and `calib reduce-int4-gptq-hessians` produces per-layer GPTQ Hessian artifacts from the same captured activation dumps. These commands are runtime-independent: they do not run forward passes and do not import ComfyUI.
- `export-int4` writes the same single `svdquant_w4a4` kitchen tile-packed safetensors checkpoint from already-quantized inputs. It can repack an already-natural SVDQuant W4A4 safetensors source or import Qwen-Image-Edit DeepCompressor-style PTQ artifacts from local `model.pt` / `scale.pt` / `smooth.pt` / `branch.pt` files before repacking. This is an import bridge, not the final quantization route.
- Safetensors source coverage can check selected tensor names and shapes before payload writing.
- Runtime forward activation-capture execution and image-quality validation are separate work packages.
- External-engine AWQ W4A16 modulation import and full downstream image inference validation remain separate work packages; the direct dense-checkpoint path can already emit AWQ modulation tensors using the kitchen-native `(uint4 - 8) * scale + zero` formula, but the mixed runtime path is still unverified.

Format mapping for full inference checkpoint export:

```text
fp8_e4m3 -> torch.float8_e4m3fn -> comfy_quant {"format":"float8_e4m3fn","full_precision_matrix_mult":true}
fp8_e5m2 -> torch.float8_e5m2   -> comfy_quant {"format":"float8_e5m2","full_precision_matrix_mult":true}
```

Minimal local config shape:

```yaml
model:
  family: qwen_image
  model_id: /absolute/path/to/local-transformer
  source: local
quant:
  algorithm: fp8_static
  target_dtype: fp8_e5m2   # or fp8_e4m3
```

Selected-payload command:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize \
  --config local-qwen-fp8.yaml \
  --work-dir runs/qwen-image-local/fp8-static-v0 \
  --device cuda:0 \
  --json
```

Single-file inference checkpoint command:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main export-model \
  --config configs/qwen_image_edit_2511_fp8_e5m2_static.yaml \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --out runs/qwen-edit-2511/export-e5m2 \
  --device cuda:0 \
  --hash-output \
  --json
```

Direct INT4 kitchen tile-pack quantization from a dense Qwen-Image-Edit checkpoint:

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

Direct INT4 with precomputed activation stats:

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

Direct INT4 with precomputed activation stats and GPTQ Hessians:

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

Direct INT4 with the experimental output-error low-rank branch calibration:

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

That path fits the SVD branch from captured activation/output error, subtracts
the branch-effective weight, and then runs GPTQ in the post-smoothing basis. It
does not remove the need for runtime parity and full external ComfyUI inference
validation.

Preflight the same calibrated run without writing checkpoint bytes:

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

For `calibrated_svdquant`, dry-run loads the activation-stats JSON and checks
that every selected linear has a matching stats key whose `input_amax` length
equals the dense weight input-channel count. Missing or shape-mismatched stats
are written to `quantization_report.json` under `activation_stats_coverage`; the
CLI returns `dry_run_validation_failed` instead of silently planning an invalid
quantization.

The INT4 reports expose the implementation boundary. For a full Qwen checkpoint
with modulation tensors, the mixed static artifact path reports:

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

If a minimal test checkpoint contains no Qwen modulation tensors, the report
falls back to `mixed_quantization_state:
svdquant_only_awq_modulation_not_implemented`. If `calibrated_svdquant` is used
instead of `svdquant_gptq_experimental`, the SVDQuant weight solve remains the
explicit RTN milestone state `experimental_smooth_rtn_svd_no_gptq`.

See `docs/int4_svdquant_runtime_contract.md` before treating an INT4 checkpoint
as a final SVDQuant+GPTQ artifact.

The activation-stats reduction stage is available without binding this package
to a model runtime. Before running an external capture implementation, create
the target list and manifest template from the dense checkpoint and normalized
records:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib plan-int4-capture \
  --family qwen_image_edit \
  --source /absolute/path/to/diffusion_pytorch_model.safetensors \
  --records runs/qwen-edit-2511/calib-v0/records/calibration_records.jsonl \
  --out runs/qwen-edit-2511/calib-v0/capture-plan \
  --json
```

This writes `capture_plan.json`, `activation_samples.template.jsonl`, and
`capture_report.json`. The command is plan-only; it does not run model forward
passes, does not write activation tensors, and does not import ComfyUI.

Materialize the reducer manifest before running a capture implementation:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib materialize-int4-capture \
  --plan runs/qwen-edit-2511/calib-v0/capture-plan/capture_plan.json \
  --out runs/qwen-edit-2511/calib-v0/capture-run \
  --json
```

This writes `activation_samples.jsonl` and
`capture_materialization_report.json` under `capture-run/`. It does not create
activation tensors. The manifest uses the planned tensor names and stable case
filenames:

```json
{"case_id":"case-1","layer":"transformer_blocks.0.attn.to_q","file":"activation_tensors/case-1.safetensors","tensor":"transformer_blocks.0.attn.to_q.input","channel_dim":-1}
```

External capture code should consume `capture_plan.json`, run its own model
forward passes, and write one safetensors file per calibration case into
`capture-run/activation_tensors/`. The Python helper
`write_int4_activation_case_safetensors(...)` is available for capture
implementations that want this repository to validate tensor names/shapes and
write the safetensors case file. It still does not provide or invoke a model
runtime.

Then reduce those local dumps into the stats file:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main calib reduce-int4-activations \
  --samples runs/qwen-edit-2511/calib-v0/capture-run/activation_samples.jsonl \
  --input-root runs/qwen-edit-2511/calib-v0/capture-run \
  --out runs/qwen-edit-2511/calib-v0/int4-stats \
  --channel-dim -1 \
  --json
```

For the GPTQ step, reduce the same captured layer-input dumps into portable
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

This writes `int4_gptq_hessian_stats.json` plus one safetensors Hessian file per
layer under `gptq_hessians/`. The `svdquant_gptq_experimental` mode consumes
that manifest, transforms raw-input Hessians into the post-smoothing basis, and
uses them for the repo-native GPTQ layer solve. When invoked with
`--lowrank-calibration output_error`, the same command also consumes the
activation sample manifest so the low-rank branch is fitted before GPTQ rather
than only initialized from a weight residual. It is still not a publishable claim
until runtime parity and full external inference are validated.

When `--out` is a directory, this direct INT4 command writes:

```text
diffusion_pytorch_model.svdquant_w4a4.safetensors
quantization_report.json
```

INT4 kitchen tile-pack export from a DeepCompressor PTQ artifact directory:

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

The INT4 pipeline and import/export code read local files only. They do not import
ComfyUI or comfy-kitchen at package runtime. DeepCompressor and Nunchaku may be
installed as optional development/oracle dependencies for parity checks, but the
base package does not require them to export artifacts.

When `--out` is a directory, default checkpoint names are format-specific:

```text
diffusion_pytorch_model.fp8_e4m3.safetensors
diffusion_pytorch_model.fp8_e5m2.safetensors
diffusion_pytorch_model.svdquant_w4a4.safetensors
```

The selected-payload writer emits:

```text
artifact/
├── quant_tensor_index.json
├── payload_report.json
├── tensors/fp8_weights.safetensors
└── scales/fp8_static_scales.safetensors
```

The writer processes the tensors selected by the static graph and quantization policy. Missing selected checkpoint keys fail clearly instead of producing a partial artifact silently.

## Main directories

```text
src/comfy_quants/
├── comfy/            # static ComfyUI artifact contract metadata only
├── cli/              # CLI commands only; no UI/node code
├── sdk/              # stable Python API surface
├── core/             # schemas/domain objects; no hard dependency on ComfyUI/diffusers
├── model_adapters/   # Qwen static contracts, adapters, ModelGraph/QuantPolicy mapping
├── calibration/      # calibration manifests and normalized prompt/edit records
├── algorithms/       # quantization algorithms; primary route is fp8_static
├── formats/          # reusable quantized tensor formats; FP8 and INT4 storage contracts live here
├── backends/         # export/import backend boundaries; torch_ref and safetensors writers
├── jobs/             # resumable job/checkpoint store
├── registry/         # central registry, Comfy Kitchen style
├── validation/       # validation reports
└── utils/            # hashing/json/system helpers
```

See also:

- `2026-05-18-qwen-image-cli-first-offline-quantization-task-brief.md`
- `docs/architecture.md`
- `docs/cli.md`
- `docs/qwen_image_edit_2511_int4_comfy_quants.md`
- `docs/formats/svdquant_w4a4_kitchen_tilepack.md`
- `docs/formats/awq_w4a16.md`
