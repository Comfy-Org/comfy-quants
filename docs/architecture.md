# Comfy Quants Architecture

## 1. Positioning

Comfy Quants is a **ComfyUI-targeted offline quantization sub-library**.

It is created for ComfyUI users and must export artifacts ComfyUI can consume, but it is not a ComfyUI wrapper and must not embed or import ComfyUI.

Current production routes:

```text
Qwen-Image / Qwen-Image-Edit
  -> offline FP8 planning/quantization
  -> fp8_e4m3 or fp8_e5m2 artifact/checkpoint export
  -> schema + manifest + report + ComfyUI-compatible safetensors

Qwen-Image-Edit
  -> direct INT4 SVDQuant pipeline from dense checkpoint
  -> svdquant_w4a4 kitchen tile-pack checkpoint export
  -> single safetensors artifact plus quantization report

Qwen-Image-Edit
  -> external INT4 PTQ artifact import or natural SVDQuant input
  -> svdquant_w4a4 kitchen tile-pack checkpoint export
  -> single safetensors artifact plus export report
```

The base-library interface is:

```text
CLI + Python SDK
```

The separate custom-node adapter is a different package. It may import ComfyUI; this base library must not.

## 2. Static contract rule

For model families that ComfyUI already supports, Comfy Quants follows the ComfyUI artifact/model expectations.

That means:

1. During development, review ComfyUI's model definitions.
2. Convert the result into static adapter contracts owned by this repository.
3. When quantizing/exporting, use only those committed contracts.
4. Export quantized artifacts that remain compatible with ComfyUI.

It does **not** mean:

- loading ComfyUI as a model parser;
- using ComfyUI as a hidden fallback;
- importing `comfy.*` from the CLI/library;
- shipping a copied ComfyUI checkout inside this package.

For Qwen-Image and Qwen-Image-Edit, the local adapters use committed static contracts in `model_adapters/qwen_contracts/`. Those contracts encode transformer block names, tensor shapes, default precision actions, and artifact metadata used by CLI inspection and planning.

## 2.1 Non-embedding rule

Comfy Quants must not:

- vendor/copy ComfyUI into this repository or distribution;
- import or call `comfy.*` in library/CLI code;
- make `core/`, `formats/`, `algorithms/`, or `model_adapters/` depend on a live ComfyUI process or checkout;
- start the ComfyUI server or workflow engine from CLI quantization jobs;
- register or execute ComfyUI custom nodes during offline quantization;
- use ComfyUI as a hidden model-format parser.

Development review is allowed. The committed output of that review is a static contract in this repo.

## 3. Comfy Kitchen pattern applied

Comfy Kitchen demonstrates a useful sub-library shape:

- package-level public API;
- import built-in modules for auto-registration side effects;
- central registry object;
- backend capability selection separated from user interface code.

Comfy Quants applies the same shape to offline quantization:

```text
comfy_quants.__init__
  ├─ imports built-in adapters/formats/algorithms/backends for registration
  └─ exposes registry/list helpers

comfy_quants.registry.global_registry
  ├─ model adapters
  ├─ quant formats
  ├─ quant algorithms
  └─ backends/exporters
```

## 4. Directory ownership

```text
src/comfy_quants/
├── __init__.py
│   Public package boundary. Imports built-ins to trigger registration.
│
├── comfy/
│   Static artifact contract metadata for ComfyUI-compatible exports.
│
├── cli/
│   CLI commands. CLI invokes use-case/domain APIs and must not contain UI,
│   custom-node code, or ComfyUI imports.
│
├── sdk/
│   Python API surface for scripts and automation.
│
├── core/
│   Domain objects: config, graph, policy, manifest, provenance, compatibility,
│   errors. Core must not hard-import ComfyUI, diffusers, TensorRT, torchao, or
│   other heavy execution-specific dependencies.
│
├── model_adapters/
│   Family-specific static contracts and policy mapping. Qwen contracts live in
│   `model_adapters/qwen_contracts/`; graph construction lives in
│   `model_adapters/qwen_graph_builder.py`.
│
├── calibration/
│   Calibration set descriptors and normalized prompt/edit record loading.
│
├── formats/
│   Reusable quantized tensor/storage formats. `fp8_e4m3` and `fp8_e5m2` live
│   here so they can be reused by many model adapters instead of being embedded
│   in Qwen code.
│
├── algorithms/
│   Quantization procedures. `fp8_static` operates on ModelGraph/ModuleSpec/
│   QuantPolicy abstractions and outputs internal QuantTensor/artifact metadata.
│
├── backends/
│   Export backend boundaries. `torch_ref`, selected-payload safetensors export,
│   and full inference checkpoint export live here. Backend internals must not
│   leak into `core`.
│
├── jobs/
│   Job manifests, checkpoints, resume state, runner, memory, and offload
│   planning boundaries.
│
├── registry/
│   Central registry, inspired by Comfy Kitchen.
│
├── validation/
│   Numerical/image/edit validation reports.
│
└── utils/
    Hashing, JSON/YAML IO, system discovery.
```

## 5. Dependency boundary

```text
Allowed in core/:
  Python stdlib, small pure helpers, internal dataclasses.

Allowed in comfy/:
  Static artifact contract metadata only.

Allowed in model_adapters/:
  Static model-family contracts, module selection policy, calibration policy,
  and conversion to ModelGraph. No ComfyUI import/call/embedded process.

Allowed in formats/:
  Reusable quantized tensor/storage format declarations and codecs.

Allowed in algorithms/:
  Quantization planning logic over internal graph/policy structures.

Allowed in backends/:
  Backend-specific libraries and compatibility gates such as torch and
  safetensors.
```

## 6. Static ComfyUI artifact compatibility metadata

`doctor --json` reports the static artifact contract index, not local ComfyUI discovery:

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

`inspect` writes graph/inspection metadata:

```json
{
  "artifact_target": "comfyui",
  "contract_source": "comfy_quants",
  "contract_mode": "static_adapter_contract",
  "artifact_contract": {
    "schema_version": "qwen_image_contract.v1",
    "artifact_target": "comfyui"
  },
  "graph_kind": "static_model_contract",
  "tensor_coverage": "declared_tensors"
}
```

## 7. Format-to-model extension rule

Quantization format is not owned by a model adapter.

```text
fp8_e4m3 / fp8_e5m2 format specs
  ├─ used by qwen_image adapter policy
  ├─ used by qwen_image_edit adapter policy
  └─ can be reused by additional FLUX/SD3/Wan/etc. adapters
```

To add a new model family:

1. Add a small adapter module and a static contract under `model_adapters/<family>/` or a family-specific contract directory.
2. Register the adapter.
3. Reuse existing formats/algorithms when possible.
4. Only add a new format if storage/scale/runtime metadata actually changes.

To add a new format:

1. Add a `formats/<format>.py` spec/codec.
2. Add common runtime metadata only when multiple writers need it; for FP8 this is `formats/fp8_common.py`.
3. Register it in the format registry.
4. Keep it model-agnostic.
5. Let adapters opt into it through `QuantPolicy.target_dtype` and include/exclude rules.

This prevents a super-class or super-file from becoming the place where all model and format combinations are hard-coded.

## 8. INT4 kitchen-native route

The INT4 route is a mixed-format checkpoint bundle, not a single tensor cast:

```text
Qwen-Image-Edit transformer layers
  attention / MLP linears -> svdquant_w4a4 + kitchen_tile_packed_w4a4
  modulation linears      -> awq_w4a16 when runtime support is available
  remaining tensors       -> high precision copy
```

The reusable format modules are:

```text
formats/int4_common.py       byte-level signed INT4 helpers
formats/kitchen_tilepack.py  SVDQuant W4A4 tile layout transforms
formats/svdquant_w4a4.py     SVDQuant W4A4 format spec
formats/awq_w4a16.py         AWQ W4A16 format spec
```

The final INT4 route is a direct quantization pipeline, not an import bridge:

```text
BF16/FP16 Qwen-Image-Edit checkpoint + calibration/edit prompt set
  -> activation/stat collection
  -> smoothing / low-rank branch solve
  -> GPTQ/Hessian signed W4 group-64 weight quantization
  -> AWQ W4A16 Qwen modulation handling
  -> kitchen tile-pack layout
  -> single ComfyUI-compatible safetensors checkpoint
```

The direct command surface is:

```bash
comfy_quants quantize-int4 --family qwen_image_edit --format svdquant_w4a4 --source ... --out ...
```

The implementation now has three direct modes plus the reducer commands that feed them:

```text
weight_only_initialization       dense weights -> group-64 signed INT4, identity smoothing, zero branch
calibrated_svdquant              dense weights + activation stats -> smoothing solve -> residual SVD branch -> group-64 signed INT4 RTN
svdquant_gptq_experimental       dense weights + activation stats + GPTQ Hessians -> smoothing -> Hessian basis transform -> branch subtraction -> grouped signed-INT4 GPTQ
calib reduce-int4-gptq-hessians  captured layer inputs -> portable per-layer GPTQ Hessian manifest
```

The calibrated and GPTQ modes deliberately start from static activation/Hessian
files so the base library stays independent from any model runtime. Runtime
activation capture is a separate work package; the writer and solver contracts
are owned by this repository.

`calibrated_svdquant` is not the publishable mixed SVDQuant+GPTQ path; it is the
RTN milestone and correctly reports `gptq_state: not_implemented`. The
`svdquant_gptq_experimental` mode does consume the Hessian manifest and does run
the repo-native GPTQ layer core after smoothing. The direct writer also emits
AWQ W4A16 tensors for Qwen `img_mod.1` / `txt_mod.1` modulation linears when the
dense source checkpoint contains them. These pieces still report
`publishable_svdquant_gptq: false` because the full CLI export still defaults to
weight-residual branch initialization, and the optional output-error branch path
is wired but has not passed target-runtime parity, AWQ dispatch parity, or
external mixed-runtime full inference validation. The detailed gates are in
`int4_svdquant_runtime_contract.md`; external runtime observations are recorded
in `int4_runtime_oracle_notes.md`.

The base library includes a plan-only activation-capture target generator:

```text
calib plan-int4-capture
  dense Qwen-Image-Edit safetensors checkpoint
  + normalized calibration records
  -> capture_plan.json
  -> activation_samples.template.jsonl

calib materialize-int4-capture
  capture_plan.json
  + normalized calibration records
  -> activation_samples.jsonl
  -> capture_materialization_report.json
```

These commands read only local checkpoint metadata and JSONL records. They list
which linear inputs a runtime capture implementation must sample, what stats
keys the calibrated writer will accept, and where each case safetensors file
should be written. They do not run model forward passes and do not import
ComfyUI.

`backends/activation_capture/materialize.py` also exposes
`write_int4_activation_case_safetensors(...)` for runtime capture
implementations that want a checked writer for one case's activation tensors.
The helper validates tensor names and channel counts against `capture_plan.json`
before writing safetensors. It still receives already-captured tensors from the
caller and does not own any model runtime.

The bridge writer remains separate for validation/import workflows:

```text
natural SVDQuant W4A4 safetensors
  -> backends/int4_kitchen_export.py
  -> single kitchen_tile_packed_w4a4 safetensors checkpoint

DeepCompressor PTQ artifact directory
  -> backends/deepcompressor_import.py
  -> natural SVDQuant W4A4 in-memory state dict
  -> backends/int4_kitchen_export.py
  -> single kitchen_tile_packed_w4a4 safetensors checkpoint
```

The bridge command surface is generic by format rather than one command per model:

```bash
comfy_quants export-int4 --format svdquant_w4a4 --source ... --out ...
```

`export-int4` selects input shape with `--source-format`:

```text
natural-safetensors             already-natural SVDQuant W4A4 tensors
deepcompressor-qwen-image-edit  local model.pt/scale.pt/smooth.pt/branch.pt PTQ artifacts
```

The DeepCompressor bridge is a static artifact importer. It reads local `.pt`
state dictionaries, maps Qwen-Image-Edit linear prefixes through
`model_adapters/qwen_image_edit_int4.py`, creates the natural SVDQuant tensor
family, and then calls the reusable tile-pack writer. It must not import or
depend on DeepCompressor, Nunchaku, ComfyUI, or comfy-kitchen at package
runtime.

The format modules do not own Qwen layer selection, QKV splitting, modulation
reordering, file I/O, or ComfyUI runtime calls. Those responsibilities belong
to model adapters and export backends.

Current INT4 boundary:

- implemented: `quantize-int4` direct dense-checkpoint-to-single-file writer
  with calibration-free weight-only initialization;
- implemented: calibrated SVDQuant solver modules for activation stats loading,
  smoothing, groupwise signed INT4 RTN, and low-rank residual SVD;
- implemented: `svdquant_gptq_experimental` checkpoint path that consumes
  activation stats plus GPTQ Hessian artifacts and feeds the repo-native GPTQ
  layer core after smoothing-basis transform and branch subtraction;
- implemented: runtime-independent output-error low-rank branch calibration in
  the GPTQ natural-layout helper and the full `quantize-int4` CLI path when
  `--lowrank-calibration output_error` and `--activation-samples` are supplied;
- implemented: AWQ W4A16 kitchen-native formula quantization for Qwen
  modulation linears in the direct mixed artifact when those dense weights are
  present;
- implemented: calibrated/GPTQ dry-run coverage validation for activation
  stats, Hessian manifests, and optional output-error activation samples,
  including input-channel shape matching before a full quantization run;
- implemented: plan-only Qwen-Image-Edit INT4 activation-capture target
  generation from safetensors weight shapes and normalized records;
- implemented: runtime-independent materialization of reducer sample manifests
  and checked per-case activation safetensors writing;
- implemented: runtime-independent calibration record normalization and
  safetensors activation-dump reduction into `int4_activation_stats.json` and
  `int4_gptq_hessian_stats.json`;
- implemented: SVDQuant W4A4 natural safetensors repack;
- implemented: Qwen-Image-Edit DeepCompressor-style PTQ artifact import for
  SVDQuant attention/MLP linears;
- implemented: optional `.weight.scale.1` handling when it can be represented
  as a single effective group-64 `weight_scale`;
- not yet implemented: runtime calibration activation capture execution from
  real Qwen-Image-Edit forward passes;
- not yet validated: exact AWQ W4A16 dispatch path against the target mixed
  runtime branch;
- not defaulted or claimed publishable: output-error low-rank branch calibration
  for the final quality target;
- not claimed here: external full downstream image inference validation for a
  complete mixed `svdquant_w4a4` + `awq_w4a16` runtime.

## 9. FP8 route

The production route is format-selectable:

```yaml
quant:
  algorithm: fp8_static
  target_dtype: fp8_e5m2   # or fp8_e4m3
  scale:
    granularity: per_tensor
    axis: null
    method: amax
```

Default policy:

- quantize transformer attention/MLP linear weights first;
- keep norm/embed/final layers high precision;
- keep VAE and text/vision paths high precision by default;
- fallback on unsupported/failed modules to bf16 rather than producing a broken artifact.

Dry-run artifact output writes `artifact/quant_tensor_index.json` with one entry per selected weight tensor. For Qwen-Image and Qwen-Image-Edit this currently means 839 FP8 weight tensors for both `fp8_e4m3` and `fp8_e5m2`.

The declared selected-payload layout is:

```text
artifact/
├── quant_tensor_index.json
├── tensors/
│   ├── fp8_weights.safetensors
│   └── bf16_kept.safetensors
└── scales/
    └── fp8_static_scales.safetensors
```

Dry-run jobs write the tensor index and declare the payload paths. Non-dry-run jobs accept a local safetensors source and write the selected FP8 weight payload plus FP32 scales. A source may be one `.safetensors` file, a safetensors index JSON, or a directory containing an index and its local shards:

```text
artifact/
├── quant_tensor_index.json
├── payload_report.json
├── tensors/
│   └── fp8_weights.safetensors
└── scales/
    └── fp8_static_scales.safetensors
```

The writer is intentionally format-oriented rather than Qwen-specific:

```text
static model contract -> quant policy -> quant_tensor_index.json
quant_tensor_index.json + local safetensors source -> payload files
```

That shape lets `fp8_e4m3` and `fp8_e5m2` serve multiple model adapters without adding a format/model matrix to a single large class or file.

The Torch reference backend has isolated generic FP8 tensor helpers:

- solve FP32 scale from tensor amax;
- quantize through `torch.float8_e4m3fn` or `torch.float8_e5m2`;
- dequantize back to FP32 for roundtrip checks;
- support `per_channel/out_features` and `per_tensor` scale modes.

The full inference checkpoint exporter writes a single safetensors checkpoint. For each selected layer it emits:

```text
<layer>.weight        -> torch.float8_e4m3fn or torch.float8_e5m2
<layer>.weight_scale  -> torch.float32 scalar
<layer>.input_scale   -> torch.float32 scalar 1.0
<layer>.comfy_quant   -> uint8 JSON bytes
```

The quant metadata is format-specific:

```json
{"format":"float8_e4m3fn","full_precision_matrix_mult":true}
{"format":"float8_e5m2","full_precision_matrix_mult":true}
```

The local safetensors writers group selected tensors by source shard, open each shard directly, and do not load a model framework. If a selected tensor key is absent or has a shape mismatch, the job fails clearly. High-precision kept-tensor payload writing and full image validation remain separate work packages.

## 9. Hardware baseline

Current production GPU baseline is RTX PRO 6000 Blackwell class 96GB.

Default planning budget:

```yaml
hardware:
  gpu_profile: rtx_pro_6000_blackwell_96gb
  max_vram_gb: 88
  cpu_offload: true
  nvme_offload: true
```

The 88GB default leaves safety room for CUDA context, fragmentation, temporary tensors, activation/stat caches, and validation.

## 10. Current non-goals for the base library

- No ComfyUI custom node inside `/workspace/comfy-quants`.
- No UI/UX surface.
- No ComfyUI import, launch, hidden parser, or embedded checkout.
- No Web UI or UX work.
- No forked canonical Qwen architecture competing with ComfyUI expectations.
- No ComfyUI import/discovery layer.

## 11. Development references

ComfyUI source can be reviewed manually during development to update this repository's static contracts. Such review is a development workflow, not a package feature or dependency.

Relevant ComfyUI areas for Qwen contract review include supported model declarations, model base wiring, Qwen image transformer implementation, and Qwen image text-encoder integration.
