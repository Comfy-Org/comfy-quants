# Architecture

Comfy Quants is a CLI-first and library-first quantization package for producing
ComfyUI-loadable `.safetensors` artifacts. The package owns artifact contracts and
writers; it does not own a ComfyUI runtime integration.

## Package boundary

`comfy_quants` may:

- read local model checkpoints;
- apply quantization or import already-quantized tensors;
- write single-file `.safetensors` artifacts;
- validate static tensor structure;
- call explicitly configured external tools by subprocess.

`comfy_quants` must not:

- import, vendor, embed, launch, or configure ComfyUI;
- use ComfyUI as a hidden model parser;
- require a ComfyUI checkout for normal package operation;
- declare DeepCompressor, Nunchaku, or comfy-kitchen as Python package dependencies;
- place ComfyUI custom-node or UI code in this base package.

A ComfyUI custom node should live in a separate adapter package and call this package
as a library or CLI.

## Source layout

```text
src/comfy_quants/
├── cli/              # command entrypoints and argument parsing
├── sdk/              # Python API surface
├── core/             # schemas and domain objects
├── model_adapters/   # model-family tensor contracts and selection policies
├── algorithms/       # quantization procedures and planners
├── formats/          # reusable storage formats and packing helpers
├── backends/         # safetensors writers, importers, and export pipelines
├── calibration/      # calibration manifests and reducers
├── registry/         # adapters, formats, algorithms, and backends registry
├── validation/       # artifact reports and checks
└── utils/            # JSON, hashing, and system helpers
```

## Ownership rules

| Code area | Responsibility | Must not own |
| --- | --- | --- |
| `model_adapters/` | model-family tensor names, shape contracts, layer selection | storage packing algorithms |
| `formats/` | reusable tensor/storage format contracts and packing helpers | model-family policy |
| `algorithms/` | quantization procedures and solver logic | UI or external runtime integration |
| `backends/` | file IO, import bridges, export pipelines | hidden dependencies on ComfyUI |
| `cli/` | stable command surface | format-specific business logic that belongs in formats/backends |

## Adding a model family

1. Add the static model contract under `model_adapters/`.
2. Add tests that verify expected tensor names and shapes.
3. Reuse existing formats when possible.
4. Add only model-family-specific mapping in the adapter or backend bridge.

## Adding a quantization format

1. Add a format module under `formats/`.
2. Define the format identifier, tensor family, metadata JSON, shapes, and packing rules.
3. Add writer/reader tests for the format.
4. Keep model selection outside the format module.

## Adding a quantization flow

1. Add solver or planner logic under `algorithms/` or a dedicated backend pipeline.
2. Add a CLI command or extend an existing command only at the boundary.
3. Write reports that can be consumed by tests and CI.
4. Link user-facing instructions from `docs/quantization/`.

## Documentation ownership

- User workflows belong in [`quantization/`](quantization/).
- Command syntax belongs in [`cli.md`](cli.md).
- Tensor layout definitions belong in [`formats/`](formats/).
- Package layout and dependency rules belong in this page.
