# Architecture

Comfy Quants is an offline quantization and checkpoint export library. It reads
local model weights, runs a selected quantization flow, and writes
ComfyUI-loadable `.safetensors` artifacts.

## Design goals

- produce single-file artifacts for compatible ComfyUI loaders;
- keep CLI runs repeatable and easy to inspect;
- keep model contracts, format contracts, and writers in separate modules;
- add new model families and quantization formats without a central monolithic file;
- let downstream custom-node projects reuse the same export logic.

## How it works with ComfyUI

The normal workflow is artifact-first:

1. run `comfy-quants` outside ComfyUI;
2. write the target `.safetensors` checkpoint;
3. copy or symlink that checkpoint to the model location used by the loader;
4. load the checkpoint in ComfyUI and run inference validation.

ComfyUI custom nodes can depend on this package when they want an in-UI
quantization workflow. The UI code stays in the custom-node project, while the
quantization/export logic stays here.

## External toolchains

Some flows use established quantization or conversion projects. Those tool
paths are provided explicitly in the CLI command, and the final artifact is still
written and inspected by `comfy-quants`.

## Source layout

```text
src/comfy_quants/
├── cli/              # command entrypoints and argument parsing
├── sdk/              # Python API surface
├── core/             # schemas and domain objects
├── model_adapters/   # model-family tensor contracts and layer selection rules
├── algorithms/       # quantization procedures and planners
├── formats/          # reusable storage formats and packing helpers
├── backends/         # safetensors writers, importers, and export pipelines
├── calibration/      # calibration manifests and reducers
├── registry/         # adapters, formats, algorithms, and backend registration
├── validation/       # artifact reports and checks
└── utils/            # JSON, hashing, and system helpers
```

## Extension map

| Area | Add here when... | Keep out of this area |
| --- | --- | --- |
| `model_adapters/` | a model family needs tensor names, shape rules, or layer selection | storage packing code |
| `formats/` | a storage layout needs identifiers, tensor families, metadata, or pack/unpack helpers | model-family policy |
| `algorithms/` | a solver, quantizer, planner, or calibration algorithm is added | UI and workflow integration |
| `backends/` | a file writer, importer, bridge, or end-to-end export pipeline is added | generic tensor math that belongs in `algorithms/` or `formats/` |
| `cli/` | a stable user command is added | format internals or model-specific business logic |

## Adding a model family

1. Add the model contract under `model_adapters/`.
2. Add tests for expected tensor names, shapes, and layer selection.
3. Reuse existing formats when possible.
4. Put only model-specific mapping in the adapter or backend bridge.

## Adding a quantization format

1. Add a format module under `formats/`.
2. Define the format id, tensor family, metadata JSON, shapes, and packing rules.
3. Add writer/reader tests for the format.
4. Keep model-family selection in `model_adapters/`.

## Adding an export flow

1. Add solver, import, or bridge logic under `algorithms/` or `backends/`.
2. Add or extend a CLI command at the boundary.
3. Emit a report that can be checked by tests and CI.
4. Link the user workflow from [`quantization/`](quantization/).
