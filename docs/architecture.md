# Architecture

Comfy Quants is organized around offline quantization and artifact export. The
package contains the model contracts, quantization logic, storage formats, and
writers needed to produce ComfyUI-loadable `.safetensors` checkpoints.

## Design goals

- produce single-file artifacts that compatible ComfyUI loaders can consume;
- keep command-line usage stable for repeatable local quantization jobs;
- keep model contracts, format contracts, and writer logic easy to review;
- make new model families and quantization formats additive rather than monolithic;
- allow downstream custom-node projects to reuse the same export logic.

## Integration model

`comfy_quants` is used before inference: it reads local source weights, applies a
quantization or import flow, writes the target checkpoint, and emits validation
reports. ComfyUI is then used separately to load the produced checkpoint and run
sampling/image validation.

Custom nodes, UI panels, and workflow templates should live in downstream adapter
projects. Those projects can depend on `comfy-quants` and call the CLI or Python
API while keeping their ComfyUI-specific code outside this base library.

## Dependency policy

Normal package operation does not require a ComfyUI checkout. The package keeps
its supported model and artifact contracts in `model_adapters/` and `formats/`, so
exports are reproducible without asking a runtime to parse model structure for it.

Some flows can coordinate external toolchains such as DeepCompressor, Nunchaku,
or comfy-kitchen. Those tools are configured explicitly by path and invoked at the
command boundary; they are not installed as unconditional Python dependencies of
`comfy-quants`.

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

| Code area | Responsibility | Keep separate |
| --- | --- | --- |
| `model_adapters/` | model-family tensor names, shape contracts, layer selection | storage packing algorithms |
| `formats/` | reusable tensor/storage format contracts and packing helpers | model-family policy |
| `algorithms/` | quantization procedures and solver logic | UI and workflow integration |
| `backends/` | file IO, import bridges, export pipelines | runtime-specific UI code |
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
- Package layout and extension rules belong in this page.
