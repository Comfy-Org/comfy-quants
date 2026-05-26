# Comfy Quants

Comfy Quants is an offline quantization toolkit for producing **ComfyUI-loadable
model artifacts**. It reads local model weights, runs a selected quantization flow,
and writes single-file `.safetensors` checkpoints that match the target ComfyUI /
comfy-kitchen storage contract.

中文说明见：[中文说明](#中文说明)。

## Install

```bash
pip install -e .
comfy-quants doctor --json
```

Source-tree equivalent:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main doctor --json
```

## Public names

| Surface | Name |
| --- | --- |
| pip distribution | `comfy-quants` |
| CLI command | `comfy-quants` |
| Python import package | `comfy_quants` |
| source directory | `src/comfy_quants/` |

## Choose a flow

| Output | Command | Guide |
| --- | --- | --- |
| FP8 E4M3 / E5M2 Qwen checkpoint | `quantize`, `export-model` | [`docs/quantization/fp8.md`](docs/quantization/fp8.md) |
| Qwen-Image-Edit-2511 INT4 tile-pack | `qwen-image-edit-2511-int4` | [`docs/quantization/qwen_image_edit_2511_int4.md`](docs/quantization/qwen_image_edit_2511_int4.md) |
| Native INT4 solver | `quantize-int4` | [`docs/quantization/native_int4.md`](docs/quantization/native_int4.md) |
| INT4 artifact tools | `inspect-int4`, `export-int4` | [`docs/quantization/int4_tools.md`](docs/quantization/int4_tools.md) |

Documentation index: [`docs/README.md`](docs/README.md).

## Package boundary

`comfy_quants` targets ComfyUI artifacts, not the ComfyUI runtime.

- The package does not import, vendor, start, or configure ComfyUI.
- The package does not use ComfyUI as a hidden parser for model formats.
- The package does not declare DeepCompressor, Nunchaku, or comfy-kitchen as Python dependencies.
- The package stores the required model and format contracts in its own source tree.
- External tools are invoked only through explicit command-line/subprocess boundaries.

## Repository layout

```text
src/comfy_quants/
├── cli/              # command entrypoints
├── sdk/              # Python API surface
├── core/             # schemas and domain objects
├── model_adapters/   # model-family tensor contracts and selection rules
├── algorithms/       # quantization algorithms and planners
├── formats/          # reusable storage formats
├── backends/         # safetensors writers, importers, and export pipelines
├── calibration/      # calibration manifests and reducers
├── registry/         # local registry
├── validation/       # artifact reports and checks
└── utils/            # JSON, hashing, and system helpers
```

Architecture details: [`docs/architecture.md`](docs/architecture.md).

## Test

```bash
python -m pytest tests/unit -q
```

---

## 中文说明

Comfy Quants 是一个离线量化工具库，用于产出 **ComfyUI 可以加载的模型文件**。
它读取本地模型权重，执行指定量化流程，并写出符合 ComfyUI / comfy-kitchen
存储约定的单文件 `.safetensors` checkpoint。

这个仓库是量化核心库，不是 ComfyUI runtime 集成包。ComfyUI custom node、UI、
workflow 集成应放在独立 adapter 仓库中。

## 安装

```bash
pip install -e .
comfy-quants doctor --json
```

源码树运行方式：

```bash
PYTHONPATH=src python -m comfy_quants.cli.main doctor --json
```

## 公开命名

| 类型 | 名称 |
| --- | --- |
| pip 分发包 | `comfy-quants` |
| CLI 命令 | `comfy-quants` |
| Python import 包 | `comfy_quants` |
| 源码目录 | `src/comfy_quants/` |

## 选择量化入口

| 输出 | 命令 | 文档 |
| --- | --- | --- |
| FP8 E4M3 / E5M2 Qwen checkpoint | `quantize`, `export-model` | [`docs/quantization/fp8.md`](docs/quantization/fp8.md) |
| Qwen-Image-Edit-2511 INT4 tile-pack | `qwen-image-edit-2511-int4` | [`docs/quantization/qwen_image_edit_2511_int4.md`](docs/quantization/qwen_image_edit_2511_int4.md) |
| 原生 INT4 solver | `quantize-int4` | [`docs/quantization/native_int4.md`](docs/quantization/native_int4.md) |
| INT4 artifact 工具 | `inspect-int4`, `export-int4` | [`docs/quantization/int4_tools.md`](docs/quantization/int4_tools.md) |

文档索引：[`docs/README.md`](docs/README.md)。

## 包边界

`comfy_quants` 的目标是产出 ComfyUI artifact，不是运行 ComfyUI runtime。

- 本库不导入、不内嵌、不启动、不配置 ComfyUI；
- 本库不把 ComfyUI 当作隐藏的模型格式解析器；
- 本库不声明 DeepCompressor、Nunchaku、comfy-kitchen 为 Python 依赖；
- 本库在自己的源码树中维护需要的模型结构 contract 和格式 contract；
- 外部工具只通过显式命令行 / subprocess 边界调用。

架构说明见：[`docs/architecture.md`](docs/architecture.md)。

## 测试

```bash
python -m pytest tests/unit -q
```
