# Comfy Quants

Comfy Quants is an offline quantization toolkit for building
**ComfyUI-loadable model checkpoints** from local model weights. Start by
choosing the quantization format you want to produce, then open that format guide
for supported model-family commands and examples.

中文说明见：[中文说明](#中文说明)。

## Quick start

Install from a local checkout:

```bash
git clone https://github.com/Comfy-Org/comfy-quants.git
cd comfy-quants
pip install -e .
```

Check the CLI:

```bash
comfy-quants --help
```

Typical workflow:

1. choose an output format from [Quantization formats](#quantization-formats);
2. open the linked format guide;
3. pick the model-family export command listed in that guide;
4. run the export and inspect the produced `.safetensors` artifact;
5. copy or symlink the artifact into the target ComfyUI model path and load it
   with a compatible loader.

Source-tree equivalent:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main --help
```

## Quantization formats

| Output format | Use when you want... | Workflow guide | Format reference |
| --- | --- | --- | --- |
| FP8 E4M3 / E5M2 checkpoint | a full FP8 transformer checkpoint | [`docs/quantization/fp8.md`](docs/quantization/fp8.md) | [`docs/formats/fp8.md`](docs/formats/fp8.md) |
| INT4 SVDQuant W4A4 tile-pack | a tile-packed INT4 checkpoint | [`docs/quantization/int4.md`](docs/quantization/int4.md) | [`docs/formats/svdquant_w4a4_kitchen_tilepack.md`](docs/formats/svdquant_w4a4_kitchen_tilepack.md) |
| INT4 AWQ W4A16 tensors | mixed INT4 modulation tensors inside supported bundles | [`docs/quantization/int4.md`](docs/quantization/int4.md) | [`docs/formats/awq_w4a16.md`](docs/formats/awq_w4a16.md) |

Each format guide links to the model-family flows currently implemented for that
format. CLI syntax is summarized in [`docs/cli.md`](docs/cli.md), and the full
documentation index is [`docs/README.md`](docs/README.md).

## Public names

| Surface | Name |
| --- | --- |
| pip distribution | `comfy-quants` |
| CLI command | `comfy-quants` |
| Python import package | `comfy_quants` |
| source directory | `src/comfy_quants/` |

## How it fits with ComfyUI

Comfy Quants runs quantization and checkpoint export outside ComfyUI, then writes
artifacts for ComfyUI-compatible loaders. A typical workflow is:

1. prepare the source model and calibration data locally;
2. run the selected `comfy-quants` export flow;
3. copy or symlink the produced `.safetensors` file into the target ComfyUI model path;
4. load the checkpoint in ComfyUI for sampling and image validation.

If you want in-ComfyUI quantization nodes, build them as a separate custom-node
project and call this package through its CLI or Python API. This keeps the export
library reusable while still allowing downstream UI/workflow integrations.

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

Comfy Quants 是一个离线量化工具库，用于把本地模型权重量化并导出为
**ComfyUI 可以加载的模型 checkpoint**。使用时先选择要产出的量化格式，再进入
对应格式文档查看已经支持的模型家族命令和示例。

## 快速开始

从本地源码安装：

```bash
git clone https://github.com/Comfy-Org/comfy-quants.git
cd comfy-quants
pip install -e .
```

查看 CLI：

```bash
comfy-quants --help
```

常见流程：

1. 从 [量化格式](#量化格式) 中选择输出格式；
2. 打开对应格式文档；
3. 在格式文档中选择已经支持的模型家族导出命令；
4. 运行导出，并检查产出的 `.safetensors` artifact；
5. 将 artifact 复制或软链到目标 ComfyUI 模型目录，并使用兼容 loader 加载。

源码树运行方式：

```bash
PYTHONPATH=src python -m comfy_quants.cli.main --help
```

## 量化格式

| 输出格式 | 适用场景 | 流程文档 | 格式定义 |
| --- | --- | --- | --- |
| FP8 E4M3 / E5M2 checkpoint | 导出完整 FP8 transformer checkpoint | [`docs/quantization/fp8.md`](docs/quantization/fp8.md) | [`docs/formats/fp8.md`](docs/formats/fp8.md) |
| INT4 SVDQuant W4A4 tile-pack | 导出 tile-packed INT4 checkpoint | [`docs/quantization/int4.md`](docs/quantization/int4.md) | [`docs/formats/svdquant_w4a4_kitchen_tilepack.md`](docs/formats/svdquant_w4a4_kitchen_tilepack.md) |
| INT4 AWQ W4A16 tensors | supported bundle 中的混合 INT4 modulation tensor | [`docs/quantization/int4.md`](docs/quantization/int4.md) | [`docs/formats/awq_w4a16.md`](docs/formats/awq_w4a16.md) |

每个格式文档会继续链接到该格式当前支持的模型家族流程。CLI 语法见
[`docs/cli.md`](docs/cli.md)，完整文档索引见 [`docs/README.md`](docs/README.md)。

## 公开命名

| 类型 | 名称 |
| --- | --- |
| pip 分发包 | `comfy-quants` |
| CLI 命令 | `comfy-quants` |
| Python import 包 | `comfy_quants` |
| 源码目录 | `src/comfy_quants/` |

## 和 ComfyUI 怎么配合

Comfy Quants 在 ComfyUI 外完成量化和 checkpoint 导出，输出文件面向
ComfyUI-compatible loader。常见流程是：

1. 在本地准备源模型和校准数据；
2. 运行选定的 `comfy-quants` 导出流程；
3. 将产出的 `.safetensors` 文件复制或软链到目标 ComfyUI 模型目录；
4. 在 ComfyUI 中加载 checkpoint，进行采样和出图验证。

如果需要在 ComfyUI 里通过节点执行量化，可以在独立 custom-node 项目中依赖
本包，并调用本包的 CLI 或 Python API。这样量化导出能力可以复用，UI / workflow
集成也可以独立迭代。

## 仓库结构

```text
src/comfy_quants/
├── cli/              # 命令入口
├── sdk/              # Python API
├── core/             # schema 和领域对象
├── model_adapters/   # 模型族 tensor contract 与层选择规则
├── algorithms/       # 量化算法与 planner
├── formats/          # 可复用存储格式
├── backends/         # safetensors writer、importer、export pipeline
├── calibration/      # 校准 manifest 与统计 reducer
├── registry/         # 本地 registry
├── validation/       # artifact 检查与报告
└── utils/            # JSON、hash、系统工具
```

架构说明见：[`docs/architecture.md`](docs/architecture.md)。

## 测试

```bash
python -m pytest tests/unit -q
```
