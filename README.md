# Comfy Quants

Comfy Quants is an offline quantization toolkit for building **ComfyUI-loadable
model checkpoints** from local model weights. Pick a quantization flow, run the
export, and get a single-file `.safetensors` artifact that can be placed in a
compatible ComfyUI model directory or loaded by a compatible ComfyUI node.

中文说明见：[中文说明](#中文说明)。

## Quick start

Install from a local checkout:

```bash
git clone https://github.com/Comfy-Org/comfy-quants.git
cd comfy-quants
pip install -e .
```

Export a Qwen-Image-Edit-2511 INT4 tile-pack checkpoint:

```bash
comfy-quants qwen-image-edit-2511-int4 \
  --model /path/to/Qwen-Image-Edit-2511 \
  --base-checkpoint /path/to/qwen_image_edit_2511_bf16_transformer.safetensors \
  --out /path/to/qwen_image_edit_2511_int4_tilepack.safetensors \
  --deepcompressor-root /path/to/DeepCompressor \
  --nunchaku-root /path/to/nunchaku \
  --calibration-samples 128 \
  --search-strength quality-r64 \
  --gpus 0 \
  --hash-output
```

Inspect the exported checkpoint:

```bash
comfy-quants inspect-int4 \
  --artifact /path/to/qwen_image_edit_2511_int4_tilepack.safetensors \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --strict-qwen-image-edit-2511 \
  --json
```

Then copy or symlink the `.safetensors` file into the target ComfyUI model path
and load it with a compatible ComfyUI setup. For other flows, start with
[`docs/README.md`](docs/README.md) or run:

```bash
comfy-quants --help
```

Source-tree equivalent:

```bash
PYTHONPATH=src python -m comfy_quants.cli.main --help
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

## How it fits with ComfyUI

Comfy Quants runs quantization and checkpoint export outside ComfyUI, then writes
artifacts for ComfyUI-compatible loaders. A typical workflow is:

1. prepare the source model and calibration data locally;
2. run one of the `comfy-quants` CLI flows;
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
**ComfyUI 可以加载的模型 checkpoint**。选择一个量化流程，运行导出命令，得到
单文件 `.safetensors` artifact；之后可以把它放到对应的 ComfyUI 模型目录，或由
兼容的 ComfyUI 节点加载。

## 快速开始

从本地源码安装：

```bash
git clone https://github.com/Comfy-Org/comfy-quants.git
cd comfy-quants
pip install -e .
```

导出 Qwen-Image-Edit-2511 INT4 tile-pack checkpoint：

```bash
comfy-quants qwen-image-edit-2511-int4 \
  --model /path/to/Qwen-Image-Edit-2511 \
  --base-checkpoint /path/to/qwen_image_edit_2511_bf16_transformer.safetensors \
  --out /path/to/qwen_image_edit_2511_int4_tilepack.safetensors \
  --deepcompressor-root /path/to/DeepCompressor \
  --nunchaku-root /path/to/nunchaku \
  --calibration-samples 128 \
  --search-strength quality-r64 \
  --gpus 0 \
  --hash-output
```

检查导出的 checkpoint 结构：

```bash
comfy-quants inspect-int4 \
  --artifact /path/to/qwen_image_edit_2511_int4_tilepack.safetensors \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --strict-qwen-image-edit-2511 \
  --json
```

然后将 `.safetensors` 文件复制或软链到目标 ComfyUI 模型目录，并在兼容的
ComfyUI 环境中加载。其他量化流程从 [`docs/README.md`](docs/README.md) 开始，
或查看命令帮助：

```bash
comfy-quants --help
```

源码树运行方式：

```bash
PYTHONPATH=src python -m comfy_quants.cli.main --help
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

## 和 ComfyUI 怎么配合

Comfy Quants 在 ComfyUI 外完成量化和 checkpoint 导出，输出文件面向
ComfyUI-compatible loader。常见流程是：

1. 在本地准备源模型和校准数据；
2. 运行对应的 `comfy-quants` CLI 量化流程；
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
