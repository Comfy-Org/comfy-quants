# Comfy Quants：Qwen-Image / Qwen-Image-Edit 离线量化长期项目任务书（CLI-first / FP8 E4M3 / ComfyUI artifact 版）

> **重写日期**：2026-05-18  
> **项目定位**：为 ComfyUI 生态创建的独立离线量化库；产物必须能被 ComfyUI 使用，但发布包不依赖 ComfyUI。  
> **首期模型范围**：Qwen-Image、Qwen-Image-Edit。  
> **首期量化路线**：FP8 E4M3（`fp8_e4m3`）静态/离线路线。  
> **首期交互方式**：CLI + Python SDK。  
> **明确暂不做**：不做 UI/UX；不接 ComfyUI custom node；不把长量化任务放进 ComfyUI 进程。  
> **生产 GPU 基线**：NVIDIA RTX PRO 6000 Blackwell 96GB class；默认单卡显存安全水位 `max_vram_gb = 88`。  
> **工程原则**：从第一天开始保证可读、可维护、可扩展、可验证、可恢复、可复现。

---

## 0. 关键纠偏

本库的目标不是“执行时连接 ComfyUI”，也不是“内嵌一个 ComfyUI 来解析模型”。本库的目标是：

> **提前研究并固化 ComfyUI 需要的模型/权重/量化 artifact 约定，然后用本库自己的 CLI 与 SDK 生成 ComfyUI 能用的量化产物。**

因此必须明确：

1. 开发阶段可以查看 ComfyUI 上游模型定义，确认 Qwen-Image / Qwen-Image-Edit 的结构与命名约定。
2. 查看结果必须转化为本仓库内的静态 adapter contract、format spec、manifest schema、export rule。
3. 发布包与 CLI 不得 import `comfy.*`。
4. 发布包与 CLI 不得 vendor/copy/嵌套 ComfyUI。
5. 发布包与 CLI 不得启动 ComfyUI server / workflow / UI / custom node。
6. 发布包与 CLI 不得要求用户提供本地 ComfyUI checkout。
7. 本库只需要关心：**导出的模型 artifact 是 ComfyUI 能用的**。

这就是解耦边界：

```text
开发期：参考 ComfyUI 模型格式
        ↓
本库：写入静态 contract + format + exporter
        ↓
执行期：本库独立量化与导出
        ↓
产物：ComfyUI 可加载/可使用
```

---

## 1. 一句话定义

**Comfy Quants** 是一个面向 ComfyUI 生态的、CLI-first 的离线量化库：它优先支持 Qwen-Image / Qwen-Image-Edit，从原始高精度权重出发，依据本仓库固化的 ComfyUI-compatible 静态模型 contract 识别组件与可量化模块，先实现 FP8 E4M3 离线量化路线，并输出带完整 schema、manifest、provenance、scale metadata、hash、兼容性声明和验证报告的量化 artifact。

本项目不是：

- 不是 ComfyUI custom node 首发项目；
- 不是 Web UI 项目；
- 不是推理 loader 项目；
- 不是重新发明一个与 ComfyUI 不兼容的 Qwen 格式；
- 不是通过运行 ComfyUI 来兜底模型格式；
- 不是一开始就覆盖所有模型族与所有量化格式。

---

## 2. P0 范围

| 项目 | P0 决策 |
|---|---|
| 模型族 | Qwen-Image、Qwen-Image-Edit |
| 结构来源 | 开发期参考 ComfyUI，执行期使用本仓库静态 adapter contract |
| 产物目标 | ComfyUI-compatible quantized artifact |
| 量化目标 | FP8 E4M3：`fp8_e4m3` |
| 算法路线 | `fp8_static` 离线静态 scale 路线；先权重量化，再扩展 activation / dynamic path |
| 首期接口 | CLI + Python SDK |
| 生产硬件 | RTX PRO 6000 Blackwell 96GB class；默认 `max_vram_gb = 88` |
| 任务管理 | 本地 job 目录、checkpoint、resume、manifest、report |
| artifact | 内部 `QuantTensor` metadata + 后续真实 safetensors / ComfyUI-compatible export |
| 验证 | smoke + 数值 roundtrip + module-level report，逐步扩展图像/编辑质量评估 |

### P0 不做

- 不做 UI/UX；
- 不做 Web Dashboard；
- 不接入 ComfyUI custom node；
- 不注册 ComfyUI node class；
- 不让 ComfyUI 工作流进程承担长时间离线量化；
- 不做完整 loader；
- 不承诺 TensorRT / torchao / Nunchaku 加速路线；
- 不做 INT4 / NVFP4 / MXFP8 生产路线；
- 不做全模型族覆盖；
- 不做 ComfyUI import/discovery 层。

### P1/P2 后置方向

| 阶段 | 后置能力 |
|---|---|
| P1 | 完整 Qwen 静态 contract 覆盖、真实 FP8 E4M3 tensor 写出、module-level dequant validation |
| P1 | Qwen-Image-Edit 校准输入闭环：prompt + input image + edit instruction + reference_latents |
| P2 | ComfyUI custom node / loader integration，只在 artifact 与验证稳定后做 |
| P2 | torchao / TensorRT QDQ / Nunchaku / DeepCompressor backend/export boundary |
| P2+ | NVFP4、MXFP8、INT4、SVDQuant、SmoothQuant、GPTQ-like 路线 |
| P2+ | FLUX、SD3/SDXL、HiDream、Wan、HunyuanVideo 等更多 ComfyUI 模型族 |

---

## 3. ComfyUI 关系：只对齐产物，不耦合执行环境

### 3.1 原则

只要模型结构已经在 ComfyUI 中存在，本项目必须遵守以下规则：

1. 开发期以 ComfyUI 的模型定义为参照，避免产物与 ComfyUI 预期冲突。
2. 本仓库维护自己的静态 adapter contract，作为执行期 source of truth。
3. 本地 adapter 只做模型族 contract、模块选择、量化策略、校准输入策略、artifact/manifest/report 映射。
4. 上游 ComfyUI 模型结构变化时，通过开发期 review 更新本仓库 contract。
5. artifact manifest 记录本库 contract 版本、模型族、量化格式、算法、hash、兼容性声明。
6. 如果 contract 尚不完整，CLI 可以输出 skeleton/plan，但不得把它声明为真实完整量化产物。

### 3.2 严禁事项

- 不 vendoring / copy ComfyUI 源码到本仓库；
- 不把 ComfyUI 作为隐藏依赖打包进量化工具；
- 不 import 或 call `comfy.*`；
- 不启动 ComfyUI server / workflow engine / UI 来解释模型格式；
- 不通过 custom node 承载离线量化核心逻辑；
- 不让 `core/`、`formats/`、`algorithms/`、`model_adapters/` hard-import 或依赖活的 ComfyUI 进程或源码树；
- 不用 ComfyUI 兜底“未知模型格式”。

### 3.3 Qwen-Image contract 策略

Qwen-Image / Qwen-Image-Edit 的首期 adapter contract 应覆盖：

- family 名称；
- 支持的 model id / revision pin；
- transformer / edit_transformer 的模块命名规则；
- attention / MLP / norm / final / VAE / text_encoder / visual path 的组件划分；
- 哪些模块默认参与 FP8；
- 哪些模块默认保持 bf16；
- Qwen-Image-Edit 的 prompt + input image + edit instruction + reference_latents 校准输入策略；
- 输出 artifact 中需要给 ComfyUI 使用的 metadata。

---

## 4. 架构总览

### 4.1 核心分层

```text
comfy_quants
├── comfy/            # 静态 ComfyUI compatibility policy/contract metadata；不接活的 ComfyUI
├── cli/              # CLI 命令；不含 UI/node；不启动 ComfyUI
├── sdk/              # Python SDK
├── core/             # graph / policy / manifest / config / provenance / errors
├── model_adapters/   # 模型族 adapter contract：Qwen-Image、Qwen-Image-Edit
├── formats/          # 可复用量化格式：fp8_e4m3 等
├── algorithms/       # 量化算法：fp8_static 等
├── backends/         # export 边界：torch_ref、未来 safetensors 等
├── calibration/      # 校准集描述与未来 activation capture
├── jobs/             # job store / checkpoint / resume
├── registry/         # Comfy Kitchen 风格 registry
├── validation/       # 验证报告
└── utils/            # hashing / json / system info
```

### 4.2 解耦规则

```text
ModelAdapter 负责：某个模型族有哪些组件、哪些模块可量化、默认策略是什么
Format       负责：一种量化存储格式如何描述/编码，例如 fp8_e4m3
Algorithm    负责：如何根据 graph + policy 生成 scale / plan / tensor metadata
Backend      负责：如何把内部 artifact 导出到某个外部消费方式
```

不要把这些揉进一个超级类，也不要做超级单体文件。

---

## 5. 一个量化格式对应多个模型时如何扩展

### 5.1 正确方向

`fp8_e4m3` 是 format，不属于 Qwen：

```text
formats/fp8_e4m3.py
  ├── name = fp8_e4m3
  ├── storage_dtype = uint8
  ├── exponent_bits = 4
  ├── mantissa_bits = 3
  ├── scale_required = true
  └── default_scale = per_channel / amax / out_features
```

Qwen adapter 只是选择使用它：

```text
model_adapters/qwen_image.py
  └── default_policy.target_dtype = fp8_e4m3

model_adapters/qwen_image_edit.py
  └── default_policy.target_dtype = fp8_e4m3
```

未来 FLUX / SD3 / Wan 等模型也可以复用同一个 `fp8_e4m3` format，只要它们自己的 adapter 选择该 format 并定义 include/exclude 策略。

### 5.2 新增模型族

新增模型族时应新增：

```text
model_adapters/<family>.py
```

它负责：

- family id；
- supported model ids；
- static contract version；
- graph/module/tensor mapping；
- default quant policy；
- calibration policy；
- compatibility metadata。

不应修改 `fp8_e4m3` 格式本身，除非格式定义真的变化。

### 5.3 新增量化格式

新增量化格式时应新增：

```text
formats/<format>.py
```

它负责：

- storage dtype；
- bit width；
- scale/zero-point 规则；
- packing/endian/subbyte 规则；
- encode/decode reference；
- schema metadata。

不应把某个模型族的模块命名硬写进 format。

### 5.4 新增算法

新增算法时应新增：

```text
algorithms/<algorithm>.py
```

它负责：

- scale 求解；
- 逐层/逐模块执行计划；
- activation capture 策略；
- fallback 策略；
- quant tensor metadata 生成。

算法消费 `ModelGraph + QuantPolicy + QuantFormatSpec`，不应知道 ComfyUI 进程或源码树。

---

## 6. 不允许的架构反模式

### 6.1 超级类

不允许出现：

```text
UniversalComfyQuantizer
MegaQwenComfyModel
AllInOneQuantizationPipeline
```

这类类会把模型族、格式、算法、backend、job、验证全部揉在一起，后续无法维护。

### 6.2 超级单体文件

不允许出现：

```text
quantize_everything.py
qwen_all.py
comfy_adapter_contract.py
```

单个文件超过合理边界时，应拆成：contract、policy、format、algorithm、exporter、validator。

### 6.3 ComfyUI 兜底

不允许用“内嵌/导入 ComfyUI”来解决模型格式问题。模型格式必须在开发阶段研究清楚，然后写成本库的 contract。

---

## 7. CLI 首期命令

### doctor

```bash
PYTHONPATH=src python -m comfy_quants.cli.main doctor --json
```

输出应包含：

- package / Python / GPU 信息；
- registered adapters；
- registered formats；
- registered algorithms；
- registered backends；
- static ComfyUI artifact compatibility policy；
- `artifact_target = comfyui`；
- `contract_mode = static_adapter_contract`。

### inspect

```bash
PYTHONPATH=src python -m comfy_quants.cli.main inspect \
  --model Qwen/Qwen-Image-Edit-2511 \
  --family qwen_image_edit \
  --out runs/qwen-edit-2511/inspect \
  --json
```

输出：

- `model_inspection.json`；
- `model_graph.json`；
- `module_table.csv`；
- `tensor_table.csv`；
- `default_policy.yaml`；
- `memory_estimate.json`；
- `provenance.json`。

### quantize dry-run

```bash
PYTHONPATH=src python -m comfy_quants.cli.main quantize \
  --config configs/qwen_image_edit_2511_fp8_static.yaml \
  --work-dir runs/qwen-edit-2511/fp8-static-v0 \
  --dry-run \
  --json
```

输出：

- config snapshot；
- system info；
- graph；
- policy；
- plan；
- skeleton artifact manifest；
- job state。

---

## 8. FP8 E4M3 P0 量化策略

P0 默认：

- target dtype：`fp8_e4m3`；
- algorithm：`fp8_static`；
- scale granularity：per-channel；
- scale method：amax；
- axis：out_features；
- 默认量化 transformer attention / MLP linear；
- 默认保留 norm / embed / final / VAE / text / vision path 为 bf16；
- 不支持或误差超阈值时 fallback bf16。

示例配置：

```yaml
quant:
  algorithm: fp8_static
  target_dtype: fp8_e4m3
  scale:
    granularity: per_channel
    axis: out_features
    method: amax
```

---

## 9. RTX PRO 6000 生产基线

当前生产显卡配置：

```yaml
hardware:
  gpu_profile: rtx_pro_6000_blackwell_96gb
  max_vram_gb: 88
  cpu_offload: true
  nvme_offload: true
```

88GB 默认水位用于预留：

- CUDA context；
- allocator fragmentation；
- 临时 tensor；
- scale/stat cache；
- activation capture；
- validation batch；
- checkpoint flush。

---

## 10. 里程碑

### M0：架构骨架与任务书完成

目标：

- 包名与目录为 `comfy_quants`；
- 保留 `genquant` legacy shim；
- 建立 Comfy Kitchen 风格 registry；
- registry 覆盖 adapters / formats / algorithms / backends；
- Qwen adapters 输出静态 contract metadata；
- FP8 E4M3 format spec 成为唯一 P0 主格式；
- CLI dry-run 可跑通；
- 单元测试通过；
- 代码与测试中不出现 ComfyUI 执行耦合。

验收：

```bash
PYTHONPATH=src python -m comfy_quants.cli.main doctor --json
PYTHONPATH=src python -m comfy_quants.cli.main inspect --model Qwen/Qwen-Image-Edit-2511 --family qwen_image_edit --out /tmp/inspect --json
PYTHONPATH=src python -m comfy_quants.cli.main quantize --config configs/qwen_image_edit_2511_fp8_static.yaml --work-dir /tmp/qwen-edit-fp8 --dry-run --json
PYTHONPATH=src python -m unittest discover -s tests -v
```

### M1：完整静态 Qwen contract

目标：

- 将 Qwen-Image / Qwen-Image-Edit 的真实 module/tensor table 固化为本库 contract；
- inspection 不再依赖临时 symbolic projection；
- contract 版本化；
- contract diff 可测试；
- manifest 记录本库 contract version 与模型 revision。

验收：

- `contract_source = comfy_quants`；
- `module_table.csv` 与 ComfyUI-compatible Qwen artifact 预期一致；
- 没有本地 forked architecture 与 ComfyUI 预期冲突；
- contract 升级有 diff 与兼容性警告。

### M2：真实 FP8 E4M3 权重量化

目标：

- FP8 E4M3 encode/decode reference；
- per-channel scale 求解；
- layer-wise quant runner；
- safetensors payload 写出；
- QuantTensor metadata 完整；
- fallback bf16；
- resume/checkpoint。

### M3：验证闭环

目标：

- dequant roundtrip；
- module-level error report；
- end-to-end smoke；
- Qwen-Image text-to-image 质量 smoke；
- Qwen-Image-Edit 编辑质量 smoke；
- validation report 阈值化。

### M4：ComfyUI 集成入口

只有在 artifact 与验证稳定后，才考虑 ComfyUI custom node / loader / workflow 集成。该阶段也必须保持：核心量化逻辑仍在本库，不进入 UI/node 单体。

---

## 11. 风险与决策

### 11.1 上游模型预期变化

风险：ComfyUI 对 Qwen artifact 或模块命名预期变化。

决策：

- 开发期 review 后更新本库 contract；
- contract version 必须进入 manifest；
- inspection diff 必须进入测试；
- 不以导入 ComfyUI 作为兜底。

### 11.2 FP8 质量风险

风险：某些层 FP8 E4M3 质量损失明显。

决策：

- 默认 exclude norm/embed/final/VAE/text/vision；
- module-level error report；
- 层级 fallback bf16；
- 不以全模型强制 FP8 为目标。

### 11.3 显存/OOM 风险

风险：20B 级模型量化过程中激活、统计、临时张量导致 OOM。

决策：

- 默认 88GB 显存水位；
- layer-wise；
- CPU/NVMe offload；
- checkpoint/resume；
- 每层释放资源；
- job event log。

### 11.4 过早 UI 风险

风险：过早做 node/UI 会把未稳定的长任务、artifact 格式和错误恢复逻辑固化。

决策：

- M0/M1/M2 禁止 UI/node；
- ComfyUI node 至少等 artifact/validation/loader API 稳定后再做；
- node 未来只做调用入口，不做核心量化逻辑。

---

## 12. 当前 M0 已落地状态

截至本次重写，仓库应具备或正在收敛到：

- `src/comfy_quants/` 包；
- `src/genquant/` legacy shim；
- static ComfyUI artifact contract index；
- `qwen_image` / `qwen_image_edit` adapter；
- central registry；
- format registry；
- `fp8_e4m3` format spec；
- `fp8_static` skeleton algorithm；
- `torch_ref` backend skeleton；
- CLI：doctor / inspect / calib build / quantize / validate / export / jobs / resume；
- FP8 E4M3 config；
- INT4 future placeholder；
- tests。

注意：当前仍是 M0 skeleton。尚未实现：

- 完整 Qwen static contract；
- 真实 FP8 E4M3 tensor payload；
- 真实 safetensors 写出；
- 真实激活采集；
- 真实图像质量验证；
- ComfyUI node。

---

## 13. 推荐下一步开发顺序

1. **M1.1：Qwen static contract 固化**  
   将 Qwen-Image / Qwen-Image-Edit module/tensor/component 命名规则写入本库 contract，替代 M0 symbolic projection。

2. **M1.2：contract diff 与兼容性检查**  
   contract version、模型 revision、module table hash、tensor table hash 进入 manifest。

3. **M1.3：format registry 接入算法计划**  
   `fp8_static` 明确消费 `QuantFormatSpec(fp8_e4m3)`，不把格式细节散落在 Qwen adapter。

4. **M2.1：FP8 E4M3 codec**  
   实现 encode/decode reference、scale metadata、roundtrip test。

5. **M2.2：layer-wise quantization runner**  
   实现逐层加载、统计、量化、写出、释放、checkpoint。

6. **M2.3：safetensors / ComfyUI-compatible export**  
   生成 ComfyUI 能使用的权重与 metadata 组合。

7. **M3：验证闭环**  
   roundtrip、module error、图像 smoke、编辑 smoke、报告阈值。

---

## 14. 最终验收原则

项目长期成功的标准不是“本库里能跑一个 ComfyUI”，而是：

1. 本库独立完成量化；
2. 产物能被 ComfyUI 使用；
3. adapter / format / algorithm / backend 清晰解耦；
4. 一个 format 可复用于多个模型；
5. 没有超级类、超级单体文件、隐藏依赖；
6. 任务可恢复、结果可验证、artifact 可复现。
