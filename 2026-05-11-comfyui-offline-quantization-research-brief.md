# ComfyUI 主流生成模型离线量化库研究任务书

> **文档性质**：这是一份交给研究型 AI agent / 外部技术团队的自包含研究任务书。阅读者无需访问本地项目文件；所有必要背景、目标、限制、候选技术路线、用卡规划和期望输出格式均已写入本文档。
>
> **重要纠偏**：本项目不是“让 ComfyUI 加载并推理量化模型”的 runtime loader 项目，而是“开发一个离线量化库 / 量化生产工具链”，用于从原始 bf16 / fp16 / fp32 权重出发，生成可复现、可验证、可导出的量化模型。
>
> **项目状态**：项目尚处于规划阶段，以下架构为计划方案，不代表已有实现；当前本地没有相关项目、没有已有代码库、没有历史实验数据。因此本文档不会声称已经跑通过任何实验，也不会虚构已有指标。
>
> **生成日期**：2026-05-11  
> **项目阶段**：规划 / 尚未开工  
> **核心研究主题**：面向 ComfyUI 生态主流生成模型的离线量化库、混合精度量化策略、量化 job 系统与 ComfyUI 前端插件接口。

---

## 第一部分：项目全景

### 1.1 项目目标

我们要开发一个**面向 ComfyUI 主流生成模型生态的离线量化库**。它的职责是：

1. 从原始 `bf16` / `fp16` / `fp32` 权重文件出发；
2. 自动识别模型家族、模块结构、层类型、敏感层、可量化层和应保留高精度的层；
3. 执行校准数据构建、激活采集、量化参数求解、权重量化、必要的激活量化参数生成；
4. 输出带完整 provenance、schema、scale、zero point、block scale、algorithm metadata、calibration metadata、hash 与验证报告的量化模型 artifact；
5. 支持恢复 / 断点续跑 / 分层缓存 / CPU 与 NVMe offload；
6. 为 ComfyUI 提供成熟的 API 与 custom node 前端，但 ComfyUI 只负责配置、启动、监控、查看报告和注册产物，不承担重型量化计算本身。

一句话定义：

> **这是一个“制造量化模型”的离线生产库，不是一个“加载量化模型推理”的运行时插件。**

初始模型覆盖范围应面向 ComfyUI 常见生成模型，而不是只做 LLM：

| 类别 | 模型族 / 组件 | 量化适配重点 |
|---|---|---|
| 传统扩散图像模型 | SD1.5、SD2.x、SDXL | UNet、CLIP text encoder、VAE、LoRA / ControlNet 兼容性 |
| DiT / MMDiT 图像模型 | SD3 / SD3.5、FLUX.1、Qwen-Image、HiDream、HunyuanDiT | transformer block、attention、MLP、norm、positional / timestep embedding、dual text encoder |
| 视频模型 | Wan、HunyuanVideo、CogVideoX、LTX-Video | video DiT、3D VAE、temporal attention、text / vision encoder、帧间一致性 |
| 附加生态 | LoRA、ControlNet、IP-Adapter、model patch、accelerated / distilled variants | base model 量化后 adapter 是否无需重算、是否需要 adapter-aware calibration |
| 文本编码器 | CLIP、T5、Qwen-VL / LLaVA 类编码器 | 可复用 LLM PTQ 方法，但要考虑 prompt 分布、长文本、中文 / 英文 / 多语文本渲染 |
| VAE / 3D VAE | image VAE、video 3D VAE | 默认倾向保留 bf16 / fp16；研究低风险量化边界 |

目标量化格式与算法至少包括：

| 方向 | 候选格式 / 算法 | 初始定位 |
|---|---|---|
| FP8 | FP8 E4M3、FP8 E5M2、`e4m3fn`、`e4m3fnuz`、dynamic / static scale | 首个相对现实的低风险路线，适合权重 / 部分激活 |
| NVIDIA FP4 | NVFP4、FP4 E2M1、E4M3 scale、global scale | 面向 Blackwell 原生或近原生验证，重点研究 block scale 与格式导出 |
| OCP Microscaling | MXFP8、MXFP4、MXFP6、E8M0 scale | 面向 OCP / Blackwell / AMD CDNA4 兼容路径 |
| Integer PTQ | INT8、INT4、W8A8、W4A16、W4A8、per-channel、per-group | 基础量化能力，便于 GPTQ / AWQ / SmoothQuant / export |
| 二阶 / 激活感知 | GPTQ、AWQ、SmoothQuant | 需要从 LLM 迁移到 DiT / diffusion / video 场景 |
| 低秩混合 | SVDQuant、low-rank branch、residual branch | 生成模型 4-bit 的重点研究方向之一 |
| 混合精度搜索 | layer sensitivity、fallback、bit allocation、format allocation | 产品级质量关键：不是全模型一个 dtype，而是 policy-driven artifact |

### 1.2 计划技术架构详解

#### 1.2.1 总体分层

建议把系统分成五层：

```text
┌──────────────────────────────────────────────────────────────┐
│ ComfyUI custom nodes / Web UI / API routes                    │
│ - 参数配置、任务启动、状态监控、报告查看、产物注册             │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ Python SDK + CLI + optional daemon                            │
│ - inspect / calibrate / quantize / validate / export / resume │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ Offline Quantization Engine                                   │
│ - adapters / calibration / activation capture / algorithms    │
│ - memory planner / offload / checkpoint / manifest            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ Format and Backend Layer                                      │
│ - safetensors_quant / ONNX QDQ / TensorRT ModelOpt / torchao  │
│ - Nunchaku / MX / NVFP4 / FP8 / INT packing metadata          │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ Validation and Benchmark Layer                                │
│ - smoke generation / numerical compare / visual metrics       │
│ - artifact diff / compatibility check / regression report     │
└──────────────────────────────────────────────────────────────┘
```

#### 1.2.2 建议目录结构

```text
comfy_quant/
├── __init__.py
├── api.py
├── cli.py
├── config.py
├── errors.py
├── model_adapters/
│   ├── base.py
│   ├── sd15.py
│   ├── sdxl.py
│   ├── sd3.py
│   ├── flux.py
│   ├── qwen_image.py
│   ├── hidream.py
│   ├── hunyuan_dit.py
│   ├── wan.py
│   ├── hunyuan_video.py
│   ├── cogvideox.py
│   ├── ltx_video.py
│   └── text_encoders.py
├── calibration/
│   ├── prompt_sets.py
│   ├── image_sets.py
│   ├── video_sets.py
│   ├── latent_sampler.py
│   ├── timestep_sampler.py
│   ├── scheduler_sampler.py
│   ├── cfg_sampler.py
│   ├── activation_capture.py
│   └── cache.py
├── algorithms/
│   ├── base.py
│   ├── fp8_static.py
│   ├── fp8_dynamic.py
│   ├── nvfp4.py
│   ├── mxfp8.py
│   ├── mxfp4.py
│   ├── int8.py
│   ├── int4.py
│   ├── gptq.py
│   ├── awq.py
│   ├── smoothquant.py
│   ├── svdquant.py
│   ├── rotation.py
│   ├── sensitivity.py
│   └── mixed_precision_search.py
├── formats/
│   ├── safetensors_quant.py
│   ├── gguf_bridge.py
│   ├── onnx_qdq.py
│   ├── tensorrt_modelopt.py
│   ├── torchao_bridge.py
│   ├── nunchaku.py
│   ├── manifest.py
│   ├── packing_fp8.py
│   ├── packing_fp4.py
│   ├── packing_int.py
│   ├── packing_mx.py
│   └── metadata_schema.py
├── runtime_validation/
│   ├── numerical.py
│   ├── smoke_comfy.py
│   ├── image_metrics.py
│   ├── video_metrics.py
│   └── report.py
├── benchmark/
│   ├── suites.py
│   ├── qwen_image.py
│   ├── flux.py
│   ├── video.py
│   └── regression.py
├── job/
│   ├── scheduler.py
│   ├── manifest.py
│   ├── checkpoint.py
│   ├── memory_planner.py
│   ├── offload.py
│   ├── logs.py
│   └── daemon.py
└── utils/
    ├── hashing.py
    ├── safetensors_io.py
    ├── device.py
    ├── dtype.py
    ├── graph.py
    └── provenance.py
```

#### 1.2.3 核心 API 设计

量化库应首先是 Python SDK，其次才是 CLI 和 ComfyUI 节点。

建议暴露的稳定 API：

```python
from comfy_quant import (
    inspect_model,
    build_calibration_set,
    quantize_model,
    resume_quant_job,
    validate_quantized_model,
    export_quantized_model,
    register_for_comfyui,
)
from comfy_quant.config import QuantConfig, CalibrationConfig, ExportConfig

model_info = inspect_model(
    model_path="models/diffusion_models/qwen_image_bf16.safetensors",
    model_family="qwen_image",
)

quant_config = QuantConfig(
    algorithm="gptq",
    target_format="int4",
    group_size=128,
    act_order=True,
    damp_percent=0.01,
    mixed_precision={
        "keep": ["final_layer", "norm", "embed", "vae"],
        "prefer_fp8": ["attention.q", "attention.k", "attention.v"],
        "prefer_int4": ["mlp.fc1", "mlp.fc2"],
        "sensitive_fallback": "bf16",
    },
    vram_budget_gb=80,
    cpu_offload=True,
    nvme_offload_dir="D:/quant_cache/qwen_image_gptq",
)

calib_config = CalibrationConfig(
    prompt_set="qwen_image_mixed_cn_en_text_rendering",
    num_samples=256,
    resolutions=[768, 1024, 1328],
    timestep_strategy="stratified",
    scheduler="default_model_scheduler",
    seed=1234,
    capture_activations=True,
)

job = quantize_model(
    model_path="models/diffusion_models/qwen_image_bf16.safetensors",
    model_family="qwen_image",
    quant_config=quant_config,
    calibration=calib_config,
    output_dir="outputs/qwen_image_gptq_int4",
    resume=True,
)

report = validate_quantized_model(
    quantized_model_dir="outputs/qwen_image_gptq_int4",
    validation_suite="qwen_image_quality_basic",
    run_smoke=True,
    output_dir="reports/qwen_image_gptq_int4",
)
```

建议 CLI：

```bash
cq inspect \
  --model qwen_image_bf16.safetensors \
  --family qwen_image \
  --out reports/qwen_image_inspect.json

cq calibrate \
  --model qwen_image_bf16.safetensors \
  --family qwen_image \
  --prompt-set qwen_image_cn_en_text_rendering \
  --samples 256 \
  --resolutions 768,1024,1328 \
  --timesteps stratified \
  --cache D:\quant_cache\qwen_image_calib

cq quantize \
  --model qwen_image_bf16.safetensors \
  --family qwen_image \
  --algorithm gptq \
  --target int4 \
  --group-size 128 \
  --vram-budget 80GiB \
  --cpu-offload \
  --nvme-offload D:\quant_cache\qwen_image_gptq \
  --resume \
  --out outputs/qwen_image_gptq_int4

cq validate \
  --model outputs/qwen_image_gptq_int4 \
  --suite qwen_image_quality_basic \
  --run-smoke \
  --out reports/qwen_image_gptq_int4_report.html
```

#### 1.2.4 ComfyUI 插件的边界

ComfyUI custom nodes 只作为**离线量化前端**，不应把几十小时量化任务塞进普通同步节点执行中。建议模型：

1. ComfyUI 节点收集参数；
2. 节点调用本地 daemon / job scheduler；
3. daemon 在独立进程中执行量化；
4. 节点返回 job id；
5. 监控节点查询 job 状态；
6. 报告节点读取产物 report；
7. 注册节点把量化产物写入 ComfyUI 可识别的模型目录或 registry metadata。

建议节点列表：

| 节点名 | 返回类型 | 作用 |
|---|---|---|
| `Quant Inspect Model` | `MODEL_INFO` | 解析模型结构、组件、层类型、参数量、dtype、建议量化边界 |
| `Quant Build Calibration Set` | `CALIBRATION_SET` | 创建 prompt / image / video / latent / timestep 校准配置 |
| `Quant Policy Builder` | `QUANT_POLICY` | 选择 FP8 / INT4 / GPTQ / AWQ / SVDQuant 等基础 policy |
| `Advanced Mixed Precision Policy` | `QUANT_POLICY` | 配置敏感层保留、fallback、per-layer overrides |
| `Start Offline Quantization Job` | `QUANT_JOB`, `STRING` | 启动后台离线量化任务，返回 job 和 job_id |
| `Quant Job Monitor` | `QUANT_JOB_STATUS` | 查询进度、当前层、显存、ETA、错误 |
| `Quant Resume Job` | `QUANT_JOB` | 从 manifest / checkpoint 恢复 |
| `Quant Report Viewer` | `QUANT_REPORT` | 显示质量、误差、性能、失败层、fallback 层 |
| `Quant Export Model` | `QUANT_ARTIFACT` | 导出 safetensors_quant / ONNX QDQ / TensorRT / Nunchaku 等 |
| `Register Quantized Model` | `STRING` | 注册产物给 ComfyUI 模型目录 / 元数据 |
| `Quant Compatibility Checker` | `COMPAT_REPORT` | 检查当前 ComfyUI、CUDA、ROCm、torch、kernel 后端是否支持目标 artifact |

代表性 custom node skeleton：

```python
class StartOfflineQuantizationJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_model_path": ("STRING", {"default": ""}),
                "model_family": ([
                    "sd15",
                    "sdxl",
                    "sd3",
                    "flux",
                    "qwen_image",
                    "hidream",
                    "hunyuan_dit",
                    "wan",
                    "hunyuan_video",
                    "cogvideox",
                    "ltx_video",
                ],),
                "quant_policy": ("QUANT_POLICY", {"forceInput": True}),
                "calibration_set": ("CALIBRATION_SET", {"forceInput": True}),
                "output_dir": ("STRING", {"default": "outputs/quantized"}),
            },
            "optional": {
                "vram_budget_gb": ("INT", {"default": 80, "min": 1, "max": 512}),
                "cpu_offload": ("BOOLEAN", {"default": True}),
                "nvme_offload_dir": ("STRING", {"default": ""}),
                "resume": ("BOOLEAN", {"default": True}),
                "fail_policy": (["stop", "fallback_to_bf16", "skip_layer"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("QUANT_JOB", "STRING")
    RETURN_NAMES = ("job", "job_id")
    FUNCTION = "start"
    CATEGORY = "quantization/offline"

    def start(
        self,
        source_model_path,
        model_family,
        quant_policy,
        calibration_set,
        output_dir,
        vram_budget_gb=80,
        cpu_offload=True,
        nvme_offload_dir="",
        resume=True,
        fail_policy="fallback_to_bf16",
        unique_id=None,
        prompt=None,
    ):
        job = daemon_client.submit_quant_job(
            source_model_path=source_model_path,
            model_family=model_family,
            quant_policy=quant_policy,
            calibration_set=calibration_set,
            output_dir=output_dir,
            vram_budget_gb=vram_budget_gb,
            cpu_offload=cpu_offload,
            nvme_offload_dir=nvme_offload_dir,
            resume=resume,
            fail_policy=fail_policy,
            comfy_node_id=unique_id,
            comfy_prompt=prompt,
        )
        return (job, job.job_id)
```

#### 1.2.5 Job manifest 与断点续跑

每个离线量化任务必须生成 job manifest，保证可复现、可恢复、可审计。

示例：

```json
{
  "schema_version": "cq.job.v1",
  "job_id": "qwen_image_gptq_int4_20260511_001",
  "model_family": "qwen_image",
  "source_model": "qwen_image_bf16.safetensors",
  "source_sha256": "SOURCE_SHA256_EXAMPLE",
  "algorithm": "gptq",
  "target_format": "int4",
  "group_size": 128,
  "calibration": {
    "samples": 256,
    "prompt_set": "qwen_cn_en_text_rendering",
    "resolution": [1024],
    "timestep_strategy": "stratified",
    "scheduler": "default_model_scheduler",
    "seed": 1234
  },
  "device": {
    "gpu": "H100 80GB",
    "vram_budget_gb": 80,
    "cpu_offload": true,
    "nvme_offload": true
  },
  "status": "running",
  "current_layer": "dit.blocks.31.mlp.fc2",
  "completed_layers": [],
  "failed_layers": [],
  "fallback_layers": [],
  "created_at": "2026-05-11T00:00:00Z"
}
```

建议 checkpoint layout：

```text
outputs/qwen_image_gptq_int4/
├── manifest.json
├── source_hash.txt
├── calibration/
│   ├── prompts.jsonl
│   ├── timesteps.json
│   ├── activations.index.json
│   └── stats/
├── checkpoints/
│   ├── layer_000_attn_q.done.json
│   ├── layer_000_attn_q.quant.safetensors
│   ├── layer_000_attn_k.done.json
│   ├── layer_000_attn_k.quant.safetensors
│   ├── layer_001_mlp_fc1.done.json
│   └── ...
├── artifacts/
│   ├── model.quant.safetensors
│   ├── scales.safetensors
│   ├── quant_metadata.json
│   └── compatibility.json
└── reports/
    ├── report.html
    ├── report.json
    ├── layer_error.csv
    ├── fallback_layers.json
    └── sample_outputs/
```

#### 1.2.6 为什么必须自己跑量化测试

虽然最终项目不是 runtime loader，但**离线量化库的验收不能只看代码是否生成文件**。必须自己跑量化任务和验证任务，原因如下：

1. **量化本身就是目标产物**：如果 GPTQ / SVDQuant 运行 30 小时后在某层崩溃，或者产物 scale metadata 与权重 packing 对不上，这就是库的核心 bug。
2. **大模型显存与时间是产品约束**：Qwen-Image 20B 级别模型做 GPTQ / SVDQuant，规划上应按约 70GB+ VRAM、几十小时级别任务预估；这类约束会反向决定 job 系统、checkpoint 粒度、offload、日志和 UI 设计。
3. **生成模型的校准不是 LLM 校准的直接复制**：diffusion / DiT 的 timestep、noise level、scheduler、CFG、resolution、prompt distribution、video temporal span 都会影响激活分布。
4. **混合精度策略需要真实敏感度数据**：哪些层可 INT4、哪些层要 FP8、哪些层必须 bf16，不能完全靠论文经验。
5. **格式导出必须验证兼容性**：safetensors metadata、ONNX Q/DQ、TensorRT explicit quantization、torchao shell dtype、Nunchaku / SVDQuant artifact 都可能有细节坑。

因此建议采用分级验证：

- 本地低显存卡：做 tensor packing、toy model、small model、ComfyUI 节点、job resume。
- 中高端卡：做 SDXL / FLUX partial / Qwen-Image 单层或单 block。
- 大显存云卡：做 Qwen-Image 20B full GPTQ / SVDQuant / FP8 release gate。
- Blackwell / MI350：做 NVFP4 / MXFP4 / MXFP8 / E2M1 原生或近原生路径验证。

### 1.3 已发生的目标纠偏与约束

由于项目尚未开工，没有真实“走过的弯路”和历史实验。但目前已经发生两次关键目标纠偏，必须写进研究任务书，避免外部研究 AI 重复理解错误。

#### 纠偏 1：不是现有项目改造

用户明确说明：

> 本地没有相关项目。还未开始。

因此：

- 不存在已有代码结构；
- 不存在已有 benchmark；
- 不存在已跑通的模型或失败记录；
- 研究 AI 不应假设已有 ComfyUI 插件、已有量化库、已有模型 adapter；
- 所有架构均应视为 planning proposal。

#### 纠偏 2：不是推理加载插件，而是离线量化库

用户明确说明：

> 该任务不是要加载并推理量化模型，而是开发一个离线量化库来对模型进行量化。

因此研究重点必须从：

```text
错误重点：ComfyUI 如何加载 quantized model 并推理
```

转为：

```text
正确重点：如何把原始模型离线量化成稳定、可验证、可导出的 quantized artifact
```

ComfyUI 的角色是：

- 可视化参数配置；
- 校准集选择；
- job 启动；
- 进度监控；
- 报告查看；
- 产物注册；
- 兼容性提示。

ComfyUI 不应成为重型量化计算的唯一执行环境。

### 1.4 产品使用流程

建议的产品流程如下：

```text
用户选择原始模型
  ↓
Quant Inspect Model 节点分析模型族、层结构、参数量、dtype、组件
  ↓
用户选择或生成 calibration set
  ↓
用户选择 quant policy：FP8 / INT8 / INT4 / GPTQ / SVDQuant / mixed
  ↓
Advanced Mixed Precision Policy 配置敏感层保留、fallback、bit allocation
  ↓
Start Offline Quantization Job 提交后台任务
  ↓
daemon 执行 inspect → calibration → activation capture → quantize → checkpoint
  ↓
Quant Job Monitor 显示进度、显存、当前层、ETA、失败层
  ↓
validate 生成 smoke samples、数值误差、视觉指标、兼容性报告
  ↓
Quant Export Model 选择 artifact 格式
  ↓
Register Quantized Model 写入 ComfyUI 模型目录 / registry metadata
  ↓
用户后续可在 ComfyUI 或其他 runtime 中使用该 artifact
```

### 1.5 量化格式、算法与模型组件的初步理解

#### 1.5.1 格式层

| 格式 | 初步理解 | 需要研究的关键细节 |
|---|---|---|
| FP8 E4M3 / E5M2 | 8-bit float，常用于 weight / activation；E4M3 精度较高、范围较小，E5M2 范围较大 | PyTorch dtype 支持边界、scale granularity、per-tensor / per-channel / per-block、硬件 kernel |
| `e4m3fn` / `e4m3fnuz` | PyTorch / GPU ecosystem 中常见 FP8 变体 | 后端差异、ROCm / CUDA 差异、是否可直接保存和恢复 |
| FP4 E2M1 | 4-bit float，通常 sub-byte packing | PyTorch `float4_e2m1fn_x2` 的 shell dtype 限制、转置 / reshape 边界 |
| NVFP4 | NVIDIA Blackwell 重点格式；数据可为 E2M1，配合 block scale | block size、E4M3 scale、global scale、TensorRT / cuDNN / kernel 路线 |
| MXFP8 | OCP microscaling，常见 block size 32，scale 可为 E8M0 | scale layout、row / column axis、MX spec 与硬件实现差异 |
| MXFP4 / MXFP6 | 更激进 microscaling 格式 | OCP spec、AMD CDNA4、Blackwell、packing / unpacking 与 validation |
| INT8 | 成熟基础格式，W8A8 / W8A16 | SmoothQuant、activation calibration、ONNX Q/DQ |
| INT4 | W4A16 / W4A8 / GPTQ / AWQ 常用 | group size、zero point、act order、packing format、kernel 对齐 |

#### 1.5.2 算法层

| 算法 | 原始强项 | 迁移到生成模型的风险 |
|---|---|---|
| GPTQ | 二阶近似的一次性 weight quantization，LLM 中成熟 | DiT / diffusion 层输入分布受 timestep / resolution / scheduler 影响；Qwen-Image 20B 显存时间压力大 |
| AWQ | activation-aware weight-only，保护 salient channels | 生成模型中“salient”的定义可能受 prompt / timestep / spatial token 影响 |
| SmoothQuant | 将 activation outlier 难度迁移到 weight，适合 W8A8 | diffusion activation outlier 是否稳定、是否跨 timestep 稳定待验证 |
| SVDQuant | 用低秩分支吸收 outlier，面向 4-bit diffusion 有明确先例 | artifact 格式、low-rank branch 与 LoRA / adapter 叠加、kernel / runtime 兼容 |
| rotation / QuaRot 类 | 通过旋转降低 outlier，改善低比特 | 对 cross-attention、modulation、time embedding、multi-modal token 的适配待研究 |
| mixed precision search | 按层选择 dtype / bit / fallback | 需要设计成本函数、敏感度指标和自动回退策略 |

#### 1.5.3 组件层

默认策略可以是：

| 组件 | 默认策略建议 | 原因 |
|---|---|---|
| Norm、embedding、final layer | 保留 bf16 / fp16 | 对质量敏感，参数占比通常不大 |
| VAE / 3D VAE | 第一阶段保留 bf16 / fp16 | 量化可能导致颜色、细节、闪烁、重建损失 |
| attention q/k/v/o | 优先 FP8 / INT8，研究 INT4 | attention 对感知质量敏感，但有成熟 kernel 路线 |
| MLP / feed-forward | 优先 INT4 / GPTQ / AWQ / SVDQuant | 参数量大，压缩收益高 |
| text encoder | 借鉴 LLM PTQ，但单独校准 | prompt 分布与视觉任务不同 |
| LoRA / adapter | 初期保留高精度或后融合 | 避免 base quant 后 adapter 行为漂移 |
| ControlNet / IP-Adapter | 作为二期 | 控制强度与条件分布复杂 |

### 1.6 初步里程碑

| 里程碑 | 目标 | 验收标准 |
|---|---|---|
| M0: 规格与 schema | 完成 config、manifest、metadata schema、packing API | 单元测试覆盖 dtype、scale、hash、manifest |
| M1: toy engine | toy Linear / toy DiT 支持 INT8 / INT4 / FP8 量化 | CPU + 小 GPU 可重复跑通 |
| M2: small model | SD1.5 / small UNet / small T5 / toy DiT 端到端 | 生成 artifact、报告、resume、失败恢复 |
| M3: ComfyUI frontend | custom nodes + daemon + report viewer | 不阻塞 ComfyUI，可启动 / 监控 / 恢复 |
| M4: SDXL / FLUX partial | 中型模型分层量化 | 可定位敏感层、自动 fallback |
| M5: Qwen-Image partial | Qwen-Image 单层 / 单 block GPTQ / FP8 / SVDQuant | 证明显存调度与 checkpoint 可用 |
| M6: Qwen-Image full gate | Qwen-Image 20B GPTQ / SVDQuant / FP8 至少一条完整跑通 | H100 80GB 级别作为严肃下限，生成完整报告 |
| M7: FP4 / MX gate | NVFP4 / MXFP8 / MXFP4 产物生成与硬件后端验证 | Blackwell / MI350 或云环境验证 |
| M8: video gate | Wan / HunyuanVideo / CogVideoX / LTX-Video 中至少一条视频模型路线 | temporal consistency、3D VAE 策略、长任务稳定性 |

### 1.7 用卡规划

#### 1.7.1 分阶段 GPU / 资源规划

| 阶段 | GPU / 资源 | 目标 | 说明 |
|---|---|---|---|
| P0 单元开发 | CPU + 任意小 GPU | tensor packing、scale 计算、manifest、CLI、job resume | 不验证大模型质量 |
| P1 小模型开发 | 12GB-24GB GPU | SD1.5 / 小 UNet / toy DiT / 小 text encoder | 验证算法流程，不代表大模型可用 |
| P2 主开发 | RTX 4090 24GB / RTX 5090 32GB | 插件、SDK、CLI、SDXL、小规模 FP8 / INT4 / GPTQ | 只能做 smoke 和小模型 |
| P3 中型验证 | RTX 6000 Ada 48GB / L40S 48GB | FLUX / SD3 部分量化，Qwen-Image 分层量化 | 大模型完整 GPTQ 仍偏紧 |
| P4 大模型最低验收 | H100 80GB | Qwen-Image 20B GPTQ / SVDQuant 最小完整跑通，FP8 验证 | 80GB 是严肃下限，不是舒适配置 |
| P5 大模型舒适验收 | H200 141GB / RTX PRO 6000 Blackwell 96GB | Qwen-Image full quant、较大 calibration、长任务稳定性 | 推荐作为正式开发验收 |
| P6 FP4 / NVFP4 验证 | RTX PRO 6000 Blackwell / B200 / GB200 | NVFP4、FP4 E2M1、Blackwell native path | 需要 Blackwell 级硬件或云资源 |
| P7 AMD 兼容性 | MI300X 192GB / MI350 288GB | ROCm、FP8 E4M3 / E5M2、MXFP4 / MXFP6 路线 | AMD 后端单独验证 |
| P8 视频模型验收 | H200 / B200 / MI300X / MI350 / 多卡 | Wan、HunyuanVideo、CogVideoX、LTX-Video | temporal consistency 和 3D VAE 压力大 |

#### 1.7.2 三档资源方案

**方案 A：低成本起步**

- 本地：RTX 4090 24GB 或 RTX 5090 32GB。
- 云端：按需租 H100 80GB / H200 141GB / Blackwell。
- 适合：个人或小团队先把架构、API、CLI、ComfyUI 节点、toy / small model 跑通。
- 风险：大模型 GPTQ / SVDQuant 无法本地闭环，release gate 依赖云资源排期。

**方案 B：严肃产品开发**

- 本地：RTX PRO 6000 Blackwell 96GB 或 RTX 6000 Ada 48GB。
- 云端：H100 / H200 / B200 / MI300X / MI350。
- 适合：需要持续做 Qwen-Image / FLUX / SD3.5 / FP4 验证的团队。
- 风险：单卡仍不足以覆盖所有视频模型与多后端，需要排队系统和 artifact storage。

**方案 C：团队级生产验证**

- 本地或私有云：2-8 张 H100 / H200 / B200 / MI300X / MI350。
- 配套：任务队列、artifact registry、NVMe cache、报告 dashboard、nightly regression。
- 适合：目标是对外提供稳定量化库、模型产物和商业级支持。
- 风险：工程复杂度高，需要专门的 job scheduler、资源隔离和成本控制。

### 1.8 测试矩阵

#### 1.8.1 单元测试

- FP8 E4M3 / E5M2 pack / unpack / clamp / roundtrip。
- FP4 E2M1 pack / unpack，特别是 2 个 4-bit 值 packed into 1 byte。
- MX block size 32，E8M0 scale layout。
- NVFP4 block size 16，E4M3 scale，global scale metadata。
- INT4 symmetric / asymmetric group quant，group size 32 / 64 / 128。
- safetensors metadata schema 读写。
- manifest hash、source hash、artifact hash。
- checkpoint resume：中断后能跳过已完成层。
- toy GPTQ：小 Linear 层与可控输入，误差可解释。
- toy AWQ / SmoothQuant：activation stats 与等价变换验证。
- toy SVDQuant：low-rank branch + residual branch 组合误差验证。

#### 1.8.2 小模型测试

- SD1.5 UNet。
- SDXL UNet 的部分 block。
- small T5 / CLIP text encoder。
- toy DiT。
- FLUX 子模块或截断模型。

#### 1.8.3 中型验证

- SD3 medium。
- FLUX partial / selected blocks。
- Qwen-Image single layer / single block / selected stage。
- HunyuanDiT partial。

#### 1.8.4 大模型 release gate

- Qwen-Image 20B GPTQ INT4。
- Qwen-Image 20B SVDQuant。
- Qwen-Image 20B FP8 E4M3。
- FLUX.1 dev mixed precision。
- SD3.5 large mixed precision。
- Wan / HunyuanVideo / CogVideoX / LTX-Video 至少一种视频模型。

#### 1.8.5 硬件特定验证

- FP8 on H100 / H200 / MI300X。
- NVFP4 / FP4 native or near-native on Blackwell。
- MXFP8 / MXFP4 backend / hardware validation。
- ROCm FP8 / MXFP4 / MXFP6。
- TensorRT explicit Q/DQ。
- Nunchaku / SVDQuant artifact。

---

## 第二部分：探索任务

> **致研究 AI 的说明**：
>
> 以下不是一系列需要你“回答”的问题，而是一组**探索区域**。每个区域描述了我们遇到的具体困境，以及我们这个外行团队目前能想到的朴素做法。
>
> 你的任务是：
>
> 1. 先深入理解每个困境的本质；
> 2. 然后自由探索所有可能技术路径，尤其是我们没有想到的；
> 3. 特别关注跨领域借鉴：编译器、数据库、游戏资产管线、视频编码、HPC checkpoint、渲染农场、EDA design space exploration、MLOps registry 等领域有没有类似问题；
> 4. 对每条发现的路径，给出具体到可以开始编码的实现方案；
> 5. 如果发现我们整个技术路线走偏了，请直接指出。
>
> 我们是这个细分领域的外行。我们能想到的方案大概率只是“加大扇叶”级别的表面方案。请像一个同时精通生成模型、低比特数值格式、GPU kernel、编译器和产品工程的资深专家那样思考。

### 探索区域 1：主流生成模型离线量化适配边界的可能性空间

**困境**：  
ComfyUI 生态的“模型”不是单一 Transformer。SDXL 有 UNet / CLIP / VAE；SD3 / FLUX / Qwen-Image 是 DiT / MMDiT；视频模型还有 3D VAE、temporal attention、image-to-video 条件、vision encoder；LoRA、ControlNet、IP-Adapter、model patch 又会改变权重或前向结构。如果离线量化库只按“遍历 Linear 然后量化”理解，很可能在第一个大模型上就失败。

**我们这个外行能想到的做法**：

> 1. 为每个模型族写一个 adapter，手工列出哪些层能量化。  
> 2. 先只支持 Linear，不碰 Conv、Norm、VAE、adapter。  
> 3. 对未知层全部 fallback 到 bf16。  

**但我们怀疑这只是冰山一角。**

请自由探索：

- 是否应该设计统一的 model IR / graph IR，而不是为每个模型写散乱 adapter？
- 如何识别生成模型中的敏感结构：time embedding、modulation、AdaLN、cross-attention、dual-stream / single-stream block、RoPE / positional embedding、VAE decoder？
- LoRA / ControlNet / IP-Adapter 与 base model 量化之间有哪些顺序关系：先量化 base 再加载 LoRA，还是 adapter-aware calibration？
- 对 Qwen-Image 这种 MMDiT，哪些模块应被视为 LLM-like，哪些模块应被视为 vision diffusion-like？
- 视频模型的 temporal attention、3D VAE、motion module 是否需要单独 adapter contract？
- **跨领域提示**：编译器 pass manager、PyTorch FX / torch.export、数据库 query planner、游戏资产 import pipeline 是如何把异构对象统一成可处理 IR 的？这些经验能不能用于“模型族 adapter + quant pass”设计？

### 探索区域 2：FP8 / NVFP4 / MXFP8 / E4M3 / E2M1 等格式生产与混合精度策略的可能性空间

**困境**：  
“支持 FP8、NVFP4、MXFP8、E4M3、E2M1”不是简单地把 tensor cast 成某个 dtype。真实问题包括 scale granularity、block size、scale dtype、packing layout、axis、rounding、clamp、metadata、backend compatibility、artifact schema、导出后谁能读取。尤其 PyTorch 中很多低比特 dtype 是 shell dtype，op 支持有限；Blackwell / cuDNN / TensorRT / AMD CDNA4 / OCP MX 之间也可能存在细节差异。

**我们这个外行能想到的做法**：

> 1. 直接用 PyTorch dtype cast 到 `float8_e4m3fn`。  
> 2. 用 safetensors 保存 quantized tensor 和 scale tensor。  
> 3. 对 FP4 / MX 先自己 pack bytes，metadata 里记录格式。  

**但我们怀疑这只是冰山一角。**

请自由探索：

- 对 FP8 E4M3 / E5M2，weight-only、activation-only、W8A8、dynamic activation、static activation 分别需要什么 scale schema？
- NVFP4 的 block size、E2M1 数据、E4M3 scale、global scale 应如何抽象，才能既支持 NVIDIA recipe，又不把库绑死在单一 runtime？
- MXFP8 / MXFP4 / MXFP6 的 OCP microscaling schema 应如何映射到 safetensors、ONNX、TensorRT、torchao、ROCm？
- 是否应设计一个 backend-independent `QuantTensor` 抽象：`payload + scale + zero + layout + axis + block_size + semantic_dtype + physical_dtype + packing`？
- 如何做格式 roundtrip 测试，保证 pack → save → load → dequant 与 reference 一致？
- 哪些格式适合“生产 artifact”，哪些只适合“runtime acceleration”，不应承诺通用保存？
- **跨领域提示**：视频编码 block quantization、GPU texture compression、音频 codec、列式数据库压缩格式是如何表达 block、scale、metadata 和兼容版本的？这些格式设计经验能不能迁移？

### 探索区域 3：GPTQ / AWQ / SmoothQuant / SVDQuant 等后训练量化算法迁移到生成模型的可能性空间

**困境**：  
GPTQ、AWQ、SmoothQuant 来自 LLM 领域；SVDQuant 明确面向 diffusion 4-bit，但也有特定实现与 runtime 假设。生成模型的激活分布由 prompt、timestep、noise、scheduler、resolution、CFG、control condition、video frame 数共同决定。直接套 LLM 校准集和 layer order 可能得到完全错误的敏感度判断。

**我们这个外行能想到的做法**：

> 1. 用一批 prompt 跑前向，收集 Linear 输入，然后套 GPTQ。  
> 2. 对 text encoder 复用 LLM AWQ / SmoothQuant。  
> 3. 对 diffusion transformer 尝试 SVDQuant，失败层 fallback bf16。  

**但我们怀疑这只是冰山一角。**

请自由探索：

- diffusion / DiT 的校准集应如何采样：prompt、negative prompt、中文 / 英文 / 多语文本渲染、resolution、timestep、sigma、scheduler、CFG、batch、seed？
- GPTQ 的 Hessian / second-order approximation 在 DiT block 中如何定义？attention 和 MLP 是否应分开求解？
- AWQ 的 activation-aware salient channel 对 MMDiT 是否稳定？saliency 是否跨 timestep 稳定？
- SmoothQuant 的 outlier migration 在 diffusion 中会不会改变不同 timestep 的动态范围，导致局部改善但全局变差？
- SVDQuant 的 low-rank branch 如何与 LoRA、ControlNet、model patch 叠加？是否会冲突、重复低秩表达或需要 merge strategy？
- 对视频模型，应如何把 temporal consistency 纳入量化误差目标，而不是只看单帧？
- 是否需要 distillation / QAT 作为 PTQ 失败后的二期路线？
- **跨领域提示**：感知视频编码的 rate-control、profile-guided optimization、自动混合精度训练、数值线性代数中的 low-rank approximation、图像压缩中的 perceptual loss 是否能提供更好的量化目标？

### 探索区域 4：大模型离线量化 job 系统、显存调度、断点续跑与用卡规划的可能性空间

**困境**：  
Qwen-Image 20B 做 GPTQ / SVDQuant，按当前规划经验应按约 70GB+ VRAM、几十小时级别任务看待。即使具体数字需要实测修正，结论也很明确：这不是一个 ComfyUI 同步节点能轻松完成的任务。没有 job manifest、分层 checkpoint、offload、恢复、失败 fallback、进度监控和资源规划，整个库无法产品化。

**我们这个外行能想到的做法**：

> 1. 一层一层量化，每层完成后保存 checkpoint。  
> 2. 显存不够就 CPU offload / NVMe offload。  
> 3. ComfyUI 里显示一个进度条和日志。  

**但我们怀疑这只是冰山一角。**

请自由探索：

- 离线量化 job 应如何切分：按 layer、block、component、algorithm phase、calibration shard 还是 artifact shard？
- 如何在任务开始前估算显存：权重、激活缓存、Hessian / covariance、临时矩阵、scale、workspace、offload buffer？
- GPTQ / SVDQuant 的 memory peak 应如何降低：streaming calibration、blockwise Hessian、CPU pinned memory、NVMe mmap、多进程隔离？
- checkpoint 粒度如何设计，既能避免重复几十小时，又不会产生海量碎片和一致性问题？
- 多 GPU 是否有意义：按层并行、按 calibration shard 并行、按模型组件并行、还是只做数据收集并行？
- job failure policy 应有哪些：stop、retry、reduce calibration、fallback bf16、fallback fp8、skip layer、mark unsafe？
- ComfyUI 前端如何避免用户误以为“点击 Queue 就会马上出图”？
- **跨领域提示**：HPC checkpoint / restart、渲染农场、数据库长事务、CI artifact cache、ZeRO / DeepSpeed offload、视频转码队列是如何处理长任务、失败恢复和资源调度的？

### 探索区域 5：ComfyUI 插件 API、用户工作流、报告系统与量化模型注册的可能性空间

**困境**：  
虽然项目核心是离线量化库，但用户最终希望在 ComfyUI 中使用成熟 API / node。ComfyUI custom node 有自己的输入输出类型、缓存模型、前端交互和 server route 机制。如果插件做得太薄，用户无法配置复杂 policy；做得太厚，又会把重型 job 塞进 UI 线程或普通执行图，导致体验灾难。

**我们这个外行能想到的做法**：

> 1. 做几个节点：选择模型、选择算法、开始量化、查看报告。  
> 2. 用 `STRING` 传路径，用自定义类型传 policy。  
> 3. 用本地 JSON 文件保存 job 状态。  

**但我们怀疑这只是冰山一角。**

请自由探索：

- ComfyUI custom node 的最佳边界是什么：哪些参数放 node widget，哪些放高级 JSON / YAML policy，哪些放外部 config？
- 是否需要独立 daemon + REST / WebSocket route，而不是只用节点函数？
- 如何设计自定义 datatype：`MODEL_INFO`、`CALIBRATION_SET`、`QUANT_POLICY`、`QUANT_JOB`、`QUANT_REPORT`、`QUANT_ARTIFACT`？
- 如何在 UI 中展示“这是离线任务，不是当前 workflow 的即时推理步骤”？
- 报告应包含哪些信息：layer error、fallback 层、量化前后大小、预估 VRAM、真实耗时、calibration 覆盖、smoke samples、兼容性矩阵？
- 注册量化产物时，如何避免污染原模型目录、如何保留 source hash、如何让用户知道 runtime 是否支持？
- 是否应该支持 headless CLI 产物导入 ComfyUI，而不是要求所有量化都从 UI 发起？
- **跨领域提示**：Blender background render、Unreal asset cooking、DaVinci Resolve proxy generation、MLOps model registry、Unity asset import pipeline 是如何把耗时离线处理包装成前端可控流程的？

### 探索区域 6：我们没想到的方向

以上五个区域是我们这些外行能划定的范围。**但几乎可以肯定，还有我们完全不知道的方向存在。**

**困境**：  
我们现在默认要做一个“大而全”的离线量化库，覆盖多模型、多算法、多格式、多后端。但这可能不是最优路径。也许应该先围绕 torchao / TensorRT ModelOpt / Nunchaku / existing PTQ libs 做 orchestration；也许应该选择 SVDQuant-first；也许应该从 Qwen-Image 单点 benchmark 反推系统；也许应该做 quantization recipe registry，而不是自己实现所有算法。

**我们这个外行能想到的做法**：

> 1. 先做自研 core，再慢慢接第三方后端。  
> 2. 先覆盖格式，再覆盖算法。  
> 3. 先做 ComfyUI 插件，再补 CLI 和 SDK。  

**但我们怀疑这只是冰山一角。**

请自由探索：

- 最近 6-12 个月是否出现了更适合生成模型的量化论文、库、kernel、artifact format？
- 是否应采用 SVDQuant-first / Nunchaku-first 路线，而不是 GPTQ-first？
- 是否应采用 TensorRT ModelOpt / torchao / ONNX QDQ 作为核心 backend，自己只做 adapter、calibration、policy 和 report？
- 是否应把 MVP 收窄为“Qwen-Image 20B offline quantization factory”，先打穿一个最难点，再泛化？
- 是否需要引入 QAT / distillation / LoRA-based correction，而不是坚持纯 PTQ？
- 是否应该量化“workflow”而不是单模型：base + text encoder + VAE + LoRA + ControlNet 作为整体 recipe？
- 是否可以训练一个 layer sensitivity predictor，减少每个大模型上重复搜索？
- 是否应该建立公共 artifact registry：每个量化产物都有 source hash、recipe、calibration、quality report、runtime compatibility？
- 有没有完全不同路线：低秩 adapter 替代低比特、sparse + quant 联合、activation caching、token pruning、scheduler-aware compression？
- **跨领域提示**：EDA design space exploration、视频编码 RDO、编译器 auto-tuning、数据库 cost model、游戏 mipmap / LOD、科学计算 adaptive precision 是否能启发“自动选择量化策略”的系统？

---

## 第三部分：我们已知的信息（请勿重复研究这些）

> 说明：以下信息是截至 2026-05-11 已收集的基础背景。研究 AI 不需要重复解释这些基础概念，但可以继续核验是否有更新版本、替代实现或更适合本项目的实践。

### 3.1 ComfyUI custom node / server 基础

- ComfyUI custom node backend 的核心属性包括 `INPUT_TYPES`、`RETURN_TYPES`、`RETURN_NAMES`、`CATEGORY`、`FUNCTION`，节点函数返回值应对应 `RETURN_TYPES`。  
  参考：https://docs.comfy.org/custom-nodes/backend/server_overview
- ComfyUI 支持 hidden inputs，例如 `UNIQUE_ID`、`PROMPT`、`EXTRA_PNGINFO`，可用于节点与服务端通信或记录 workflow metadata。  
  参考：https://docs.comfy.org/custom-nodes/backend/more_on_inputs
- ComfyUI 自定义 datatype 可以用大写字符串表示，并通过 `forceInput` 强制作为连线输入，而不是普通 widget。  
  参考：https://docs.comfy.org/custom-nodes/backend/more_on_inputs
- ComfyUI 有 server routes / comms 文档，应研究是否用于 job daemon、状态查询、WebSocket / route 扩展。  
  参考：https://docs.comfy.org/development/comfyui-server/comms_overview

### 3.2 ComfyUI 生态中的目标模型事实

- Qwen-Image 是 20B 参数 MMDiT 图像生成模型；ComfyUI 文档中有 `fp8_e4m3fn` 版本与 bf16 / fp8 文件大小参考，并给出 RTX 4090D 24GB 上 fp8 workflow 的 VRAM / 时间示例。  
  参考：https://docs.comfy.org/tutorials/image/qwen/qwen-image
- HunyuanVideo 是 13B 参数级视频生成模型，包含 DiT 架构、3D VAE、text-to-video / image-to-video workflow。  
  参考：https://docs.comfy.org/advanced/hunyuan-video

### 3.3 PyTorch / torchao / dtype 基础

- PyTorch tensor attributes 文档列出 `torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch.float8_e4m3fnuz`、`torch.float8_e5m2fnuz`、`torch.float8_e8m0fnu`、`torch.float4_e2m1fn_x2` 等 dtype；其中多个低比特 dtype 是 shell dtype，op / backend 支持有限。  
  参考：https://docs.pytorch.org/docs/stable/tensor_attributes.html
- `torch.float4_e2m1fn_x2` 表示一个 byte 中 packed 两个 4-bit 值；形状 / stride 类操作在 byte boundary 上工作，不会自动 unpack / repack sub-byte 值。  
  参考：https://docs.pytorch.org/docs/stable/tensor_attributes.html
- torchao 是 PyTorch-native optimization / quantization 库，支持 quantized training、QAT、quantized inference、float8、int4 等工作流，可作为后端或参考实现。  
  参考：https://docs.pytorch.org/ao/stable/

### 3.4 NVIDIA / TensorRT / cuDNN / Blackwell 格式信息

- TensorRT explicit quantization 支持 INT8、FP8E4M3、INT4、FP4E2M1 等低精度类型；ONNX 导出通常使用显式 Q/DQ 表示。  
  参考：https://docs.nvidia.com/deeplearning/tensorrt/10.12.0/inference-library/work-quantized-types.html
- cuDNN frontend block scaling 文档中，MXFP8 recipe 可按 32 个 FP32 元素生成 32 个 FP8 值与 1 个 E8M0 scale；NVFP4 recipe 可按 16 个 FP32 元素生成 16 个 FP4 E2M1 值与 1 个 FP8 E4M3 scale。  
  参考：https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/BlockScaling.html
- NVIDIA H100 支持 FP8 Transformer Engine；官方规格中 H100 SXM 为 80GB，H100 NVL 为 94GB。  
  参考：https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA H200 官方规格为 141GB GPU memory、4.8TB/s bandwidth，适合作为大模型离线量化的更舒适资源。  
  参考：https://www.nvidia.com/en-us/data-center/h200/
- NVIDIA Blackwell 架构引入第二代 Transformer Engine、micro-tensor scaling、FP4 AI 等能力。  
  参考：https://www.nvidia.com/en-gb/data-center/technologies/blackwell-architecture/
- NVIDIA RTX PRO 6000 Blackwell 系列提供 96GB GDDR7 ECC，适合作为高端本地工作站验证资源候选。  
  参考：https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/
- NVIDIA developer blog 中有 NVFP4 与 FP4 image generation / FLUX / ComfyUI 相关内容，应作为 Blackwell 路线进一步研究。  
  参考：https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/  
  参考：https://developer.nvidia.com/blog/?p=99256

### 3.5 OCP MX / AMD 路线

- OCP Microscaling Formats MX v1.0 spec 是 MXFP8 / MXFP4 等 microscaling 格式的重要标准来源。  
  参考：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- AMD MI300X 官方规格包含 192GB HBM3、FP8 E5M2 / E4M3 performance 信息，可作为 ROCm / FP8 验证资源。  
  参考：https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- AMD MI350 系列提供 288GB HBM3E、8TB/s bandwidth，并标称支持 MXFP6 / MXFP4 等扩展 datatype。  
  参考：https://www.amd.com/en/products/accelerators/instinct/mi350.html
- AMD CDNA4 架构文档列出 MI350 系列 Matrix Core 支持 MXFP4、MXFP6、MXFP8、OCP FP8 等，适合研究 AMD 后端 microscaling 路线。  
  参考：https://www.amd.com/en/technologies/cdna.html

### 3.6 算法论文与已有实现方向

- SVDQuant：`arXiv:2411.05007`，题为 *SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models*。核心思想是用高精度低秩分支吸收 outlier，低比特分支处理 residual；论文还提出 Nunchaku inference engine，并在 SDXL、PixArt、FLUX.1 等生成模型上验证。  
  参考：https://arxiv.org/abs/2411.05007
- GPTQ：`arXiv:2210.17323`，题为 *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*。二阶近似 one-shot weight quantization，是 INT4 / INT3 LLM PTQ 的经典路线。  
  参考：https://arxiv.org/abs/2210.17323
- AWQ：`arXiv:2306.00978`，题为 *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*。通过 activation-aware salient channel 保护实现 weight-only 低比特量化。  
  参考：https://arxiv.org/abs/2306.00978
- SmoothQuant：`arXiv:2211.10438`，题为 *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*。通过把 activation outlier 难度迁移到 weights，实现 W8A8 PTQ。  
  参考：https://arxiv.org/abs/2211.10438
- QuaRot：`arXiv:2404.00456`，题为 *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*。旋转消除 outlier 的思路可作为生成模型低比特量化补充方向。  
  参考：https://arxiv.org/abs/2404.00456

### 3.7 不应重复犯的理解错误

- 不要把本项目写成“ComfyUI 加载量化模型推理”的项目。
- 不要假设已有本地代码、已有量化结果、已有 benchmark。
- 不要只研究 LLM 量化，不研究 diffusion / DiT / MMDiT / video DiT 的校准差异。
- 不要只列格式名；必须研究 scale、packing、metadata、artifact、backend compatibility。
- 不要认为 24GB 卡可以验收 Qwen-Image 20B GPTQ / SVDQuant full quant；24GB 主要用于开发和 smoke。
- 不要只给理论建议；输出必须具体到可以开始编码。

---

## 第四部分：期望的输出格式

请研究 AI 对**每个探索区域**按以下结构输出。每个方案都必须具体到可以开始编码、可以排期、可以做验收。

### 探索区域 X：[名称]

#### 发现的方案全景

请先列出所有发现路径，包括已知路径和新发现路径：

| 编号 | 方案名 | 一句话描述 | 适用阶段 | 主要风险 |
|---|---|---|---|---|
| X.1 | 方案名 | 描述 | MVP / 中期 / 终局 | 风险 |
| X.2 | 方案名 | 描述 | MVP / 中期 / 终局 | 风险 |

#### 方案详情

##### 方案 X.1：[名称]

**一句话原理**：  
用一句话说明这个方案为什么能解决问题。

**灵感来源**：  
来自哪个论文、项目、库、硬件文档、编译器设计、视频编码、数据库系统、游戏引擎或其他领域？

**具体实现步骤**：

1. 要新增哪些模块 / 文件 / 类 / 函数？
2. 输入是什么？输出是什么？
3. 数据结构如何定义？请给出 schema 或 Python dataclass 示例。
4. 核心算法伪代码是什么？
5. 如何与 `comfy_quant` SDK / CLI / job system / ComfyUI node 对接？
6. 如何做最小单元测试？
7. 如何做小模型 smoke？
8. 如何做大模型 release gate？

**涉及组件**：

- model adapter：
- calibration：
- algorithm：
- format：
- job：
- ComfyUI node：
- validation：

**与计划系统的对接方式**：

- 需要修改哪些模块？
- 输入格式怎么变？
- 输出 artifact 怎么变？
- manifest / metadata 需要增加哪些字段？
- 是否影响已有 policy？
- 是否依赖特定硬件或 runtime？

**难度评估**：  
低 / 中 / 高，并说明原因。必须拆分：

- 算法难度；
- 工程难度；
- 硬件依赖；
- 验证成本；
- 维护成本。

**预期效果**：

- 能降低多少显存 / 文件大小？
- 对量化耗时有什么影响？
- 对生成质量可能有什么影响？
- 对推理 runtime 兼容性有什么影响？
- 有哪些模型族最可能受益？

**先例 / 参考实现**：

- 论文：
- GitHub repo：
- 官方文档：
- 博客：
- 相关 issue / PR：

**风险与不确定性**：

- 最可能在哪里翻车？
- 如果失败，fallback 是什么？
- 如何尽早用小实验验证？
- 需要什么 GPU / 云资源？

#### 技术顾问建议

如果你是这个项目的技术顾问，请给出：

1. **推荐 MVP 路径**：最快出结果的路线是什么？为什么？
2. **推荐终极路径**：效果最好、最可持续的路线是什么？为什么？
3. **实施顺序**：按周或按里程碑列出。
4. **不要做什么**：哪些路线看起来诱人但现在不值得做？
5. **验收 gate**：每个阶段必须通过哪些测试才算完成？

#### 意外发现

请单独列出研究过程中发现的、不属于上述探索区域的新方向。尤其是：

- 完全超出当前思维框架的方案；
- 能显著降低用卡成本的方案；
- 能避免自研大部分底层算法的方案；
- 能把项目从“大而全”变成“先打穿关键点”的方案；
- 发现我们技术路线可能走偏的证据。

**特别强调**：如果发现类似“不是加大扇叶，而是重新设计风道”的灵感，请务必单独列出。这类发现对我们价值最高。

### 4.1 研究报告必须包含的总表

请在报告开头提供以下总表：

| 类别 | 推荐路线 | MVP 是否需要 | 终局是否需要 | 依赖硬件 | 风险 |
|---|---|---|---|---|---|
| model adapter |  |  |  |  |  |
| FP8 |  |  |  |  |  |
| NVFP4 |  |  |  |  |  |
| MXFP8 / MXFP4 |  |  |  |  |  |
| GPTQ |  |  |  |  |  |
| SVDQuant |  |  |  |  |  |
| job / checkpoint |  |  |  |  |  |
| ComfyUI plugin |  |  |  |  |  |
| validation |  |  |  |  |  |

### 4.2 研究报告必须给出的工程交付物建议

请输出一个可执行 backlog：

| 优先级 | 任务 | 目标文件 / 模块 | 估算工作量 | 依赖 | 验收方式 |
|---|---|---|---|---|---|
| P0 |  |  |  |  |  |
| P1 |  |  |  |  |  |
| P2 |  |  |  |  |  |

### 4.3 研究报告必须给出的用卡建议

请明确回答：

1. 如果只有 RTX 4090 24GB / RTX 5090 32GB，能做哪些开发？不能做哪些验收？
2. Qwen-Image 20B GPTQ / SVDQuant full quant 最低建议 GPU 是什么？为什么？
3. H100 80GB、H200 141GB、RTX PRO 6000 Blackwell 96GB、B200 / GB200、MI300X / MI350 分别适合做什么？
4. 哪些任务可以云端偶发跑，哪些任务需要持续本地资源？
5. 如何设计 nightly / weekly / release gate，避免每次改代码都烧几十小时？

### 4.4 研究报告必须给出的 MVP 建议

请给出一个收敛的 MVP，不要只给“大而全”路线。建议至少比较以下三种 MVP：

1. **格式与 job MVP**：先实现 FP8 / INT4 packing、manifest、checkpoint、CLI、ComfyUI job。
2. **Qwen-Image 单点 MVP**：围绕 Qwen-Image 20B 做 FP8 + GPTQ partial / full gate。
3. **SVDQuant-first MVP**：围绕 SVDQuant / Nunchaku 思路做 4-bit diffusion artifact。

每个 MVP 都要说明：

- 4 周能完成什么；
- 8 周能完成什么；
- 12 周能完成什么；
- 需要什么 GPU；
- 最大技术风险；
- 若失败如何转向。

### 4.5 研究报告禁止事项

请不要输出以下内容：

- 只介绍“什么是量化”的科普；
- 只列论文名，不给工程路径；
- 只给 runtime 推理方案，不给离线量化生产方案；
- 只建议“使用现成库”，但不说明如何封装、如何扩展、如何验证；
- 只说“需要更多实验”，但不给实验矩阵、GPU 预算和成功 / 失败判据；
- 假设已有本地项目、已有代码、已有实验结果；
- 忽略 ComfyUI 插件只是前端这一边界。

---

## 附录 A：初始配置草案

### A.1 QuantConfig 草案

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

Algorithm = Literal[
    "fp8_static",
    "fp8_dynamic",
    "nvfp4",
    "mxfp8",
    "mxfp4",
    "int8",
    "int4",
    "gptq",
    "awq",
    "smoothquant",
    "svdquant",
    "mixed",
]

TargetFormat = Literal[
    "fp8_e4m3",
    "fp8_e5m2",
    "nvfp4",
    "mxfp8",
    "mxfp4",
    "int8",
    "int4",
    "mixed",
]

@dataclass
class QuantConfig:
    algorithm: Algorithm
    target_format: TargetFormat
    group_size: int = 128
    block_size: Optional[int] = None
    scale_dtype: Optional[str] = None
    zero_point: bool = False
    symmetric: bool = True
    act_order: bool = False
    damp_percent: float = 0.01
    mixed_precision: dict = field(default_factory=dict)
    keep_modules: list[str] = field(default_factory=list)
    fallback_dtype: str = "bf16"
    vram_budget_gb: int = 80
    cpu_offload: bool = True
    nvme_offload_dir: Optional[str] = None
    deterministic: bool = True
    seed: int = 1234
```

### A.2 CalibrationConfig 草案

```python
@dataclass
class CalibrationConfig:
    prompt_set: str
    num_samples: int
    resolutions: list[int]
    timestep_strategy: Literal["uniform", "stratified", "scheduler_weighted"]
    scheduler: str
    cfg_values: list[float] = field(default_factory=lambda: [1.0, 3.5, 7.0])
    negative_prompt_policy: Literal["none", "default", "mixed"] = "mixed"
    seed: int = 1234
    capture_activations: bool = True
    cache_dir: Optional[str] = None
```

### A.3 QuantTensor metadata 草案

```json
{
  "schema_version": "cq.tensor.v1",
  "tensor_name": "dit.blocks.0.attn.q.weight",
  "semantic_dtype": "nvfp4",
  "physical_dtype": "uint8",
  "logical_shape": [4096, 4096],
  "packing": {
    "values_per_byte": 2,
    "bit_order": "low_first",
    "format": "fp4_e2m1"
  },
  "scale": {
    "dtype": "fp8_e4m3",
    "granularity": "block",
    "block_size": 16,
    "axis": -1,
    "tensor_name": "dit.blocks.0.attn.q.weight.scale"
  },
  "global_scale": {
    "dtype": "fp32",
    "value": 1.0
  },
  "algorithm": {
    "name": "nvfp4",
    "version": "0.1.0",
    "rounding": "nearest_even",
    "clamp": true
  }
}
```

---

## 附录 B：建议的验收原则

1. **能生成不等于能验收**：量化库必须生成可读、可恢复、可验证、可解释的 artifact。
2. **小模型通过不代表大模型通过**：SD1.5 / toy DiT 只验证工程流程，不能代表 Qwen-Image 20B。
3. **单张图通过不代表模型质量通过**：生成模型需要 prompt set、resolution set、seed set、timestep coverage 和视觉指标组合。
4. **低比特格式必须 roundtrip**：所有 packing 格式都必须有 reference dequant 和 bit-exact / tolerance 测试。
5. **混合精度必须可解释**：每个 fallback 层都要记录原因，例如 error 超阈值、NaN、显存不足、unsupported op。
6. **大任务必须可恢复**：任何超过 1 小时的任务都必须有 checkpoint；任何超过 10 小时的任务都必须能跨进程恢复。
7. **ComfyUI 不能被长任务绑死**：重型量化必须走后台 daemon / job system。
8. **硬件相关能力必须分层声明**：artifact 能生成、runtime 能读取、硬件能加速是三件不同的事。

---

## 附录 C：给外部研究 AI 的最终提醒

本项目真正需要的是一个**离线量化生产系统**，它的价值在于：

- 对主流 ComfyUI 生成模型做结构识别；
- 对不同模型族构建正确 calibration；
- 对 FP8 / NVFP4 / MXFP8 / MXFP4 / INT4 / INT8 / SVDQuant / GPTQ 等路线生成可靠 artifact；
- 对几十小时大任务提供可恢复 job；
- 对用户提供可理解报告；
- 对 ComfyUI 提供成熟前端节点；
- 对未来 runtime / backend 保持兼容。

请不要把研究收束成“怎么让 ComfyUI 直接跑某个量化模型”。那个方向可以作为 validation / compatibility 的一部分，但不是项目核心。

如果你只能给一个建议，请优先回答：

> 我们应该如何用最小工程量，先打穿一个真实大模型离线量化闭环，同时不把架构做死？

