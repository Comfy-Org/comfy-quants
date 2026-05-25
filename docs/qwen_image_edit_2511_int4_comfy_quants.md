# Qwen-Image-Edit-2511 INT4 量化说明（comfy-quants 集成版）

目标：从 **Qwen-Image-Edit-2511 dense transformer 权重** 直接导出 **ComfyUI / comfy-kitchen tile-pack INT4 safetensors**。

```text
dense transformer safetensors
  -> comfy_quants quantize-int4
  -> diffusion_pytorch_model.svdquant_w4a4.safetensors
```

当前集成版的关键点：

- 主入口是本仓库 CLI：`comfy_quants quantize-int4`。
- 不走 DeepCompressor / Nunchaku bridge。
- 不需要 `model.pt / scale.pt / smooth.pt / branch.pt`。
- 不需要 `--base-comfy` scaffold。
- 不需要 `tools/kitchen_native`。
- 输出就是单文件 tile-pack checkpoint + JSON report。

> 约定：本仓库根目录就是 `Comfy-Org/comfy-quants` 发布仓库；本地示例路径使用 `/workspace/comfy-quants`，实现代码位于 `src/comfy_quants`。

---

## 1. 固定几个路径

```bash
cd /workspace/comfy-quants

export CR="PYTHONPATH=src python -m comfy_quants.cli.main"

# Qwen-Image-Edit-2511 本地模型目录；不一样就改这里
export MODEL_ROOT=/workspace/models/hf-downloads/Qwen__Qwen-Image-Edit-2511

# transformer dense 权重：可以是 index.json、单个 safetensors、或 shard 目录
export SOURCE=${MODEL_ROOT}/transformer/diffusion_pytorch_model.safetensors.index.json

# 本次输出目录
export WORK=runs/qwen-edit-2511-int4-comfy-quants

# 量化设备
export DEVICE=cuda:0
```

确认命令可用：

```bash
$CR quantize-int4 --help
$CR calib --help
```

---

## 2. 先跑最短 smoke test

这个只验证格式和导出链路，不代表最终质量。

```bash
$CR quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source "$SOURCE" \
  --out "$WORK/int4-bootstrap" \
  --rank 64 \
  --device "$DEVICE" \
  --hash-output \
  --json
```

输出：

```text
$WORK/int4-bootstrap/diffusion_pytorch_model.svdquant_w4a4.safetensors
$WORK/int4-bootstrap/quantization_report.json
```

报告里的模式会是：

```text
quantization_mode = weight_only_initialization
pipeline_kind     = direct_quantize_to_kitchen_tilepack
```

如果这一步都失败，先不要跑校准/GPTQ，先检查 `SOURCE` 路径。

---

## 3. 推荐最终路线：校准 + GPTQ

推荐最终导出用：

```text
--quantization-mode svdquant_gptq_experimental
```

它需要先生成两个文件：

```text
int4_activation_stats.json
int4_gptq_hessian_stats.json
```

如果你已经有这两个文件，可以直接跳到 [第 4 节](#4-已有-statshessians-直接导出)。

### 3.1 准备校准记录

校准 JSONL 每行一个样本，最小格式：

```json
{"case_id":"case-0001","prompt":"把衣服改成红色，保持人物和背景不变","image":"0001.png","edit_type":"edit"}
```

设置路径：

```bash
export RAW_RECORDS=/data/qwen-edit-calib/records.jsonl
export IMAGE_ROOT=/data/qwen-edit-calib/images
```

字段说明：

- `case_id` 或 `id`：样本 ID。
- `prompt`：编辑提示词。
- `image`：输入图；相对路径会拼到 `IMAGE_ROOT` 下。
- `edit_type`：可选。

### 3.2 生成标准 records

```bash
$CR calib records \
  --input "$RAW_RECORDS" \
  --image-root "$IMAGE_ROOT" \
  --limit 128 \
  --out "$WORK/calib/records" \
  --json
```

输出：

```text
$WORK/calib/records/calibration_records.jsonl
```

### 3.3 生成 capture plan

```bash
$CR calib plan-int4-capture \
  --family qwen_image_edit \
  --source "$SOURCE" \
  --records "$WORK/calib/records/calibration_records.jsonl" \
  --out "$WORK/calib/capture-plan" \
  --json
```

输出：

```text
$WORK/calib/capture-plan/capture_plan.json
$WORK/calib/capture-plan/activation_samples.template.jsonl
$WORK/calib/capture-plan/capture_report.json
```

### 3.4 生成 activation_samples.jsonl

```bash
$CR calib materialize-int4-capture \
  --plan "$WORK/calib/capture-plan/capture_plan.json" \
  --out "$WORK/calib/capture-run" \
  --json
```

输出：

```text
$WORK/calib/capture-run/activation_samples.jsonl
$WORK/calib/capture-run/activation_tensors/
$WORK/calib/capture-run/capture_materialization_report.json
```

### 3.5 抓 activation

```bash
python scripts/dev/capture_qwen_image_edit_int4_activations.py \
  --model-root "$MODEL_ROOT" \
  --plan "$WORK/calib/capture-plan/capture_plan.json" \
  --records "$WORK/calib/records/calibration_records.jsonl" \
  --out-dir "$WORK/calib/capture-run" \
  --pipeline-class auto \
  --device "$DEVICE" \
  --model-dtype bfloat16 \
  --storage-dtype bfloat16 \
  --num-inference-steps 2 \
  --true-cfg-scale 1.0 \
  --max-sequence-length 512 \
  --max-rows-per-layer 128 \
  --max-rows-per-call 128 \
  --limit 128 \
  --json
```

说明：

- 这是开发用 diffusers capture 脚本，不是 ComfyUI runtime。
- 图片缺失时可以加：`--fallback-image /path/to/fallback.png`。
- 显存不够时先降：`--limit`、`--max-rows-per-layer`、`--max-rows-per-call`。

### 3.6 reduce 出 activation stats

```bash
$CR calib reduce-int4-activations \
  --samples "$WORK/calib/capture-run/activation_samples.jsonl" \
  --input-root "$WORK/calib/capture-run" \
  --out "$WORK/calib/activation-stats" \
  --channel-dim -1 \
  --json
```

输出：

```text
$WORK/calib/activation-stats/int4_activation_stats.json
```

### 3.7 reduce 出 GPTQ Hessians

```bash
$CR calib reduce-int4-gptq-hessians \
  --samples "$WORK/calib/capture-run/activation_samples.jsonl" \
  --input-root "$WORK/calib/capture-run" \
  --out "$WORK/calib/gptq-hessians" \
  --channel-dim -1 \
  --hessian-block-size 512 \
  --device "$DEVICE" \
  --json
```

输出：

```text
$WORK/calib/gptq-hessians/int4_gptq_hessian_stats.json
$WORK/calib/gptq-hessians/gptq_hessians/*.safetensors
```

### 3.8 dry-run 预检

正式写大文件前先跑这个：

```bash
$CR quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source "$SOURCE" \
  --activation-stats "$WORK/calib/activation-stats/int4_activation_stats.json" \
  --gptq-hessian-stats "$WORK/calib/gptq-hessians/int4_gptq_hessian_stats.json" \
  --quantization-mode svdquant_gptq_experimental \
  --lowrank-branch-input-basis raw \
  --lowrank-calibration weight_residual \
  --rank 64 \
  --device "$DEVICE" \
  --out "$WORK/int4-gptq-dryrun" \
  --dry-run \
  --json
```

如果失败，看：

```text
$WORK/int4-gptq-dryrun/quantization_report.json
```

重点查：

```text
activation_stats_coverage_state
gptq_hessian_coverage_state
activation_stats_missing_layer_count
gptq_hessian_missing_layer_count
```

### 3.9 正式导出

```bash
$CR quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source "$SOURCE" \
  --activation-stats "$WORK/calib/activation-stats/int4_activation_stats.json" \
  --gptq-hessian-stats "$WORK/calib/gptq-hessians/int4_gptq_hessian_stats.json" \
  --quantization-mode svdquant_gptq_experimental \
  --lowrank-branch-input-basis raw \
  --lowrank-calibration weight_residual \
  --rank 64 \
  --device "$DEVICE" \
  --out "$WORK/int4-gptq" \
  --hash-output \
  --json
```

最终输出：

```text
$WORK/int4-gptq/diffusion_pytorch_model.svdquant_w4a4.safetensors
$WORK/int4-gptq/quantization_report.json
```

---

## 4. 已有 stats/hessians 直接导出

如果已经有校准文件：

```bash
export ACT_STATS=/path/to/int4_activation_stats.json
export GPTQ_STATS=/path/to/int4_gptq_hessian_stats.json
```

直接跑：

```bash
$CR quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source "$SOURCE" \
  --activation-stats "$ACT_STATS" \
  --gptq-hessian-stats "$GPTQ_STATS" \
  --quantization-mode svdquant_gptq_experimental \
  --lowrank-branch-input-basis raw \
  --lowrank-calibration weight_residual \
  --rank 64 \
  --device "$DEVICE" \
  --out "$WORK/int4-gptq" \
  --hash-output \
  --json
```

当前仓库中可参考的已有路径：

```text
runs/qwen-image-edit-2511-int4-svdquant-gptq/activation-stats/int4_activation_stats.json
runs/qwen-image-edit-2511-int4-svdquant-gptq/gptq-hessians/int4_gptq_hessian_stats.json
```

---

## 5. 输出文件和层类型

输出目录模式：

```text
<out>/diffusion_pytorch_model.svdquant_w4a4.safetensors
<out>/quantization_report.json
```

如果 `--out` 直接给 `.safetensors` 文件：

```text
/path/model.safetensors
/path/model.quantization_report.json
```

量化层结构：

```text
attention / MLP linears:
  SVDQuant W4A4
  storage_layout = kitchen_tile_packed_w4a4

img_mod.1 / txt_mod.1:
  AWQ W4A16（如果 dense checkpoint 里有这些 tensor）

其他 tensor:
  从 dense source 复制，保持原始 dtype
```

完整 Qwen-Image-Edit-2511 的典型计数：

```text
selected_layer_count       = 720
quantized_layer_count      = 720
awq_modulation_layer_count = 120
output_tensor_count        ≈ 5893
```

最终以 `quantization_report.json` 为准。

---

## 6. report 重点字段

```text
status                         # model_written 表示写出成功
pipeline_kind                  # direct_quantize_to_kitchen_tilepack
target_format                  # svdquant_w4a4
storage_layout                 # kitchen_tile_packed_w4a4
quantization_mode              # svdquant_gptq_experimental
activation_stats_state         # loaded
gptq_hessian_stats_state       # loaded
output_hash                    # --hash-output 时有 sha256
```

当前报告仍会有这些保守标记：

```text
publishable_svdquant_gptq: false
runtime_contract_state: static_artifact_contract_only
mixed_quantization_state: experimental_svdquant_w4a4_awq_w4a16_runtime_unverified
```

意思是：仓库已经导出静态 tile-pack artifact；外部 ComfyUI/full-image runtime parity 还需要单独验证，不在本 CLI 内完成。

---

## 7. 常用参数

```text
--rank 64                          推荐先不改
--device cuda:0                     量化设备
--quantization-mode svdquant_gptq_experimental
--lowrank-branch-input-basis raw    推荐保持 raw
--lowrank-calibration weight_residual
--hash-output                       建议打开
--dry-run                           正式跑前先打开
```

OOM 时优先降低：

```text
capture: --limit 32 --max-rows-per-layer 64 --max-rows-per-call 64
hessian: --hessian-block-size 256
GPTQ:    --gptq-block-size 64
```

只做 calibrated RTN、不做 GPTQ：

```bash
$CR quantize-int4 \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --source "$SOURCE" \
  --activation-stats "$WORK/calib/activation-stats/int4_activation_stats.json" \
  --quantization-mode calibrated_svdquant \
  --out "$WORK/int4-calibrated-rtn" \
  --rank 64 \
  --device "$DEVICE" \
  --hash-output \
  --json
```

---

## 8. 放到 ComfyUI

本仓库只导出 safetensors，不启动 ComfyUI。通常把最终文件复制或软链到对应 loader 读取的模型目录，例如：

```bash
ln -s \
  /workspace/comfy-quants/$WORK/int4-gptq/diffusion_pytorch_model.svdquant_w4a4.safetensors \
  /path/to/ComfyUI/models/diffusion_models/qwen_image_edit_2511_int4_svdquant_w4a4.safetensors
```

实际目录以你的 ComfyUI/custom node loader 约定为准。

---

## 9. 最常见问题

### `SOURCE` 填什么？

优先填 transformer index：

```text
.../Qwen-Image-Edit-2511/transformer/diffusion_pytorch_model.safetensors.index.json
```

单文件 `.safetensors` 或 shard 目录也可以。

### dry-run 报 missing 怎么办？

一般是 stats/hessian 和当前 `SOURCE` 不匹配，或者 capture 没跑全。用同一个 `SOURCE` 从 3.3 重新生成 plan、capture、reduce，然后再 dry-run。

### 图片找不到怎么办？

检查 records 里的 `image`。临时可给 capture 脚本加：

```bash
--fallback-image /path/to/fallback.png
```

### 这是不是已经完全可发布？

不是。当前是集成版静态 artifact 导出流程；外部 ComfyUI/full-image 推理正确性还要另做验证。所以 report 会保守标记 `publishable_svdquant_gptq: false`。

---

## 10. 一句话命令

无校准 smoke test：

```bash
$CR quantize-int4 --family qwen_image_edit --format svdquant_w4a4 --source "$SOURCE" --out "$WORK/int4-bootstrap" --rank 64 --device "$DEVICE" --hash-output --json
```

已有 stats + hessians 后导出推荐 GPTQ 版本：

```bash
$CR quantize-int4 --family qwen_image_edit --format svdquant_w4a4 --source "$SOURCE" --activation-stats "$ACT_STATS" --gptq-hessian-stats "$GPTQ_STATS" --quantization-mode svdquant_gptq_experimental --lowrank-branch-input-basis raw --lowrank-calibration weight_residual --rank 64 --device "$DEVICE" --out "$WORK/int4-gptq" --hash-output --json
```
