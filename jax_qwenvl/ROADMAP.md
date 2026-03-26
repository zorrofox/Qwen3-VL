# JAX Qwen3-VL 功能路线图

> 基于代码深度分析（2026-03-26）生成，评估当前状态与可实现的下一步功能。

---

## 当前代码状态

| 功能 | 状态 |
|------|------|
| 训练循环（DP/FSDP/Hybrid） | 完整 |
| 多机 checkpoint（Orbax 原生 GCS） | 完整 |
| 权重加载/导出（HF safetensors） | 完整 |
| 固定形状填充（避免 XLA 重编译） | 完整 |
| 断点续训 | 完整 |
| **推理/生成** | **完全缺失** |
| TP（张量并行） | mesh 占位但未实现 |
| Vision 模型分片 | 未分片（全复制） |
| 训练验证管道 | 未实现 |
| Qwen2.5-VL 完整支持 | 权重格式已支持，架构未验证 |

---

## 功能评估，按优先级排序

### P0：推理/生成模式（最关键）

**现状**：模型只能做 forward pass 计算 loss，**无法生成文本**。这是产品化的硬性缺口。

**需实现**：
- `generate()` 方法（autoregressive 解码循环）
- Token 采样策略：greedy、top-k、top-p、temperature
- KV-cache（长文本效率必需）
- 多图/视频输入的推理 pipeline

**代码基础**：可基于 `model/qwen3_vl.py:forward()` 扩展，约 400-600 行。

---

### P1：Tensor Parallelism（TP）

**现状**：`train/sharding.py` 的 mesh 已预留 `('dp', 'fsdp', 'tp')` 三轴（第 42 行），但 `get_param_sharding_rules()` 完全没有 TP 规则，对 32B+ 模型是瓶颈。

**需实现**：
- Attention head 按 TP 轴分片（Q/K/V 按 head 维度列切，O 行切）
- MLP 中间层按 TP 轴分片（列切/行切）
- all-reduce 通信由 XLA 从 PartitionSpec 自动推导

**推荐配置**（v6e-16，32B 模型）：

| 配置 | dp | fsdp | tp | 说明 |
|------|----|----|-----|------|
| DP+TP | 4 | 1 | 4 | host 内 TP，host 间 DP |
| Hybrid+TP | 2 | 2 | 4 | host 内 FSDP+TP，host 间 DP |

---

### P2：Qwen2.5-VL 完整支持

**现状**：
- 权重加载器已支持双命名格式（`model.language_model.*` vs `model.*`）
- RoPE 有三种实现：`get_rope_index_2/25/3`
- `data_processor.py` 通过 `model_type` 参数切换 RoPE 计算
- **未验证**：ViT 架构参数（层数、隐层维度）是否与 Qwen2.5-VL 完全对齐

**需实现**：
- `Qwen2_5VLConfig` 配置类（或在现有 `Qwen3VLConfig` 中增加 `model_type` 分支）
- 端到端前向推理验证（对比 HF transformers 输出）
- `train_tpu.sh` 增加 `MODEL_TYPE` 环境变量支持

---

### P3：训练验证管道

**现状**：训练循环没有验证集评估，无法在训练中监控过拟合或生成质量。

**需实现**：
- 验证集 loss 计算（每 N 步，`eval_steps` 参数）
- 可选：接入 `evaluation/` 下的基准评测（VideoMME、MMMU 等）
- 可选：训练中每 N 步采样生成样本并记录到 tensorboard

---

### P4：Vision 模型分片优化

**现状**：所有视觉数据（`pixel_values`、ViT 参数）在所有设备上**全复制**，`sharding.py` 第 166-172 行明确对所有 `_VISION_FIELDS` 使用 `replicated`。对 27 层 ViT encoder 浪费 HBM。

**需实现**：
- VisionModel 参数按 FSDP 轴分片（与 text model 统一）
- Vision batch (`pixel_values`) 按数据轴分片

**影响**：8B 模型中 ViT 参数约 0.6B，分片后每设备节省 ~1.2GB HBM（bfloat16）。

---

### P5：完整 LoRA 覆盖

**现状**：`LoRADense` 已实现，但仅在 text attention 的 Q/K/V/O 投影上应用（`model/llm.py`）。Vision encoder 和 MLP projector 无 LoRA 支持。

**需实现**：
- Vision attention 的 LoRA（`model/vit.py` 中的 `VisionAttention`）
- MLP projector 的 LoRA（`model/vit.py` 中的 `VisionMLP`）
- `train_tpu.sh` 增加 `LORA_TARGET_MODULES` 参数

---

## 建议实施顺序

```
P0 推理生成  →  P2 Qwen2.5-VL  →  P3 验证管道  →  P1 TP  →  P4 Vision 分片  →  P5 LoRA 扩展
（可用性）       （多模型支持）     （训练质量）     （大模型）    （内存优化）        （参数效率）
```

**决策指引**：
- **目标是模型训练后能对话**：P0 是唯一必做项
- **目标是支持 32B+ 模型**：P1 是瓶颈
- **目标是扩展到 Qwen2.5-VL**：P2 风险最低，代码基础最好
- **目标是节省 HBM**：P4 工程量小，收益直接

---

## 已完成功能（供参考）

- ✅ 零 `import torch` 依赖的完整 JAX/Flax 模型实现
- ✅ XLA 固定形状填充（~250x 编译加速）
- ✅ bfloat16 混合精度训练
- ✅ DP / FSDP / Hybrid 三种并行模式
- ✅ Orbax 原生 GCS checkpoint（多机协调）
- ✅ 断点续训（opt_state re-shard）
- ✅ 8B 模型完整 1 epoch 训练验证（avg_loss=1.7937）
- ✅ HF safetensors 权重导出（流式上传 GCS）
- ✅ wandb + tensorboard 日志
