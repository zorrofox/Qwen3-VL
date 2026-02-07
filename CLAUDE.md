# Qwen3-VL 项目笔记

## 项目概览

Qwen3-VL 是阿里巴巴的多模态视觉语言模型仓库，支持图像理解、视频理解、OCR、UI 操控等能力。

### 目录结构

```
Qwen3-VL/
├── qwen-vl-utils/          # 视觉处理工具包（图像/视频加载、缩放）
│   └── src/qwen_vl_utils/
│       ├── __init__.py
│       └── vision_process.py   # 图像/视频预处理核心（smart_resize、fetch_image/video）
├── qwen-vl-finetune/       # 微调框架（PyTorch/GPU 版）
│   ├── qwenvl/
│   │   ├── data/
│   │   │   ├── data_processor.py  # 数据加载、collation、padding
│   │   │   ├── rope2d.py          # 2D RoPE 位置编码（三个变体）
│   │   │   └── __init__.py        # 数据集注册表
│   │   └── train/
│   │       ├── train_qwen.py      # 训练入口，模型加载、LoRA 配置
│   │       ├── trainer.py         # 自定义 Trainer，Flash Attention patch、分组件 LR
│   │       └── argument.py        # 训练参数 dataclass
│   ├── scripts/             # 训练脚本 + DeepSpeed 配置（zero2/3/offload）
│   ├── demo/                # 示例数据
│   └── tools/               # 数据打包、bbox 转换工具
├── jax_qwenvl/              # JAX/TPU 训练框架（已完成迁移）
│   ├── __init__.py
│   ├── types.py             # 共享类型定义（Batch, CausalLMOutput）
│   ├── config.py            # token ID、shape 常量、视觉处理默认值
│   ├── requirements.txt     # TPU 训练依赖
│   ├── data/
│   │   ├── __init__.py      # 数据集注册表
│   │   ├── rope2d.py        # 3 个 RoPE 函数（纯 numpy）
│   │   └── data_processor.py # Dataset + 2 个 Collator + 固定形状填充
│   ├── utils/
│   │   ├── __init__.py      # 公共 API 导出
│   │   └── vision_process.py # 图像/视频加载（无 torch 依赖）
│   ├── model/
│   │   ├── __init__.py      # 导出 Config, Model, load_hf_weights
│   │   ├── config.py        # Qwen3VLConfig / VisionConfig / TextConfig
│   │   ├── layers.py        # RMSNorm, SwiGLUMLP, VisionMLP, LoRADense
│   │   ├── rope.py          # Vision RoPE (2D) + Text MRoPE (3D)
│   │   ├── vit.py           # PatchEmbed3D, VisionAttention, VisionBlock, PatchMerger, VisionModel
│   │   ├── llm.py           # TextAttention (GQA), DecoderLayer, TextModel
│   │   ├── qwen3_vl.py      # Qwen3VLForConditionalGeneration 组合模型
│   │   ├── weight_loader.py  # HF safetensors → Flax params
│   │   └── weight_exporter.py # Flax params → HF safetensors + LoRA 合并
│   ├── train/
│   │   ├── __init__.py
│   │   ├── optimizer.py      # Optax 6+1 参数组 + warmup+cosine schedule
│   │   ├── train_state.py    # 扩展 Flax TrainState
│   │   ├── train_step.py     # @jax.jit train_step + 梯度累积
│   │   ├── train.py          # 训练入口（参数解析、模型加载、训练循环）
│   │   ├── sharding.py       # SPMD mesh + DP/FSDP 分片
│   │   ├── checkpoint.py     # Orbax CheckpointManager 封装
│   │   └── metrics_logger.py # wandb + tensorboard 统一日志
│   └── scripts/
│       ├── train_tpu.sh      # TPU 训练启动脚本
│       ├── tpu_validate.py   # TPU 硬件验证脚本（8 项测试）
│       ├── e2e_train_validate.py # 端到端训练验证（真实图片）
│       └── download_llava_data.sh # LLaVA 数据集下载
├── evaluation/              # 基准评测套件
│   ├── VideoMME/            # 视频多模态评测
│   ├── mmmu/                # MMMU 评测
│   ├── MathVision/          # 数学视觉推理
│   ├── ODinW-13/            # 目标检测
│   └── RealWorldQA/         # 真实世界 QA
├── cookbooks/               # Jupyter 示例（OCR、grounding、agent 等）
├── docker/                  # Docker 部署
└── web_demo_mm.py           # Gradio Web 演示界面
```

### 支持的模型

- Qwen2-VL → `Qwen2VLForConditionalGeneration`
- Qwen2.5-VL → `Qwen2_5_VLForConditionalGeneration`
- Qwen3-VL (Dense) → `Qwen3VLForConditionalGeneration`
- Qwen3-VL-MoE → `Qwen3VLMoeForConditionalGeneration`（仅支持 ZeRO-2）

规模：2B、4B、8B、32B、30B-A3B（MoE）、235B-A22B（MoE）
版本：Instruct（标准指令跟随）和 Thinking（增强推理）

### 核心依赖

- transformers (>=4.57.0) + torch — 模型加载与推理
- vLLM (>=0.11.0) — 高吞吐推理
- DeepSpeed (>=0.17.1) — 分布式训练（ZeRO-2/3）
- peft (>=0.17.1) — LoRA 微调
- Gradio — Web 界面
- av / decord — 视频解码

---

## 微调模块分析 (`qwen-vl-finetune/`)

### 支持的训练方法

| 方法 | 状态 | 说明 |
|------|------|------|
| SFT（监督微调） | 支持 | 唯一实现的训练目标，基于 next-token prediction |
| LoRA | 支持 | 通过 `--lora_enable True`，目标模块为 q/k/v/o_proj |
| 全参数微调 | 支持 | 可分组件控制（视觉编码器/MLP 投影器/LLM） |
| DeepSpeed ZeRO-2/3 | 支持 | 分布式训练优化 |

### 不支持的训练方法

- RLHF — 无实现
- DPO / PPO — 无偏好学习
- QLoRA（量化+LoRA） — 不支持 8-bit/4-bit 量化
- Prefix Tuning / Adapter — 只有 LoRA 一种 PEFT 方法
- 多任务训练 / 辅助损失 — 仅单一 causal LM 目标

### 组件级调控

通过三个标志位独立控制哪些部分参与训练：

```bash
--tune_mm_vision False   # 视觉编码器（默认冻结）
--tune_mm_mlp True       # MLP 投影层
--tune_mm_llm True       # 语言模型主体
```

支持分组件学习率：`--mm_projector_lr` 和 `--vision_tower_lr` 可独立于 `--learning_rate`。

---

## JAX/TPU 迁移完成记录

### 迁移总览

采用路线 A（Flax/NNX），从 PyTorch/GPU 完整迁移到 JAX/TPU。分 4 层 7 个 Agent 并行执行，全部代码零 `import torch` 依赖。

**Git 分支**：`feat/jax-tpu-migration`

**Commit 历史**：

| Commit | 内容 |
|--------|------|
| `043b918` | Layer 0-3 全部迁移（模型、训练、分布式、生产化） |
| `8f1b01b` | bfloat16 训练支持，8/8 TPU 测试通过 |
| `de3c208` | 端到端真实图片训练验证，vision pipeline bug 修复 |
| `a0b260a` | LLaVA-Instruct-150K 数据集支持，训练 pipeline 修复 |
| `a59ac9c` | XLA 重编译修复（所有张量固定形状填充，~250x 加速） |

---

### Layer 0 迁移（CPU 端数据预处理）

**完成时间**：2026-02-07
**Agent**：Agent 1 (RoPE) + Agent 2 (数据管道) + Agent 3 (视觉工具) 并行

#### 生成文件

| 文件 | 行数 | Agent | 说明 |
|------|------|-------|------|
| `jax_qwenvl/__init__.py` | 1 | 共享 | 包入口 |
| `jax_qwenvl/types.py` | 30 | 共享 | `Batch`、`CausalLMOutput` NamedTuple，所有字段 `np.ndarray` |
| `jax_qwenvl/config.py` | 31 | 共享 | token ID、shape 常量、视觉处理默认值 |
| `jax_qwenvl/data/__init__.py` | 65 | Agent 2 | 数据集注册表（纯 Python 配置） |
| `jax_qwenvl/data/rope2d.py` | 494 | Agent 1 | 3 个 RoPE 函数（`get_rope_index_3/25/2`），纯 numpy |
| `jax_qwenvl/data/data_processor.py` | 727+ | Agent 2 | Dataset + 2 个 Collator，返回 `Batch` NamedTuple |
| `jax_qwenvl/utils/__init__.py` | 7 | Agent 3 | 公共 API 导出 |
| `jax_qwenvl/utils/vision_process.py` | 492 | Agent 3 | 图像/视频加载，无 torch/torchvision 依赖 |

#### 关键设计决策

1. **纯 NumPy 而非 JAX NumPy**：Layer 0 是 CPU 端数据预处理，不需要 `jax.jit`，使用纯 numpy 避免动态 shape 问题
2. **`_ensure_numpy()` 辅助函数**：处理 HF processor 可能返回 torch tensor 的情况，通过 duck typing (`hasattr(val, 'numpy')`) 而非 import torch
3. **删除 torchvision 视频后端**：decord 为主、torchcodec 为备（lazy import），无 torchvision fallback
4. **PIL 逐帧 resize 替代 torchvision resize**：CPU 端预处理性能足够
5. **`_pad_sequence()` 自实现**：替代 `torch.nn.utils.rnn.pad_sequence()`
6. **Collator 返回 `Batch` NamedTuple**：确保下游 `jax.device_put()` 兼容

---

### Layer 1 迁移（模型定义 + 训练基础设施）

**完成时间**：2026-02-07
**Agent**：Agent 4 (模型定义) + Agent 5 (训练循环)

#### Agent 4：模型定义（8 个文件）

| 文件 | 行数 | 内容 |
|---|---|---|
| `jax_qwenvl/model/__init__.py` | 5 | 导出 Config, Model, load_hf_weights |
| `jax_qwenvl/model/config.py` | 156 | Qwen3VLConfig / VisionConfig / TextConfig dataclass，含 `from_pretrained` |
| `jax_qwenvl/model/layers.py` | 105 | RMSNorm, SwiGLUMLP, VisionMLP, LoRADense |
| `jax_qwenvl/model/rope.py` | 191 | Vision RoPE（2D 空间位置查找），Text MRoPE（3D 交错频率），apply_rotary_pos_emb |
| `jax_qwenvl/model/vit.py` | 427 | PatchEmbed3D, VisionAttention, VisionBlock, PatchMerger, VisionModel（含 DeepStack + remat） |
| `jax_qwenvl/model/llm.py` | 318 | TextAttention（GQA + q/k_norm）, DecoderLayer, TextModel（含 DeepStack + remat） |
| `jax_qwenvl/model/qwen3_vl.py` | 301 | Qwen3VLForConditionalGeneration, cross_entropy_loss, _scatter_embeddings, _make_packed_causal_mask |
| `jax_qwenvl/model/weight_loader.py` | 308 | HF safetensors → Flax params 转换，key 映射 + 转置 + 分片加载 |

#### Agent 5：训练基础设施（5 个文件）

| 文件 | 行数 | 内容 |
|---|---|---|
| `jax_qwenvl/train/__init__.py` | 6 | 导出 create_optimizer, TrainState, train_step 等 |
| `jax_qwenvl/train/optimizer.py` | 349 | optax.multi_transform 6+1 参数组，warmup+cosine schedule |
| `jax_qwenvl/train/train_state.py` | 45 | 扩展 Flax TrainState |
| `jax_qwenvl/train/train_step.py` | 156 | @jax.jit train_step + train_step_with_accumulation (jax.lax.scan) |
| `jax_qwenvl/train/train.py` | 487+ | 训练入口：参数解析、模型加载、权重合并、训练循环 |

#### 关键架构实现

1. **Vision RoPE**：2D 空间位置查找表 + flatten + duplicate，匹配 HF `rot_pos_emb` 方法
2. **Text MRoPE**：3D 交错频率布局 `[THWTHW...]`，匹配 HF `apply_interleaved_mrope`
3. **DeepStack**：ViT 在指定层（如 [8,16,24]）提取中间特征，注入 LLM 早期层
4. **q_norm / k_norm**：Qwen3 特有的 per-head RMSNorm（在 projection 后、RoPE 前）
5. **GQA**：通过 `jnp.repeat` 展开 K/V heads 到 Q heads 数量
6. **3D Conv PatchEmbed**：Flax `nn.Conv` 实现 `(temporal_patch_size, patch_size, patch_size)` 3D 卷积
7. **cu_seqlens → block-diagonal mask**：替代 Flash Attention varlen，用 JAX 实现
8. **LoRADense**：内联 LoRA（lora_rank=0 时退化为普通 Dense），初始化 lora_B=zeros

---

### Layer 2 迁移（分布式训练 + Checkpoint）

**完成时间**：2026-02-07
**Agent**：Agent 6

#### 新建文件

| 文件 | 行数 | 内容 |
|---|---|---|
| `jax_qwenvl/train/sharding.py` | 132 | `create_device_mesh()`, `get_param_sharding_rules()`, `shard_params()`, `shard_batch()` |
| `jax_qwenvl/train/checkpoint.py` | 83 | `CheckpointManager` 封装 Orbax `ocp.CheckpointManager` |

#### 关键实现

1. **SPMD Mesh**：3 轴 `('dp', 'fsdp', 'tp')`，DP 模式参数全复制 `P()`，FSDP 模式 2D kernel 沿 fsdp 轴分片 `P('fsdp', None)`
2. **Batch 分片**：`position_ids` 特殊处理（shape `(3, B, L)`，batch 在 axis=1：`P(None, 'dp', None)`）
3. **梯度累积**：`jax.lax.scan` 在 JIT 内循环累积，平均后 `apply_gradients`
4. **梯度检查点**：`nn.remat(DecoderLayer, policy=nothing_saveable)` 和 `nn.remat(VisionBlock)`
5. **Checkpoint**：Orbax `StandardSave`/`StandardRestore`，支持分片参数自动处理

#### TrainingArguments 字段

```python
gradient_accumulation_steps: int = 1    # micro-batch 累积数
gradient_checkpointing: bool = False    # nn.remat 激活重算
fsdp: bool = False                      # FSDP 模式（否则纯 DP）
max_checkpoints: int = 3               # 保留的 checkpoint 数量
resume_from_checkpoint: Optional[str]   # checkpoint 恢复路径
report_to: str = "none"                # "wandb", "tensorboard", "none"
run_name: str = ""                      # wandb/tensorboard run 名称
warmup_ratio: float = 0.0              # 若 > 0 且 warmup_steps == 0，则从 ratio 计算
```

---

### Layer 3 迁移（生产化功能）

**完成时间**：2026-02-07
**Agent**：Agent 7

#### 新建文件

| 文件 | 行数 | 内容 |
|---|---|---|
| `jax_qwenvl/model/weight_exporter.py` | 416 | LoRA 合并 + Flax→HF safetensors 导出（反向 key 映射 + 转置 + 分片保存） |
| `jax_qwenvl/train/metrics_logger.py` | 60 | wandb + tensorboard 统一日志封装（report_to="none" 时为 no-op） |
| `jax_qwenvl/scripts/train_tpu.sh` | 102 | TPU 训练启动脚本（环境变量覆盖所有参数） |

#### 关键实现

1. **LoRA 合并**：递归遍历参数树，检测 `{base, lora_A, lora_B}` 子结构，合并为 `kernel = base + (A @ B) * (alpha/rank)`，移除 LoRA 子键
2. **Flax → PyTorch key 映射**：反向执行 `weight_loader.py` 的映射规则（`blocks_N` → `blocks.N`，`kernel` → `weight`，vision norm `scale` → `weight`）
3. **反向转置**：Dense `(in,out)→(out,in)`，Conv3D `(T,H,W,in,out)→(out,in,T,H,W)`，embedding/norm/bias 不转置
4. **分片保存**：超过 `max_shard_size`（默认 5GB）时自动分片，生成 `model.safetensors.index.json`
5. **Packed causal mask**：从 `cu_seqlens`（1D cumsum）构建 segment_ids，然后 `same_segment & causal` 生成 block-diagonal mask
6. **数据 packing 集成**：`data_flatten=True` 或 `data_packing=True` 时使用 `FlattenedDataCollatorForSupervisedDataset`
7. **MetricsLogger**：lazy import wandb/tensorboard，`report_to="none"` 时完全无副作用
8. **训练后导出**：自动导出 HF safetensors + 保存 processor/tokenizer

---

## TPU 硬件验证记录

### 第一轮：float32（6/8 通过）

**日期**：2026-02-07
**环境**：v6e-4 spot (us-central1-b)，4 chips，runtime=v2-alpha-tpuv6e
**JAX**：0.6.2 + libtpu 0.0.17 + flax 0.10.7 + orbax-checkpoint 0.11.15
**模型**：Qwen/Qwen3-VL-2B-Instruct（2.1B 参数，float32）

| 测试 | 结果 | 详情 |
|------|------|------|
| 1. TPU 设备检测 | PASS | backend=tpu, 4 chips (topology 2x2) |
| 2. 模型权重加载 | PASS | 2,127,532,032 params, HuggingFace Hub 下载 |
| 3. 前向推理 | PASS | logits=(1,32,151936), no NaN/Inf, 5.4s |
| 4. SPMD 分片 (DP) | PASS | dp=4, fsdp=1, 参数复制到 4 chips |
| 5. 单步训练 | PASS | loss=10.2661, step=1, 29.6s |
| 6. Checkpoint | FAIL | OOM (float32 replicated 2B 模型占满 HBM) |
| 7. 梯度累积 | FAIL | OOM (需 32.35G，仅有 31.25G HBM/chip) |
| 8. FSDP 训练 | PASS | loss=10.2678, params sharded across 4 chips |

#### 发现的代码 Bug 及修复

1. **weight_loader.py key mapping**：Qwen3-VL safetensors 使用 `model.language_model.` 前缀（而非 `model.`），导致 text model 权重全部丢失。已修复，支持两种命名。
2. **weight_loader.py embed_tokens**：`embed_tokens` 需映射为直接叶节点（`self.param()` 而非 `nn.Embed`）。
3. **sharding.py FSDP 兼容**：参数第一维不能被 FSDP 设备数整除时（如 Conv3D shape=(2,16,16,3,1024)），自动回退到 replicated。
4. **sharding.py embed_tokens**：`shard_params` 的 kernel 检测增加 `embed_tokens` 匹配。
5. **config.py tie_word_embeddings**：Qwen3-VL-2B-Instruct 的 `tie_word_embeddings=True`，无独立 lm_head。
6. **TPU runtime**：v6e TPU 需使用 `v2-alpha-tpuv6e` runtime（而非 `tpu-ubuntu2204-base`）。

---

### 第二轮：bfloat16（8/8 通过）

**日期**：2026-02-07
**环境**：同上
**模型**：Qwen/Qwen3-VL-2B-Instruct（2.1B 参数，**bfloat16**）

| 测试 | 结果 | 详情 |
|------|------|------|
| 1. TPU 设备检测 | PASS | backend=tpu, 4 chips (topology 2x2) |
| 2. 模型权重加载 (bf16) | PASS | 2.1B params, dtype=bfloat16, memory=4.26GB |
| 3. 前向推理 (bf16) | PASS | logits=(1,32,151936), dtype=bfloat16, no NaN/Inf |
| 4. SPMD 分片 (DP) | PASS | dp=4, fsdp=1, dtype=bfloat16 |
| 5. 单步训练 (bf16) | PASS | loss=10.1919, opt_dtype=bfloat16, opt_mem=6.88GB |
| 6. Checkpoint | PASS | max_diff=0.0, save+restore 40s, dtype=bfloat16 |
| 7. 梯度累积 | PASS | loss=10.1919, accum_steps=2, 54.8s |
| 8. FSDP 训练 (bf16) | PASS | loss=10.2434, params sharded across 4 chips |

#### 内存对比（float32 vs bfloat16，DP 模式 per chip）

| 项目 | float32 | bfloat16 | 节省 |
|------|---------|----------|------|
| 参数 | 8.4GB | 4.2GB | 50% |
| Adam 优化器状态 | 16.8GB | 6.88GB | 59% |
| 总计 | ~25GB | ~11GB | 56% |
| 剩余 HBM (31.25GB/chip) | ~6GB | ~20GB | — |

#### 混合精度策略

bfloat16 训练中，以下操作保持 float32 以确保数值稳定性：
- `RMSNorm`：variance 计算在 float32，结果 cast 回 bf16
- `Softmax`（attention 和 loss）：在 float32 中计算
- `RoPE`：cos/sin 计算在 float32，结果 cast 回 bf16
- `cross_entropy_loss`：logits 和 log_softmax 在 float32 中计算

---

### 第三轮：端到端真实图片训练

**日期**：2026-02-07
**环境**：v6e-4 spot (us-central1-b)
**数据集**：LLaVA-Instruct-150K（157,712 样本，COCO train2017 图片）
**模型**：Qwen3-VL-2B-Instruct (bfloat16)
**配置**：per_device_batch=4, global_batch=16, model_max_length=1024, gradient_checkpointing=True

训练成功运行 7 步（修复前的 baseline，每步有 XLA 重编译）：

| Step | Loss | Step Time | Tokens/s |
|------|------|-----------|----------|
| 1 | 1.6721 | 77.92s | 57 |
| 2 | 1.7768 | 74.61s | 45 |
| 3 | 1.6847 | 85.19s | 54 |
| 4 | 1.5777 | 86.91s | 61 |
| 5 | 1.5763 | 106.41s | 45 |
| 6 | 1.6574 | 103.77s | 43 |
| 7 | 1.6925 | 88.30s | 48 |

**问题**：每步 60-106 秒，因为文本和视觉张量形状动态变化，导致每步都触发 XLA 重编译。

---

## XLA 重编译修复（~250x 加速）

### 问题分析

JAX/XLA 为每个唯一的张量形状编译一次计算图。训练中两类张量形状在 batch 间变化：

**视觉张量**（图片数量和尺寸随 batch 变化）：

| 张量 | 形状 | 变化原因 |
|------|------|----------|
| `pixel_values` | `(N, 6, 16, 16)` | N = 所有图片 patch 总数 |
| `image_grid_thw` | `(num_images, 3)` | 图片数量变化 |
| `image_pos_ids_2d` | `(N, 2)` | 同 pixel_values |
| `image_pos_ids_1d` | `(N,)` | 同 pixel_values |
| `image_cu_seqlens` | `(num_segments+1,)` | segment 数变化 |

**文本张量**（`_pad_sequence` 填充到 batch 内最大长度，非固定长度）：

| 张量 | 形状 | 变化原因 |
|------|------|----------|
| `input_ids` | `(B, max_len_in_batch)` | 每个 batch 内最长序列不同 |
| `labels` | `(B, max_len_in_batch)` | 同上 |
| `position_ids` | `(3, B, max_len_in_batch)` | 同上 |
| `attention_mask` | `(B, max_len_in_batch)` | 同上 |

### 解决方案

在 DataCollator 中将所有张量填充到固定形状：

#### 视觉张量固定形状填充

从 config 参数计算最大尺寸（无需扫描数据集）：

```
max_patches_per_image = max_pixels // (patch_size²) = 50176 // 256 = 196
max_total_patches     = batch_size × max_patches_per_image = 4 × 196 = 784
max_num_images        = batch_size  (假设每样本最多 1 张图)
```

`DataCollatorForSupervisedDataset` 新增 `_pad_vision_inputs()` 方法，将 `pixel_values`、`grid_thw`、`pos_ids_2d/1d`、`cu_seqlens` 填充到固定形状。

#### 文本张量固定形状填充

使用 `training_args.model_max_length`（而非 tokenizer 的 262K 默认值）作为固定文本长度。

**踩坑**：
- Qwen3-VL tokenizer 的 `model_max_length = 262144`（256K），用此值做填充会导致编译挂起
- `model_max_length = 8192` 导致 OOM（attention 矩阵 `f32[4,16,8192,8192]` = 16GB）
- `model_max_length = 1024` 适合 LLaVA 数据集（大部分对话 < 1024 tokens）

#### 修改的文件

| 文件 | 变更 |
|------|------|
| `jax_qwenvl/data/data_processor.py` | 两个 Collator 添加 `max_total_patches`/`max_num_images`/`model_max_length` 字段 + `_pad_vision_inputs()` 方法 + `pad_and_cat()` 支持 `max_length` 参数 + 文本填充到固定 `model_max_length` |
| `jax_qwenvl/train/train.py` | 从 config 计算 max 尺寸并传递给 Collator 构造函数 |

#### 安全性保证

填充输入不影响模型计算结果：

| 组件 | 为何安全 |
|------|----------|
| PatchEmbed3D | 逐 patch 独立处理，padding 0 → ~0 embedding |
| `_build_block_diagonal_mask` | padding tokens 形成独立 segment，不与真实 tokens 互相 attend |
| PatchMerger | reshape 分组，padding patches 合并为 padding merged tokens |
| `_scatter_embeddings` | cumsum+clip 只选前 `count(mask)` 个 embedding，padding embedding 不被使用 |

### 修复后验证结果

**环境**：v6e-4 (us-east5-b), Qwen3-VL-2B-Instruct (bf16)
**数据集**：LLaVA-Instruct-150K (157,712 samples)
**配置**：per_device_batch=4, global_batch=16, model_max_length=1024, max_total_patches=3136, max_num_images=16

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Step 1 (XLA 编译) | ~78s | ~78s |
| Step 2 | ~75s (重编译) | ~77s (第 2 次 trace) |
| Step 3+ | 60-106s (每步重编译) | **0.30s** |
| 吞吐量 | ~50 tokens/s | **~14,000 tokens/s** |
| 加速比 | — | **~250x** |

训练稳定运行 1087+ 步，loss 从 1.67 下降到 1.29，无 NaN/Inf。

```
Step 1:  loss=1.6721  step_time=77.55s  (XLA compilation)
Step 3:  loss=1.6847  step_time=0.30s   (no recompilation!)
Step 100: loss=1.4721  step_time=0.30s
Step 500: loss=1.3291  step_time=0.30s
Step 1087: loss=1.2138 step_time=0.30s
```

---

## TPU 运维指南

### TPU Runtime 版本选择（最关键）

v6e TPU **必须**使用 `v2-alpha-tpuv6e` runtime：

```bash
# 正确：v6e 专用 runtime
gcloud compute tpus tpu-vm create qwen3vl-test \
    --zone=us-central1-b \
    --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e \
    --spot

# 错误：通用 runtime，JAX 无法初始化 TPU backend
# --version=tpu-ubuntu2204-base
```

错误 runtime 的症状：
- `/dev/accel*` 设备文件不存在（v6e 使用 `/dev/vfio/0,1,2,3`）
- `jax.devices()` 报错 `Failed to get global TPU topology`

### 网络和防火墙配置

如果 TPU VM 在自定义 VPC 中：

```bash
# 指定网络和子网
gcloud compute tpus tpu-vm create qwen3vl-test \
    --zone=us-east5-b \
    --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e \
    --spot \
    --network=kube-vpc \
    --subnetwork=kube-vpc-us-east5

# SSH 方式 1：IAP tunnel（推荐）
gcloud compute tpus tpu-vm ssh qwen3vl-test --zone=us-east5-b --tunnel-through-iap

# SSH 方式 2：临时防火墙规则（测试用）
gcloud compute firewall-rules create allow-tpu-ssh-test \
    --network=kube-vpc \
    --allow=tcp:22 \
    --source-ranges=0.0.0.0/0
```

### 代码上传和环境准备

```bash
# 打包代码
tar czf /tmp/jax_qwenvl.tar.gz jax_qwenvl/

# 上传到 TPU VM
gcloud compute tpus tpu-vm scp /tmp/jax_qwenvl.tar.gz qwen3vl-test:~ --zone=us-east5-b

# 解压 + 安装依赖
gcloud compute tpus tpu-vm ssh qwen3vl-test --zone=us-east5-b \
    --command="tar xzf jax_qwenvl.tar.gz && pip install -r jax_qwenvl/requirements.txt"
```

### 清理资源

```bash
gcloud compute tpus tpu-vm delete qwen3vl-test --zone=us-east5-b --quiet
gcloud compute firewall-rules delete allow-tpu-ssh-test --quiet
```

### Package 依赖兼容性

#### 已验证的版本快照

```
jax==0.6.2
jaxlib==0.6.2
libtpu==0.0.17
flax==0.10.7
optax==0.2.5+
orbax-checkpoint==0.11.15
safetensors==0.5.x
transformers==4.51.x+
huggingface_hub==0.30.x
```

#### orbax-checkpoint 版本兼容性（踩坑重点）

与 JAX 0.6.2 兼容的版本**非常有限**：

| orbax-checkpoint 版本 | 兼容性 | 错误信息 |
|---|---|---|
| **0.11.15** | 可用 | — |
| 0.11.32 (最新) | 不可用 | `jax.sharding.set_mesh` 不是 context manager |
| 0.10.0 | 不可用 | `jax._src.config.enable_memories` 缺失 |
| 0.9.1 | 不可用 | `jax.lib.xla_extension.XlaRuntimeError` 已移除 |

### 常见问题排查

| 问题 | 症状 | 解决方案 |
|---|---|---|
| JAX 无法检测 TPU | `Failed to get global TPU topology` | 使用 `v2-alpha-tpuv6e` runtime 重建 VM |
| orbax checkpoint 崩溃 | `set_mesh` / `enable_memories` 错误 | 降级到 `orbax-checkpoint==0.11.15` |
| HuggingFace 下载失败 | 网络超时或 401 | 设置 `HF_TOKEN` 环境变量 |
| OOM（float32 2B 模型） | `RESOURCE_EXHAUSTED` | 使用 bfloat16 + gradient checkpointing |
| SSH 连接超时 | `Connection timed out` | 检查防火墙规则或使用 `--tunnel-through-iap` |
| Checkpoint async 错误 | `Array has been deleted` | 使用 `enable_async_checkpointing=False` |
| XLA 每步重编译 | step time 不下降（60-100s/步） | 检查所有输入张量是否填充到固定形状 |
| tokenizer model_max_length 过大 | 编译挂起或 OOM | 使用 training_args.model_max_length 而非 tokenizer 默认值 |

---

## 下一步：待完成工作

- MoE 模型支持（Expert Parallelism）
- 性能调优（XLA 编译优化、通信与计算重叠）
- Qwen2.5-VL 支持
- 推理/生成模式
- 多图/视频样本的固定形状填充（当前假设每样本最多 1 张图）
- 混合 text-only + vision batch 支持（当前要求每个 batch 都有图片）

---

## 参考资源

- MaxText (Google JAX LLM 训练框架): https://github.com/google/maxtext
- Optax (JAX 优化器库): https://github.com/google-deepmind/optax
- Orbax (JAX checkpoint 管理): https://github.com/google/orbax
- Grain (JAX 数据加载库): https://github.com/google/grain
- Pallas (JAX 自定义 TPU kernel): https://jax.readthedocs.io/en/latest/pallas/
- Flax NNX: https://flax.readthedocs.io/en/latest/
