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
| `7525c60` | CLAUDE.md 重命名 + 迁移记录更新 |
| `1b1f516` | 多机训练支持（v6e-16, 4 hosts），LLaVA 完整 1 epoch 验证 |
| `dafa706` | Tensorboard GCS 定期同步 + 固定 orbax-checkpoint 版本 |
| `7839edf` | 修复 Orbax async checkpoint 多机崩溃 + GCS 模型/checkpoint 自动上传 |
| `512a846` | 修复多机 checkpoint 死锁（bypass Orbax）+ transformers 5.x 图片加载 + max_steps |
| `21428c7` | Checkpoint 断点续训：re-shard restored state + 恢复 global_step + GCS 路径修复 |
| `TBD` | JAX 0.6.2 → 0.9.0 升级：Python 3.11+ venv，XLA 编译加速 35% |

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
| `jax_qwenvl/train/checkpoint.py` | 107 | `CheckpointManager` 封装 Orbax（async 禁用 + GCS 同步） |

#### 关键实现

1. **SPMD Mesh**：3 轴 `('dp', 'fsdp', 'tp')`，DP 模式参数全复制 `P()`，FSDP 模式 2D kernel 沿 fsdp 轴分片 `P('fsdp', None)`
2. **Batch 分片**：`position_ids` 特殊处理（shape `(3, B, L)`，batch 在 axis=1：`P(None, 'dp', None)`）
3. **梯度累积**：`jax.lax.scan` 在 JIT 内循环累积，平均后 `apply_gradients`
4. **梯度检查点**：`nn.remat(DecoderLayer, policy=nothing_saveable)` 和 `nn.remat(VisionBlock)`
5. **Checkpoint**：单机用 Orbax `CheckpointManager`（sync 模式），多机用 process-0-only `flax.serialization`（Orbax 0.11.15 多机 barrier 死锁，需共享文件系统），支持 GCS 同步

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
gcs_output_dir: Optional[str] = None   # GCS 路径，自动上传模型和 checkpoint
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
| `jax_qwenvl/scripts/train_tpu.sh` | 117 | TPU 训练启动脚本（环境变量覆盖所有参数，含 GCS_OUTPUT_DIR） |

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

### 数据集准备

LLaVA-Instruct-150K 数据集已缓存在 GCS，从 GCS 下载比从源站（HuggingFace + COCO）快得多：

```bash
# 从 GCS 下载（推荐，所有 worker 同时执行）
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all \
    --command='mkdir -p ~/llava_data && gcloud storage cp -r gs://grhuang-02-vertex-ai/datasets/llava_data/* ~/llava_data/'

# 训练时设置环境变量
LLAVA_DATA_ROOT=~/llava_data DATASETS=llava_instruct_150k bash jax_qwenvl/scripts/train_tpu.sh
```

GCS 数据集路径：`gs://grhuang-02-vertex-ai/datasets/llava_data/`
- `llava_instruct_150k.json` — 标注文件（157,712 样本）
- `train2017/` — COCO train2017 图片（~118K 张）

如需从源站重新下载：`bash jax_qwenvl/scripts/download_llava_data.sh ~/llava_data`

### 代码上传和环境准备

```bash
# 打包代码
tar czf /tmp/jax_qwenvl.tar.gz jax_qwenvl/

# 上传到 TPU VM（所有 worker）
gcloud compute tpus tpu-vm scp /tmp/jax_qwenvl.tar.gz VM_NAME:~ --zone=ZONE --worker=all

# 解压 + 安装 Python 3.11 + 创建 venv + 安装依赖（所有 worker）
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all --command='
tar xzf jax_qwenvl.tar.gz && \
sudo add-apt-repository -y ppa:deadsnakes/ppa && \
sudo apt-get update -qq && \
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev && \
python3.11 -m venv ~/venv311 && \
source ~/venv311/bin/activate && \
pip install --upgrade pip && \
pip install "jax[tpu]==0.9.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
pip install -r jax_qwenvl/requirements.txt
'

# 训练时需先激活 venv
source ~/venv311/bin/activate
```

### 清理资源

```bash
gcloud compute tpus tpu-vm delete qwen3vl-test --zone=us-east5-b --quiet
gcloud compute firewall-rules delete allow-tpu-ssh-test --quiet
```

### Package 依赖兼容性

#### 已验证的版本快照（当前）

```
Python==3.11.14 (venv on v2-alpha-tpuv6e)
jax==0.9.0
jaxlib==0.9.0
libtpu==0.0.34
flax==0.12.3
optax==0.2.7
orbax-checkpoint==0.11.32
numpy==2.3.5
scipy==1.17.0
safetensors==0.7.0
transformers==5.1.0
huggingface_hub==1.4.1
```

#### 历史版本快照（JAX 0.6.2，已废弃）

```
jax==0.6.2
jaxlib==0.6.2
libtpu==0.0.17
flax==0.10.7
optax==0.2.5+
orbax-checkpoint==0.11.15
```

#### orbax-checkpoint 版本兼容性

| orbax-checkpoint 版本 | JAX 0.6.2 | JAX 0.9.0 | 错误信息 |
|---|---|---|---|
| 0.9.1 | 不可用 | — | `XlaRuntimeError` 已移除 |
| 0.10.0 | 不可用 | — | `enable_memories` 缺失 |
| **0.11.15** | 可用 | 未测试 | — |
| **0.11.32** | 不可用 | **可用** | JAX 0.6.2: `set_mesh` 不是 context manager |

### 常见问题排查

| 问题 | 症状 | 解决方案 |
|---|---|---|
| JAX 无法检测 TPU | `Failed to get global TPU topology` | 使用 `v2-alpha-tpuv6e` runtime 重建 VM |
| orbax checkpoint 崩溃 | `set_mesh` / `enable_memories` 错误 | 升级到 JAX 0.9.0 + orbax 0.11.32（或降级到 orbax 0.11.15 + JAX 0.6.2） |
| JAX 0.7.0+ 安装失败 | `No matching distribution found` | TPU runtime Python 3.10 不支持 JAX 0.7.0+，需安装 Python 3.11 并创建 venv |
| TPU 被占用 | `The TPU is already in use by process with pid XXX` | `sudo kill -9 PID` 或 `pkill -9 -u $(whoami) python3` |
| HuggingFace 下载失败 | 网络超时或 401 | 设置 `HF_TOKEN` 环境变量 |
| OOM（float32 2B 模型） | `RESOURCE_EXHAUSTED` | 使用 bfloat16 + gradient checkpointing |
| SSH 连接超时 | `Connection timed out` | 检查防火墙规则或使用 `--tunnel-through-iap` |
| Checkpoint async 错误（单机） | `Array has been deleted` | 使用 `enable_async_checkpointing=False`（已默认禁用） |
| Checkpoint 多机死锁 | `Timed out waiting for array_metadatas` 或 barrier hang | Orbax 0.11.15 多机需共享文件系统（GCS/NFS）。当前方案：bypass Orbax，process-0-only 用 flax.serialization 保存 |
| transformers 5.x 图片加载 | `Incorrect padding` (base64 decode) | 在 `_build_messages()` 中用 `PIL.Image.open()` 预加载图片，不传路径字符串 |
| XLA 每步重编译 | step time 不下降（60-100s/步） | 检查所有输入张量是否填充到固定形状 |
| tokenizer model_max_length 过大 | 编译挂起或 OOM | 使用 training_args.model_max_length 而非 tokenizer 默认值 |
| Checkpoint 恢复后 loss 跳回初始值 | restored state 未 re-shard 到 device mesh | 确保恢复后调用 `shard_params()` + `jax.device_put(opt_state, replicate)` |
| GCS checkpoint 路径嵌套 | `gs://bucket/100/100/state.msgpack` | 使用 `cp -r src/* dst/` 而非 `cp -r src dst/` |
| 多机恢复找不到 checkpoint | 其他 host 本地无文件 | 恢复前调用 `download_from_gcs()`（已自动集成） |

---

## 多机训练（Multi-Host TPU Pod Slice）

### 概述

**日期**：2026-02-09
**环境**：TPU v6e-16 spot (asia-northeast1-b)，4 hosts × 4 chips = 16 chips
**数据集**：LLaVA-Instruct-150K（157,712 样本，COCO train2017）
**模型**：Qwen3-VL-2B-Instruct (bfloat16)
**配置**：per_device_batch=4, global_batch=64, model_max_length=1024, max_pixels=50176, gradient_checkpointing=True

### 训练结果

完整 1 epoch 训练（2464 步），总耗时 ~61 分钟：

| Step | Loss | Avg Loss | Step Time | Tokens/s |
|------|------|----------|-----------|----------|
| 1 | 1.6649 | 1.6649 | 98.89s | 179 (XLA 编译) |
| 2 | 1.6652 | 1.6650 | 101.50s | 173 (第2次 trace) |
| 3 | 1.6337 | 1.6546 | **0.68s** | **25,993** |
| 100 | 1.3062 | 1.4538 | 0.68s | 26,070 |
| 500 | 1.2799 | 1.3198 | 0.68s | 26,707 |
| 1000 | 1.2693 | 1.2923 | 0.68s | 26,221 |
| 1500 | 1.2933 | 1.2814 | 0.68s | 24,436 |
| 2000 | 1.2624 | 1.2757 | 0.68s | 24,972 |
| 2464 | 1.2339 | **1.2727** | 0.68s | 24,977 |

### 与单机 (v6e-4) 对比

| 指标 | v6e-4 (4 chips) | v6e-16 (16 chips) |
|------|-----------------|-------------------|
| Global batch size | 16 | 64 |
| Step time (after compilation) | 0.30s | 0.68s |
| Throughput (tokens/s) | ~14,000 | ~25,000 |
| 吞吐量提升 | — | **1.8x** |

### 代码修改

| 文件 | 变更 |
|------|------|
| `jax_qwenvl/train/train.py` | `jax.distributed.initialize()`（try-except 包裹），`output_dir` 强制绝对路径，`is_main_process` 守卫日志/保存/导出，`logging_dir` 参数，`gcs_output_dir` 参数 + `_upload_to_gcs()` 训练后自动上传模型 |
| `jax_qwenvl/train/sharding.py` | `shard_batch()` 自动检测多机（`process_count > 1`），`_shard_batch_multihost()` 使用 `host_local_array_to_global_array` |
| `jax_qwenvl/train/metrics_logger.py` | 替换 `flax.metrics.tensorboard` 为 `torch.utils.tensorboard`，GCS 路径写入本地临时目录 + `finish()` 时 gsutil 同步 |
| `jax_qwenvl/train/checkpoint.py` | `enable_async_checkpointing=False` 修复多机崩溃，`gcs_dir` 参数 + `_sync_to_gcs()` checkpoint 自动同步 |
| `jax_qwenvl/scripts/train_tpu.sh` | 新增 `GCS_OUTPUT_DIR` 环境变量 |

### 多机训练注意事项

1. **所有 worker 必须同时运行训练脚本**：使用 `--worker=all` 参数
   ```bash
   gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all --command='...'
   ```

2. **`jax.distributed.initialize()` 必须调用**：Orbax 多机 checkpoint 需要显式初始化分布式系统

3. **Orbax checkpoint 需要绝对路径**：相对路径在多机模式下会报错
   ```
   ValueError: Checkpoint path should be absolute. Got output_v6e16/...
   ```

4. **`max_pixels` 必须控制**：视觉注意力矩阵 `(num_heads, N, N)` 与 `max_total_patches` 的平方成正比
   - `max_pixels=451584`（默认）+ batch=64 → `max_total_patches=112896` → OOM（406 GB attention matrix）
   - `max_pixels=50176` + batch=64 → `max_total_patches=12544` → OK（~5 GB attention matrix）

5. **单机兼容性**：所有多机修改向后兼容单机模式
   - `jax.distributed.initialize()` 在 try-except 中，单机失败时安静跳过
   - `shard_batch()` 通过 `process_count() > 1` 自动切换路径
   - `MetricsLogger` 非主进程使用 `report_to="none"` 无操作

6. **GCS tensorboard 不支持追加写入**：`torch.utils.tensorboard.SummaryWriter` 需要 `gcsfs`，但 GCS 不支持追加模式。解决方案：写入本地临时目录，训练结束后 gsutil 同步到 GCS

7. **Spot TPU 随时可能被驱逐**：us-central1-b 多次被驱逐。建议开启 checkpoint 保存以支持断点续训

8. **Orbax 0.11.15 多机 checkpoint 死锁**：Orbax 的 `CheckpointManager` 和 `StandardCheckpointer` 在多机模式下都会死锁。根本原因：Orbax 要求 checkpoint 目录对所有 host 可见（GCS 或 NFS），但本地路径 `~/output` 只有本机能看到。解决方案：
   - **当前**：bypass Orbax，process-0-only 用 `jax.device_get()` + `flax.serialization.to_bytes()` 保存（DP 模式每个 host 有完整参数副本）
   - **推荐长期方案**：使用 `gs://` 路径作为 Orbax checkpoint 目录（MaxText 的做法），这样所有 host 都能看到目录
   - 多机 checkpoint 每次 ~11 GB，耗时 15-55 秒（msgpack 序列化 + 磁盘写入）

9. **GCS 模型/checkpoint 自动上传**：通过 `--gcs_output_dir gs://bucket/path` 参数启用：
   - 每次 checkpoint 保存后自动同步到 `gs://.../path/<step>/`（仅 process 0）
   - 训练完成后自动上传 safetensors + json + jinja 模型文件到 GCS
   - 使用 `gcloud storage cp` 上传，失败时 warning 不中断训练

---

## Checkpoint 断点续训

### 修复的问题

**日期**：2026-02-10
**Commit**：`21428c7`

| 问题 | 原因 | 修复 |
|------|------|------|
| `global_step` 恢复后从 0 开始 | `train.py` 硬编码 `global_step = 0`，未读取 `state.step` | 从 `state.step` 恢复 `global_step` |
| 恢复的 state 未分片到 device mesh | `flax.serialization.from_bytes` 返回 numpy 数组，不在 TPU 上 | 恢复后 `shard_params()` + `jax.device_put(replicate)` |
| GCS checkpoint 路径双重嵌套 | `gcloud storage cp -r /path/100 gs://bucket/100/` → `gs://bucket/100/100/state.msgpack` | 改为 `cp -r /path/100/* gs://bucket/100/` 上传目录内容 |
| 多机恢复时其他 host 无 checkpoint | checkpoint 仅存在于 process 0 本地 + GCS | 新增 `download_from_gcs()` 方法，所有 host 在 restore 前从 GCS 下载 |
| `train_tpu.sh` 无 resume 支持 | 缺少环境变量 | 新增 `RESUME_FROM_CHECKPOINT` 环境变量 |

### 代码修改

| 文件 | 变更 |
|------|------|
| `jax_qwenvl/train/checkpoint.py` | 修复 `_sync_to_gcs()` 路径嵌套（`src/*` 而非 `src`）+ 新增 `download_from_gcs()` 方法（扫描 GCS 最新 step → 下载到本地） |
| `jax_qwenvl/train/train.py` | 恢复后 re-shard params（`shard_params`）+ replicate opt_state（`jax.device_put(x, NamedSharding(mesh, P()))`）+ 从 `state.step` 恢复 `global_step` + 恢复前调用 `download_from_gcs()` |
| `jax_qwenvl/scripts/train_tpu.sh` | 新增 `RESUME_FROM_CHECKPOINT` 环境变量 |

### 使用方法

```bash
# 从 GCS checkpoint 恢复训练，继续跑到 step 200
RESUME_FROM_CHECKPOINT=True \
MAX_STEPS=200 \
SAVE_STEPS=10 \
GCS_OUTPUT_DIR=gs://bucket/qwen3vl/model \
bash jax_qwenvl/scripts/train_tpu.sh
```

### 恢复流程

1. `CheckpointManager.download_from_gcs()` — 扫描 GCS 目录找最新 step，下载 `state.msgpack` 到本地（所有 host）
2. `CheckpointManager.restore(state_template=state)` — 用 `flax.serialization.from_bytes` 反序列化（返回 numpy 数组）
3. `shard_params(restored.params, mesh, rules)` — 将 params 放到 device mesh（DP 模式复制，FSDP 模式分片）
4. `jax.device_put(opt_state, NamedSharding(mesh, P()))` — opt_state 复制到所有设备
5. `global_step = int(state.step)` — 恢复训练步数计数器
6. 训练循环从 `global_step` 继续，`max_steps` 控制总步数上限

### 注意事项

- **数据顺序**：恢复后数据从 epoch 开头重新迭代（deterministic seed），不会精确跟原训练对齐。optimizer state 正确恢复保证训练从正确的参数空间点继续
- **`max_steps` 语义**：表示训练的**总步数上限**（包括已完成的步数），不是恢复后再跑的步数。从 step 100 恢复 + `max_steps=200` = 再跑 100 步
- **GCS 路径**：`gcs_output_dir` 同时用于 checkpoint 上传和下载，确保恢复训练时使用与原训练相同的 GCS 路径
- **单机兼容**：所有修改向后兼容单机模式。无 GCS 配置时 `download_from_gcs()` 为无操作

---

## JAX 0.6.2 → 0.9.0 升级

### 升级动机

- Orbax 0.11.15 多机 checkpoint 死锁（当前 bypass Orbax 方案可用但非最优）
- JAX 0.9.0 的 Shardy 分区器 XLA 编译更快
- Orbax 0.11.32 多机 checkpoint 原生支持（需 GCS 路径，后续优化）

### 升级步骤

1. **requirements.txt 版本更新**：`jax==0.9.0`, `flax>=0.12.0`, `orbax-checkpoint>=0.11.32`
2. **Python 3.11 安装**：JAX 0.7.0+ 要求 Python 3.11+，v2-alpha-tpuv6e runtime 只有 Python 3.10
   ```bash
   sudo add-apt-repository -y ppa:deadsnakes/ppa
   sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
   python3.11 -m venv ~/venv311
   source ~/venv311/bin/activate
   ```
3. **训练代码**：零修改，所有 API 向后兼容

### 验证结果

**日期**：2026-02-10
**环境**：v6e-16 spot (asia-northeast1-b)，4 hosts × 4 chips = 16 chips
**配置**：同多机训练配置（per_device_batch=4, global_batch=64, max_steps=20）

| 指标 | JAX 0.6.2 | JAX 0.9.0 | 变化 |
|------|-----------|-----------|------|
| XLA 编译 (step 1-2) | ~100s | ~65s | **35% 加速** |
| 稳态 step time | 0.68s | 0.67s | 持平 |
| Throughput | ~25,000 tok/s | ~25,000 tok/s | 持平 |
| avg_loss (20步) | 1.6649 | 1.6697 | 一致 |

### API 兼容性确认

| API | 状态 |
|-----|------|
| `jax.jit`, `jax.value_and_grad` | 无变化 |
| `jax.sharding.{Mesh, NamedSharding, PartitionSpec}` | 无变化 |
| `jax.distributed.initialize()` | 无变化 |
| `jax.experimental.multihost_utils.host_local_array_to_global_array` | 仍可用 |
| `jax.tree_util.tree_map_with_path` | 无变化 |
| `flax.linen` (nn.Module, nn.Dense, nn.remat 等) | 无变化（Linen API 已冻结） |
| `flax.serialization.to_bytes/from_bytes` | 无变化 |
| `optax.chain/adamw/clip_by_global_norm` | 无变化 |

---

## 下一步：待完成工作

- MoE 模型支持（Expert Parallelism）
- 性能调优（XLA 编译优化、通信与计算重叠）
- Qwen2.5-VL 支持
- 推理/生成模式
- 多图/视频样本的固定形状填充（当前假设每样本最多 1 张图）
- 混合 text-only + vision batch 支持（当前要求每个 batch 都有图片）
- ~~多机 checkpoint 到 GCS~~（已完成：`gcs_output_dir` 参数支持 checkpoint + 模型自动上传）
- ~~Checkpoint 断点续训~~（已完成：re-shard restored state + 恢复 global_step + GCS 下载 + 路径修复）
- 视觉模型 DP 分片（当前视觉模型在所有设备上复制，浪费计算）

---

## 参考资源

- MaxText (Google JAX LLM 训练框架): https://github.com/google/maxtext
- Optax (JAX 优化器库): https://github.com/google-deepmind/optax
- Orbax (JAX checkpoint 管理): https://github.com/google/orbax
- Grain (JAX 数据加载库): https://github.com/google/grain
- Pallas (JAX 自定义 TPU kernel): https://jax.readthedocs.io/en/latest/pallas/
- Flax NNX: https://flax.readthedocs.io/en/latest/
