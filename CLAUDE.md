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
│   │   ├── gcs.py           # GCS 上传工具（google-cloud-storage SDK）
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
│   │   ├── sharding.py       # SPMD mesh + DP/FSDP/Hybrid 分片
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
| `8500f81` | JAX 0.6.2 → 0.9.0 升级：Python 3.11+ venv，XLA 编译加速 35% |
| `ed78fc1` | Orbax 原生 GCS checkpoint：删除 bypass 代码，StandardRestore 自动 re-shard |
| `8d4a002` | 修复多机 checkpoint resume opt_state sharding（v6e-16 验证通过） |
| `204b57c` | GCS 上传改用 google-cloud-storage SDK，替代 gcloud CLI subprocess 调用 |
| `f1f768a` | 修复最后一步 checkpoint 重复保存 warning + 恢复 gcsfs 依赖（Orbax 需要）|
| `1d3425f` | 修复 FSDP batch 分片 + 多机 weight export（FSDP 2x faster than DP on v6e-16）|
| (pending) | 混合 DP+FSDP 模式：`fsdp_devices` 参数，`mode='hybrid'`，`P(('dp','fsdp'))` batch 分片 |

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
| `jax_qwenvl/train/checkpoint.py` | 86 | `CheckpointManager` 封装 Orbax（GCS 原生多机支持） |

#### 关键实现

1. **SPMD Mesh**：3 轴 `('dp', 'fsdp', 'tp')`，DP 模式参数全复制 `P()`，FSDP/Hybrid 模式 2D kernel 沿 fsdp 轴分片 `P('fsdp', None)`，Hybrid 模式 batch 沿 `('dp', 'fsdp')` 双轴分片
2. **Batch 分片**：`position_ids` 特殊处理（shape `(3, B, L)`，batch 在 axis=1：`P(None, 'dp', None)`）
3. **梯度累积**：`jax.lax.scan` 在 JIT 内循环累积，平均后 `apply_gradients`
4. **梯度检查点**：`nn.remat(DecoderLayer, policy=nothing_saveable)` 和 `nn.remat(VisionBlock)`
5. **Checkpoint**：Orbax `CheckpointManager` 直接指向 `gs://` 路径，原生多机协调（JAX 0.9.0 + Orbax 0.11.32），无需手动 GCS 同步或 re-shard

#### TrainingArguments 字段

```python
gradient_accumulation_steps: int = 1    # micro-batch 累积数
gradient_checkpointing: bool = False    # nn.remat 激活重算
fsdp: bool = False                      # FSDP 模式（否则纯 DP）
fsdp_devices: int = 0                  # FSDP 轴设备数（0=用 fsdp bool；>0 启用显式 dp/fsdp 分割，支持混合模式）
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
| Checkpoint 多机死锁 | `Timed out waiting for array_metadatas` 或 barrier hang | 使用 JAX 0.9.0 + Orbax 0.11.32，`CheckpointManager` 指向 `gs://` 路径（已默认启用） |
| transformers 5.x 图片加载 | `Incorrect padding` (base64 decode) | 在 `_build_messages()` 中用 `PIL.Image.open()` 预加载图片，不传路径字符串 |
| XLA 每步重编译 | step time 不下降（60-100s/步） | 检查所有输入张量是否填充到固定形状 |
| tokenizer model_max_length 过大 | 编译挂起或 OOM | 使用 training_args.model_max_length 而非 tokenizer 默认值 |
| Checkpoint 恢复后 loss 跳回初始值 | restored state 未正确放到 device mesh | Orbax `StandardRestore` 使用 template sharding 恢复大数组；opt_state 标量需 `_ensure_global()` 选择性 re-shard（已内置） |
| Checkpoint resume 多机 `incompatible devices` | opt_state 标量（Adam count）在单 host 设备上 | `_ensure_global()` 检查 `len(x.devices()) == global_device_count`，不足的转 numpy + replicate（已内置） |
| 最后一步 checkpoint 重复保存 | `Final checkpoint save failed: Checkpoint for step N already exists` | `max_steps` 恰好是 `save_steps` 的倍数时，训练循环已保存 checkpoint，post-loop 重复保存。已修复：检查 `latest_step() != global_step`（`f1f768a`） |
| Orbax GCS checkpoint 报 `ImportError: gcsfs` | `Please install gcsfs to access Google Storage` | Orbax 通过 `etils/epath` → `fsspec` → `gcsfs` 访问 GCS 路径，需要 `gcsfs` 依赖（已加回 requirements.txt） |

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
| `jax_qwenvl/train/metrics_logger.py` | 替换 `flax.metrics.tensorboard` 为 `torch.utils.tensorboard`，GCS 路径写入本地临时目录 + `finish()` 时通过 google-cloud-storage SDK 同步 |
| `jax_qwenvl/train/checkpoint.py` | Orbax `CheckpointManager` 直接指向 `gs://` 路径，原生多机协调，无手动 GCS 同步 |
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

6. **GCS tensorboard 不支持追加写入**：`torch.utils.tensorboard.SummaryWriter` 不支持 GCS 追加模式。解决方案：写入本地临时目录，定期和训练结束后通过 `google-cloud-storage` SDK 同步到 GCS

7. **Spot TPU 随时可能被驱逐**：us-central1-b 多次被驱逐。建议开启 checkpoint 保存以支持断点续训

8. **Orbax 原生 GCS checkpoint**：`--gcs_output_dir gs://bucket/path` 参数启用后，`CheckpointManager` 直接指向 `gs://.../path/checkpoints/`，所有 host 通过 GCS 读写 checkpoint，无需 `gcloud` CLI 或手动同步。Orbax 使用 tensorstore/zarr 格式（比 msgpack 更高效）

9. **GCS 模型自动上传**：训练完成后自动上传 safetensors + json + jinja 模型文件到 `gcs_output_dir`（仅 process 0，使用 `google-cloud-storage` SDK）

---

## Checkpoint 断点续训

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

1. `CheckpointManager.restore(state_template=state)` — Orbax 从 `gs://.../checkpoints/` 读取最新 step，使用 template 的 sharding 恢复到正确设备
2. `_ensure_global(opt_state)` — 检查 opt_state 每个叶节点是否在所有 global devices 上；不足的（如 Adam `count` 标量）转为 numpy 后 `device_put` 到 replicated sharding
3. `global_step = int(state.step)` — 恢复训练步数计数器
4. 训练循环从 `global_step` 继续，`max_steps` 控制总步数上限

### Opt_state sharding 问题（多机模式）

**问题**：Orbax `StandardRestore` 使用 template 的 sharding 正确恢复大数组（params、Adam mu/nu），但 opt_state 标量（如 Adam `count`）在 template 中没有显式 global sharding，恢复后可能只在单个 host 的设备上，导致 `train_step` 报 `ValueError: Received incompatible devices for jitted computation`。

**修复尝试**：
1. ~~`jax.device_put(x, replicate)` 直接 re-shard~~ — 多机失败：local array 只有 4 devices，global sharding 需要 16 devices → `CopyArrays only supports destination device list of the same size`
2. ~~`jax.device_put(np.asarray(x), replicate)` 全部转 numpy~~ — OOM：大数组（mu/nu ~2.7GB each）转 numpy 后 `process_allgather` 需要额外 2.3GB buffer，超出 HBM
3. **`_ensure_global()` 选择性 re-shard** — 只对 `len(x.devices()) < global_device_count` 的叶节点转 numpy + replicate，跳过已正确分片的大数组

**Commit**：`8d4a002`

### 验证结果（v6e-16）

**日期**：2026-02-10
**环境**：v6e-16 spot (asia-northeast1-b)，4 hosts × 4 chips = 16 chips
**配置**：per_device_batch=4, global_batch=64, max_steps=30, save_steps=10
**GCS 路径**：`gs://grhuang-02-vertex-ai/qwen3vl-orbax-test/model/checkpoints/`

#### Test 1：训练 20 步 + checkpoint 保存

| 指标 | 结果 |
|------|------|
| Checkpoint step 10 | PASS — 所有 4 hosts 参与，~77s |
| Checkpoint step 20 | PASS — 所有 4 hosts 参与，~74s |
| avg_loss (20 步) | 1.6697 |
| Step time (稳态) | 0.67s |
| Throughput | ~25,000 tokens/s |
| 格式 | Orbax tensorstore/zarr（非 msgpack） |
| 存储 | GCS 原生读写，无 `gcloud` CLI |

#### Test 2：从 checkpoint 恢复到 step 30

| 指标 | 结果 |
|------|------|
| 恢复日志 | "Resumed from step 20"、"Resuming from global_step=20" |
| 训练范围 | step 21 → step 30 |
| Loss 连续性 | 1.6640 → 1.5902（未跳回初始值） |
| avg_loss (10 步) | 1.6588 |
| Checkpoint step 30 | PASS — 保存到 GCS，~80s |
| 模型导出 | HF safetensors 自动上传到 GCS |

### 注意事项

- **数据顺序**：恢复后数据从 epoch 开头重新迭代（deterministic seed），不会精确跟原训练对齐。optimizer state 正确恢复保证训练从正确的参数空间点继续
- **`max_steps` 语义**：表示训练的**总步数上限**（包括已完成的步数），不是恢复后再跑的步数。从 step 100 恢复 + `max_steps=200` = 再跑 100 步
- **GCS 路径**：`gcs_output_dir` 同时用于 checkpoint 保存和恢复，确保恢复训练时使用与原训练相同的 GCS 路径。Checkpoint 存储在 `<gcs_output_dir>/checkpoints/` 子目录下
- **单机兼容**：无 GCS 配置时 checkpoint 保存到本地 `output_dir`
- **Opt_state re-shard**：`StandardRestore` 正确恢复大数组（params、mu、nu）的 sharding，但 opt_state 标量需要 `_ensure_global()` 选择性 re-shard（仅多机模式需要）

---

## JAX 0.6.2 → 0.9.0 升级

### 升级动机

- Orbax 0.11.15 多机 checkpoint 死锁（已通过升级到 Orbax 0.11.32 + GCS 路径解决）
- JAX 0.9.0 的 Shardy 分区器 XLA 编译更快
- Orbax 0.11.32 多机 checkpoint 原生支持（已启用，使用 `gs://` 路径）

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
| `flax.serialization.to_bytes/from_bytes` | 不再使用（Orbax 原生格式替代） |
| `optax.chain/adamw/clip_by_global_norm` | 无变化 |

---

## Orbax 原生 GCS Checkpoint 迁移

### 概述

**日期**：2026-02-10
**Commit**：`ed78fc1`（迁移）、`8d4a002`（多机 opt_state 修复）

升级到 JAX 0.9.0 + Orbax 0.11.32 后，将 checkpoint 系统从 process-0-only `flax.serialization` bypass 迁移到 Orbax 原生 `CheckpointManager` 直接指向 `gs://` 路径。v6e-16 (4 hosts) 验证通过：checkpoint 保存/恢复/续训均正常。

### 改动

| 文件 | 变更 |
|------|------|
| `jax_qwenvl/train/checkpoint.py` | 完全重写：86 行替代 294 行。删除 `_save_multihost`、`_restore_multihost`、`download_from_gcs`、`_sync_to_gcs`、`_cleanup_old`、`_latest_step_multihost` 及单机/多机分支逻辑 |
| `jax_qwenvl/train/train.py` | 简化 resume：删除 `download_from_gcs()` + 删除手动 `shard_params`。新增 `_ensure_global()` 选择性 re-shard opt_state 标量（多机模式下 Adam count 等标量可能不在全局设备上） |

### 删除的代码

| 方法/功能 | 行数 | 原因 |
|-----------|------|------|
| `_save_multihost()` | ~40 | Orbax 原生多机保存 |
| `_restore_multihost()` | ~20 | Orbax `StandardRestore` 替代 |
| `download_from_gcs()` | ~40 | Orbax 直接读 GCS |
| `_sync_to_gcs()` | ~20 | Orbax 直接写 GCS |
| `_cleanup_old()` | ~15 | Orbax `max_to_keep` 管理 |
| `_latest_step_multihost()` | ~10 | Orbax `latest_step()` 替代 |
| 单机/多机分支逻辑 | ~30 | 统一代码路径 |
| `train.py` 手动全量 re-shard | ~10 | 替换为 `_ensure_global()` 选择性 re-shard（仅标量） |

### 关键变化

1. **Checkpoint 路径**：`gcs_dir` 提供时，checkpoint 存储在 `<gcs_dir>/checkpoints/` 子目录（避免与模型导出文件冲突）
2. **存储格式**：从 msgpack（`flax.serialization`）变为 tensorstore/zarr（Orbax 原生格式）
3. **多机协调**：所有 host 通过 GCS 直接读写，Orbax 内部处理同步，无需 `gcloud` CLI
4. **恢复几乎无需 re-shard**：`StandardRestore` 使用 state template 的 sharding 恢复大数组（params、mu、nu）；仅 opt_state 标量需 `_ensure_global()` 选择性处理
5. **公共 API 不变**：`save()`、`restore()`、`latest_step()`、`should_save()`、`wait_for_completion()` 接口和参数完全相同

---

## GCS 上传改用 google-cloud-storage SDK

### 概述

**日期**：2026-02-11
**Commit**：`204b57c`

训练后的模型上传（`train.py` 的 `_upload_to_gcs()`）和 tensorboard 日志同步（`metrics_logger.py` 的 `_sync_to_gcs()`）原先通过 `subprocess.run(["gcloud", "storage", "cp", ...])` 和 `subprocess.run(["gsutil", ...])` 调用外部 CLI。改为使用 `google-cloud-storage` Python SDK 原生上传，消除对外部 CLI 工具的依赖。

### 改动

| 文件 | 变更 |
|------|------|
| `jax_qwenvl/utils/gcs.py` | 新建：`upload_files_to_gcs(local_dir, gcs_uri, extensions)` 按扩展名过滤上传 + `sync_dir_to_gcs(local_dir, gcs_uri)` 全目录同步 |
| `jax_qwenvl/utils/__init__.py` | 新增导出 `upload_files_to_gcs`, `sync_dir_to_gcs` |
| `jax_qwenvl/train/train.py` | `_upload_to_gcs()` 改用 `upload_files_to_gcs()` 替代 subprocess |
| `jax_qwenvl/train/metrics_logger.py` | `_sync_to_gcs()` 改用 `sync_dir_to_gcs()` 替代 gcloud/gsutil 双 fallback |
| `jax_qwenvl/requirements.txt` | 新增 `google-cloud-storage>=2.19.0`，保留 `gcsfs>=2024.0.0`（Orbax 依赖） |
| `jax_qwenvl/train/train.py` | 修复最后一步 checkpoint 重复保存：检查 `latest_step() != global_step` |

### GCS 依赖关系

```
模型上传 / TB 同步:  google-cloud-storage SDK  (我们的代码)
Orbax checkpoint:    gcsfs → fsspec → etils   (Orbax 内部)
```

- `google-cloud-storage`：用于 `_upload_to_gcs()`（模型文件）和 `_sync_to_gcs()`（tensorboard 日志），替代 gcloud/gsutil CLI
- `gcsfs`：Orbax `CheckpointManager` 通过 `etils/epath` → `fsspec` → `gcsfs` 访问 GCS checkpoint 路径，不可移除

### 验证结果

**日期**：2026-02-11
**Commit**：`204b57c`（SDK 迁移）、`f1f768a`（bug 修复 + gcsfs 恢复）
**环境**：v6e-16 spot (us-central1-b)，4 hosts × 4 chips = 16 chips
**数据集**：LLaVA-Instruct-150K
**配置**：per_device_batch=4, global_batch=64, model_max_length=1024, max_pixels=50176

#### Phase 1：训练 20 步 + GCS 上传

| 指标 | 结果 |
|------|------|
| 训练 | 20 步完成，avg_loss=1.6586 |
| Step time（稳态） | 0.65s |
| Throughput | ~25,000-27,000 tokens/s |
| Checkpoint step 10 | PASS — Orbax 原生写 GCS |
| Checkpoint step 20 | PASS — Orbax 原生写 GCS |
| 模型上传 | PASS — **7 files via google-cloud-storage SDK** |
| TB 日志同步 | PASS — **1 file via google-cloud-storage SDK** |
| Final checkpoint warning | 无（bug 已修复） |

#### Phase 2：从 step 20 恢复 + 继续到 step 40

| 指标 | 结果 |
|------|------|
| 恢复 | "Resumed from step 20" |
| 训练 | step 21→40，avg_loss=1.5453 |
| Loss 连续性 | 1.6218→1.5234（未跳回初始值） |
| Checkpoint step 30/40 | PASS |
| 模型上传 | PASS — **7 files via SDK** |
| TB 日志同步 | PASS — **1 file via SDK** |
| Final checkpoint warning | 无 |

#### GCS Artifacts（`gs://grhuang-02-vertex-ai/qwen3-vl/`）

| 路径 | 内容 |
|------|------|
| `*.safetensors` + `*.json` + `*.jinja` | 7 个模型文件（~7.9 GB） |
| `checkpoints/20/`, `30/`, `40/` | Orbax tensorstore/zarr 格式 |
| `tensorboard/` | 2 个 event 文件 |

---

## FSDP 模式修复与验证

### 概述

**日期**：2026-02-11
**Commit**：`1d3425f`
**环境**：TPU v6e-16 spot (us-central1-b)，4 hosts × 4 chips = 16 chips
**数据集**：LLaVA-Instruct-150K
**模型**：Qwen3-VL-2B-Instruct (bfloat16)
**配置**：per_device_batch=4, global_batch=64, model_max_length=1024, max_pixels=50176

### 修复的 Bug

1. **`shard_batch()` 硬编码 `'dp'` 轴**：FSDP 模式下 `dp=1, fsdp=N`，`P('dp')` 等于 `P()`（复制），batch 从未被分片。修复：添加 `mode` 参数，FSDP 模式使用 `'fsdp'` 轴
2. **`batch_size` 计算错误**：使用 `mesh.shape['dp']`（FSDP 下为 1），导致 global_batch=4 而非 64。修复：根据 mode 选择 `mesh.shape['fsdp']` 或 `mesh.shape['dp']`
3. **`export_hf_weights` 多机死锁**：仅 process 0 调用，但 `process_allgather` 是集合操作需所有进程参与。修复：所有进程调用 `export_hf_weights`，通过 `is_main_process` 参数控制仅 process 0 写文件
4. **`create_device_mesh` 缺 `fsdp=-1` 自动填充**：修复：添加 `elif fsdp == -1` 分支

### 改动文件

| 文件 | 变更 |
|------|------|
| `jax_qwenvl/train/sharding.py` | `shard_batch()` + `_shard_batch_multihost()` 添加 `mode` 参数；`create_device_mesh` 添加 `fsdp=-1` |
| `jax_qwenvl/train/train.py` | `batch_size` 计算使用 `num_data_devices`；`shard_batch()` 传 `mode=sharding_mode`；`export_hf_weights` 所有进程调用 |
| `jax_qwenvl/model/weight_exporter.py` | `_flatten_params` 添加 `process_allgather(v, tiled=True)` 处理 FSDP 分片数组；`export_hf_weights` 添加 `is_main_process` 参数 |

### FSDP vs DP 对比（v6e-16, 16 chips）

| 指标 | FSDP (`dp=1, fsdp=16`) | DP (`dp=16, fsdp=1`) |
|------|----------------------|---------------------|
| XLA 编译 (step 1) | 132s | 65s |
| XLA 编译 (step 2) | 140s | 67s |
| 稳态 step time | **0.35s** | 0.67s |
| 稳态 throughput | **~49,000 tok/s** | ~25,000 tok/s |
| avg_loss (20 步) | 1.6704 | 1.6697 |

FSDP 模式稳态速度 **2x 于 DP 模式**。原因：每设备仅存储 1/16 参数和优化器状态，内存压力更低，XLA 可更高效分配计算资源。

### 验证结果

**Phase 1: FSDP 训练 20 步**
- `Parameters sharded with mode=fsdp` ✓
- `num_data_devices=16, global_batch=64` ✓
- Checkpoint at step 10, 20 ✓
- Weight export (safetensors) ✓
- GCS upload (7 files) ✓

**Phase 2: Checkpoint 恢复 (step 20→40)**
- `Resumed from step 20` ✓
- 训练从 step 21 继续到 step 40 ✓
- Loss 连续性 ✓（未跳回初始值）
- Checkpoint at step 30, 40 ✓
- 旧 checkpoint (step 10) 自动删除 ✓ (`max_checkpoints=3`)
- Weight export + GCS upload ✓

---

## 混合 DP+FSDP 模式

### 概述

**日期**：2026-02-12

新增 `fsdp_devices` 参数，支持混合 DP+FSDP 并行模式（`mode='hybrid'`）。此前训练只支持纯 DP (`dp=N, fsdp=1`) 或纯 FSDP (`dp=1, fsdp=N`)。混合模式 (e.g. `dp=4, fsdp=4`) 将 FSDP 通信限制在 host 内高带宽 ICI 连接，跨 host 只做 DP 的 allreduce，适合大模型 (8B+) 在大 pod slice (v6e-64+) 上训练。

### 三种并行模式

| 模式 | 示例 (v6e-16) | 参数分片 | Batch 分片 | 适用场景 |
|------|--------------|---------|-----------|---------|
| DP | `dp=16, fsdp=1` | 全复制 `P()` | `P('dp')` | 小模型，每设备可放完整参数 |
| FSDP | `dp=1, fsdp=16` | `P('fsdp', None)` | `P('fsdp')` | 大模型，需跨所有设备分片 |
| Hybrid | `dp=4, fsdp=4` | `P('fsdp', None)` | `P(('dp', 'fsdp'))` | 大模型 + 大 pod，FSDP 限 host 内 |

### 使用方法

```bash
# 混合模式：FSDP 在每 host 的 4 chips 内，DP 跨 4 hosts
FSDP_DEVICES=4 bash jax_qwenvl/scripts/train_tpu.sh

# 纯 FSDP（向后兼容）
FSDP=True bash jax_qwenvl/scripts/train_tpu.sh

# 纯 DP（向后兼容，默认）
bash jax_qwenvl/scripts/train_tpu.sh
```

### `fsdp_devices` 参数优先级

| 配置 | 结果 mesh | mode |
|------|----------|------|
| `fsdp_devices=4` on 16 devices | `dp=4, fsdp=4` | `hybrid` |
| `fsdp_devices=16` on 16 devices | `dp=1, fsdp=16` | `fsdp` |
| `fsdp_devices=1` on 16 devices | `dp=16, fsdp=1` | `dp` |
| `fsdp_devices=0, fsdp=True` | `dp=1, fsdp=16` | `fsdp` |
| `fsdp_devices=0, fsdp=False` | `dp=16, fsdp=1` | `dp` |

### 改动文件

| 文件 | 变更 |
|------|------|
| `jax_qwenvl/train/train.py` | 添加 `fsdp_devices: int = 0`；mesh 创建三分支逻辑；`num_data_devices = dp * fsdp`（统一公式） |
| `jax_qwenvl/train/sharding.py` | `get_param_sharding_rules` 支持 `mode='hybrid'`（同 fsdp）；`shard_batch` / `_shard_batch_multihost` 支持 `mode='hybrid'` → `P(('dp','fsdp'))` |
| `jax_qwenvl/scripts/train_tpu.sh` | 添加 `FSDP_DEVICES` 环境变量和 `--fsdp_devices` 传参 |

### 关键设计

1. **Batch 分片 `P(('dp', 'fsdp'))`**：JAX 标准语法，表示该维度同时在 dp 和 fsdp 两个轴上分片，总分片数 = dp_size × fsdp_size
2. **参数分片与纯 FSDP 相同**：hybrid 模式下参数仍沿 fsdp 轴分片 `P('fsdp', None)`，dp 轴上每组独立复制
3. **`num_data_devices = dp * fsdp`**：统一公式替代 if-else，对三种模式均正确（DP: N×1, FSDP: 1×N, Hybrid: M×K）
4. **XLA 自动推导通信**：`train_step` 无需修改，XLA 从 PartitionSpec 自动推导：fsdp 组内 all-gather/reduce-scatter 参数，dp 组间 all-reduce 梯度

### Checkpoint / Weight Export 兼容性

| 组件 | hybrid 模式行为 | 是否改动 |
|------|----------------|---------|
| Checkpoint save | Orbax 序列化 `P('fsdp', None)` 数组，与纯 FSDP 相同 | 无 |
| Checkpoint restore | `StandardRestore` 使用 template sharding，`_ensure_global()` 对任意 mesh 有效 | 无 |
| Weight export | `fsdp=local_device_count` 时每 host 已有完整分片，跳过 `process_allgather`；fsdp 跨 host 时走 `process_allgather` | 无 |
| train_step | XLA 自动插入 fsdp all-gather + dp all-reduce | 无 |

### 推荐配置

| TPU 类型 | 设备数 | hosts | 推荐 hybrid 配置 | 说明 |
|----------|-------|-------|-----------------|------|
| v6e-16 | 16 | 4×4 | `FSDP_DEVICES=4` → dp=4, fsdp=4 | FSDP 在 host 内，DP 跨 host |
| v6e-32 | 32 | 8×4 | `FSDP_DEVICES=4` → dp=8, fsdp=4 | 同上 |
| v6e-64 | 64 | 16×4 | `FSDP_DEVICES=4` → dp=16, fsdp=4 | 同上 |
| v6e-64 | 64 | 16×4 | `FSDP_DEVICES=16` → dp=4, fsdp=16 | FSDP 跨 4 hosts（更大模型） |

**原则**：`fsdp_devices` 设为每 host 的设备数（v6e 为 4），FSDP 通信限制在 host 内 ICI 高带宽连接。仅当模型太大无法放入单 host 时才增大 `fsdp_devices`。

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
- ~~Orbax 原生 GCS checkpoint~~（已完成：删除 bypass 代码，Orbax 0.11.32 直接写 GCS，StandardRestore 自动 re-shard）
- 视觉模型 DP 分片（当前视觉模型在所有设备上复制，浪费计算）
- ~~FSDP 模式修复~~（已完成：batch 分片轴修复 + batch_size 计算修复 + 多机 weight export 修复，v6e-16 验证 FSDP 2x faster than DP）
- ~~混合 DP+FSDP 模式~~（已完成：`fsdp_devices` 参数，`mode='hybrid'`，`P(('dp','fsdp'))` batch 分片，待 TPU 验证）

---

## 参考资源

- MaxText (Google JAX LLM 训练框架): https://github.com/google/maxtext
- Optax (JAX 优化器库): https://github.com/google-deepmind/optax
- Orbax (JAX checkpoint 管理): https://github.com/google/orbax
- Grain (JAX 数据加载库): https://github.com/google/grain
- Pallas (JAX 自定义 TPU kernel): https://jax.readthedocs.io/en/latest/pallas/
- Flax NNX: https://flax.readthedocs.io/en/latest/
