# Qwen3-VL 项目笔记

> 所有的回答都要使用中文。

> **重要限制**：只允许修改 `CLAUDE.md`、`.claude/` 文件夹、`jax_qwenvl/` 文件夹内的文件。其他所有文件和文件夹（如 `qwen-vl-finetune/`、`qwen-vl-utils/`、`evaluation/`、`cookbooks/` 等）**禁止修改**。

## 快速参考

```bash
# 默认训练（cambrian_737k 数据集，2B 模型，纯 DP）
bash jax_qwenvl/scripts/train_tpu.sh

# LLaVA-Instruct-150K（推荐，已缓存于 GCS）
LLAVA_DATA_ROOT=~/llava_data DATASETS=llava_instruct_150k bash jax_qwenvl/scripts/train_tpu.sh

# 8B 模型 Hybrid 模式（v6e-16 推荐配置）
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
BATCH_SIZE=2 FSDP_DEVICES=4 \
GCS_OUTPUT_DIR=gs://bucket/qwen3vl-8b \
bash jax_qwenvl/scripts/train_tpu.sh

# 断点续训
RESUME_FROM_CHECKPOINT=True MAX_STEPS=200 \
GCS_OUTPUT_DIR=gs://bucket/qwen3vl/model \
bash jax_qwenvl/scripts/train_tpu.sh
```

### 训练脚本主要环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `Qwen/Qwen3-VL-2B-Instruct` | HF Hub ID 或本地路径，自动 `snapshot_download` |
| `DATASETS` | `cambrian_737k` | 逗号分隔，支持 `llava_instruct_150k` |
| `BATCH_SIZE` | `4` | 每设备 batch size |
| `GRAD_ACCUM` | `1` | 梯度累积步数 |
| `LR` | `2e-7` | 学习率 |
| `MAX_PIXELS` | `50176` | 图片最大像素数（多机建议 50176，避免 OOM） |
| `MODEL_MAX_LENGTH` | `1024` | 序列固定填充长度（影响 XLA 编译，勿用 tokenizer 默认 262K） |
| `FSDP` | `False` | 纯 FSDP 模式 |
| `FSDP_DEVICES` | `0` | >0 时启用 Hybrid 模式（v6e 推荐设为 4） |
| `GCS_OUTPUT_DIR` | `` | GCS 路径，自动上传模型 + checkpoint |
| `LOGGING_DIR` | `` | Tensorboard 日志目录（支持 `gs://`） |
| `SAVE_STEPS` | `1000` | Checkpoint 保存间隔 |
| `MAX_STEPS` | `-1` | -1 = 完整 epoch；续训时为总步数上限 |
| `RESUME_FROM_CHECKPOINT` | `` | 非空则从 `GCS_OUTPUT_DIR/checkpoints/` 恢复 |
| `REPORT_TO` | `none` | `none` / `wandb` / `tensorboard` |
| `LORA_ENABLE` | `False` | 启用 LoRA |

---

## 项目概览

Qwen3-VL 是阿里巴巴的多模态视觉语言模型仓库，支持图像理解、视频理解、OCR、UI 操控等能力。

### 目录结构

```
Qwen3-VL/
├── qwen-vl-utils/          # 视觉处理工具包（图像/视频加载、缩放）
├── qwen-vl-finetune/       # 微调框架（PyTorch/GPU 版，DeepSpeed ZeRO）
├── jax_qwenvl/              # JAX/TPU 训练框架（主要开发目标）
│   ├── types.py             # Batch, CausalLMOutput NamedTuple
│   ├── config.py            # token ID、shape 常量、视觉处理默认值
│   ├── requirements.txt     # TPU 训练依赖（锁定版本）
│   ├── data/
│   │   ├── __init__.py      # 数据集注册表
│   │   ├── rope2d.py        # 3 个 RoPE 函数（纯 numpy）
│   │   └── data_processor.py # Dataset + 2 个 Collator + 固定形状填充
│   ├── utils/
│   │   ├── gcs.py           # GCS 上传工具（google-cloud-storage SDK）
│   │   └── vision_process.py # 图像/视频加载（无 torch 依赖）
│   ├── model/
│   │   ├── config.py        # Qwen3VLConfig / VisionConfig / TextConfig
│   │   ├── layers.py        # RMSNorm, SwiGLUMLP, VisionMLP, LoRADense
│   │   ├── rope.py          # Vision RoPE (2D) + Text MRoPE (3D)
│   │   ├── vit.py           # PatchEmbed3D, VisionAttention, VisionBlock, PatchMerger
│   │   ├── llm.py           # TextAttention (GQA), DecoderLayer, TextModel
│   │   ├── qwen3_vl.py      # Qwen3VLForConditionalGeneration 组合模型
│   │   ├── weight_loader.py  # HF safetensors → Flax params
│   │   └── weight_exporter.py # Flax params → HF safetensors + LoRA 合并
│   ├── train/
│   │   ├── optimizer.py      # Optax 6+1 参数组 + warmup+cosine schedule
│   │   ├── train_step.py     # @jax.jit train_step + 梯度累积（jax.lax.scan）
│   │   ├── train.py          # 训练入口（参数解析、模型加载、训练循环）
│   │   ├── sharding.py       # SPMD mesh + DP/FSDP/Hybrid 分片
│   │   ├── checkpoint.py     # Orbax CheckpointManager（原生 GCS 多机支持）
│   │   └── metrics_logger.py # wandb + tensorboard 统一日志
│   └── scripts/
│       ├── train_tpu.sh      # TPU 训练启动脚本
│       ├── tpu_validate.py   # TPU 硬件验证脚本（8 项测试）
│       └── download_llava_data.sh # LLaVA 数据集下载
├── evaluation/              # 基准评测套件（VideoMME, MMMU, MathVision 等）
├── cookbooks/               # Jupyter 示例（OCR、grounding、agent 等）
└── web_demo_mm.py           # Gradio Web 演示界面
```

### 支持的模型

- Qwen2-VL / Qwen2.5-VL / Qwen3-VL (Dense) / Qwen3-VL-MoE
- 规模：2B、4B、8B、32B、30B-A3B（MoE）、235B-A22B（MoE）
- 版本：Instruct（标准指令跟随）和 Thinking（增强推理）

---

## JAX/TPU 框架关键架构

JAX/TPU 框架（`jax_qwenvl/`）从 PyTorch/GPU 完整迁移，零 `import torch` 依赖。

### 关键实现细节

- **固定形状填充**：所有输入张量必须填充到固定形状，否则每步触发 XLA 重编译（~250x 性能损失）
  - 视觉：`max_total_patches = batch_size × (max_pixels // patch_size²)`
  - 文本：填充到 `model_max_length`（训练参数值，非 tokenizer 默认 262K）
- **混合精度**：训练用 bfloat16；RMSNorm variance、Softmax、RoPE、cross-entropy 保持 float32
- **GQA**：通过 `jnp.repeat` 展开 K/V heads
- **cu_seqlens → block-diagonal mask**：替代 Flash Attention varlen，JAX 原生实现
- **LoRADense**：`lora_rank=0` 退化为普通 Dense；lora_B 初始化为 zeros
- **DeepStack**：ViT 在指定层提取中间特征注入 LLM 早期层（Qwen3-VL 特有）
- **GCS 依赖关系**：模型上传/TB 同步用 `google-cloud-storage` SDK；Orbax checkpoint 用 `gcsfs`（不可移除）

### 并行训练模式

| 模式 | 启动方式 | Mesh 示例 (v6e-16) | 稳态速度 | 适用场景 |
|------|---------|-------------------|---------|---------|
| DP（默认） | `bash train_tpu.sh` | dp=16, fsdp=1 | 0.67s | 小模型（2B） |
| FSDP | `FSDP=True` | dp=1, fsdp=16 | 0.35s | 大模型全分片 |
| **Hybrid（推荐）** | `FSDP_DEVICES=4` | dp=4, fsdp=4 | **0.38s** | 大模型（8B+），编译快 |

Hybrid 原则：`fsdp_devices` = 每 host 设备数（v6e 为 4），FSDP 通信限 host 内 ICI 高带宽。

### Checkpoint 架构

- Orbax `CheckpointManager` 直接指向 `gs://` 路径，原生多机协调（Orbax 0.11.32）
- 格式：tensorstore/zarr（非 msgpack）
- Checkpoint 存储于 `<gcs_output_dir>/checkpoints/` 子目录
- `_ensure_global()`：对 opt_state 标量（Adam count 等）选择性 re-shard（多机恢复时需要）

---

## PyTorch/GPU 微调（`qwen-vl-finetune/`）

| 方法 | 状态 |
|------|------|
| SFT（监督微调） | 支持（next-token prediction） |
| LoRA（q/k/v/o_proj） | 支持（`--lora_enable True`） |
| 全参数微调 | 支持（分组件控制） |
| DeepSpeed ZeRO-2/3 | 支持 |
| RLHF / DPO / QLoRA | 不支持 |

组件级调控：`--tune_mm_vision False --tune_mm_mlp True --tune_mm_llm True`
分组件学习率：`--mm_projector_lr` / `--vision_tower_lr` 可独立于 `--learning_rate`

---

## TPU 运维指南

### 创建 TPU VM

```bash
# v6e 必须使用 v2-alpha-tpuv6e runtime（否则 JAX 无法初始化）
gcloud compute tpus tpu-vm create qwen3vl-test \
    --zone=us-central1-b \
    --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e \
    --spot

# 自定义 VPC 时需指定网络
gcloud compute tpus tpu-vm create ... \
    --network=kube-vpc --subnetwork=kube-vpc-us-east5

# SSH（推荐 IAP tunnel）
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --tunnel-through-iap

# 多机（worker=all 同时操作所有 host）
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all --command='...'
```

### 数据集准备

```bash
# LLaVA-Instruct-150K（已缓存于 GCS，推荐从此下载）
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all \
    --command='mkdir -p ~/llava_data && gcloud storage cp -r gs://grhuang-02-vertex-ai/datasets/llava_data/* ~/llava_data/'
```

GCS 路径：`gs://grhuang-02-vertex-ai/datasets/llava_data/`（157,712 样本 + COCO train2017 ~118K 张）

### 环境准备

```bash
# 打包上传代码
tar czf /tmp/jax_qwenvl.tar.gz jax_qwenvl/
gcloud compute tpus tpu-vm scp /tmp/jax_qwenvl.tar.gz VM_NAME:~ --zone=ZONE --worker=all

# 安装 Python 3.11 + venv（JAX 0.9.0 要求 Python 3.11+）
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all --command='
tar xzf jax_qwenvl.tar.gz &&
sudo add-apt-repository -y ppa:deadsnakes/ppa &&
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev &&
python3.11 -m venv ~/venv311 && source ~/venv311/bin/activate &&
pip install "jax[tpu]==0.9.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&
pip install -r jax_qwenvl/requirements.txt'

source ~/venv311/bin/activate  # 训练前必须激活
```

### 清理资源

```bash
gcloud compute tpus tpu-vm delete VM_NAME --zone=ZONE --quiet
```

### 已验证依赖版本（当前）

```
Python==3.11.x  (venv on v2-alpha-tpuv6e)
jax==0.9.0 + jaxlib==0.9.0 + libtpu==0.0.34
flax==0.12.4
optax==0.2.6
orbax-checkpoint==0.11.32   # 必须 >=0.11.32（多机 GCS checkpoint）
numpy==2.3.5
safetensors==0.7.0
transformers==5.1.0
google-cloud-storage>=2.19.0
gcsfs>=2024.0.0             # Orbax 内部依赖，不可移除
```

---

## 常见问题排查

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| JAX 无法检测 TPU | `Failed to get global TPU topology` | 使用 `v2-alpha-tpuv6e` runtime 重建 VM |
| orbax checkpoint 崩溃 | `set_mesh` / `enable_memories` 错误 | 升级到 JAX 0.9.0 + orbax 0.11.32 |
| JAX 0.7.0+ 安装失败 | `No matching distribution found` | runtime 只有 Python 3.10；需安装 Python 3.11 venv |
| TPU 被占用 | `The TPU is already in use by pid XXX` | `pkill -9 -u $(whoami) python3` |
| HuggingFace 下载失败 | 网络超时或 401 | 设置 `HF_TOKEN` 环境变量 |
| OOM（float32 模型） | `RESOURCE_EXHAUSTED` | 使用 bfloat16 + gradient checkpointing |
| XLA 每步重编译 | step time 不下降（60-100s/步） | 检查所有输入张量是否固定形状；尤其 `model_max_length` |
| tokenizer max_length 过大 | 编译挂起或 OOM | 使用 `--model_max_length 1024`，勿用 tokenizer 默认 262K |
| `max_pixels` 过大（多机） | attention matrix OOM | 设 `MAX_PIXELS=50176`（默认 451584 在 batch=64 时 → 406GB） |
| Checkpoint 多机死锁 | barrier hang / `Timed out waiting for array_metadatas` | 用 JAX 0.9.0 + Orbax 0.11.32 + `gs://` 路径（已默认启用） |
| Checkpoint 恢复后 loss 跳回初始 | restored state 未在正确 device mesh | `_ensure_global()` 选择性 re-shard 已内置；检查 GCS 路径一致 |
| Checkpoint resume 多机 `incompatible devices` | opt_state 标量在单 host 设备上 | 已由 `_ensure_global()` 自动处理（`8d4a002`） |
| Orbax GCS 报 `ImportError: gcsfs` | `Please install gcsfs` | `pip install gcsfs>=2024.0.0`（已在 requirements.txt） |
| TPU watchdog 超时（大模型分片） | `TpuSyncFlagReadCallbackWatchdog expired` | 已用 batched `jax.device_put` 修复；8B ~325s 正常 |
| Weight export 磁盘空间不足 | `No space left on device` | 设置 `GCS_OUTPUT_DIR`，逐 shard 流式上传（`50b9f1b`） |
| Weight export shutdown barrier 超时 | `DEADLINE_EXCEEDED: Barrier timed out` | 已用 `sync_global_devices("weight_export_done")` 修复 |
| transformers 5.x 图片加载 | `Incorrect padding` (base64 decode) | 用 `PIL.Image.open()` 预加载，不传路径字符串 |
| Tensorboard 未同步到 GCS | GCS 下无 `tensorboard/` 子目录 | 设置 `GCS_OUTPUT_DIR` 后自动推导 `logging_dir`（`d164312`） |
| SSH 连接超时 | `Connection timed out` | 检查防火墙规则或使用 `--tunnel-through-iap` |

---

## Checkpoint 断点续训

```bash
# max_steps 为总步数上限（含已完成步数）
# 例：从 step 100 恢复 + max_steps=200 = 再跑 100 步
RESUME_FROM_CHECKPOINT=True \
MAX_STEPS=200 \
GCS_OUTPUT_DIR=gs://bucket/qwen3vl/model \
bash jax_qwenvl/scripts/train_tpu.sh
```

注意：
- 数据从 epoch 开头重新迭代（deterministic seed），optimizer state 正确恢复保证参数空间连续
- GCS 路径须与原训练相同；checkpoint 在 `<gcs_output_dir>/checkpoints/` 子目录

---

## 已验证训练性能

| 配置 | 硬件 | 步时 | 吞吐 | 最终损失 |
|------|------|------|------|---------|
| 2B DP (dp=16) | v6e-16, 4h | 0.68s | ~25k tok/s | avg_loss=1.27 (2464 步) |
| 2B Hybrid (dp=4, fsdp=4) | v6e-16 | 0.38s | ~45k tok/s | — |
| 8B Hybrid (dp=4, fsdp=4) | v6e-16, 4h | 0.69s | ~13k tok/s | avg_loss=1.79 (4928 步) |

8B 内存占用（每设备）：参数 4GB + Adam 8GB + 梯度 4GB + 激活 1GB ≈ 17GB / 31.25GB HBM

---

## 下一步：待完成工作

- MoE 模型支持（Expert Parallelism）
- 性能调优（XLA 编译优化、通信与计算重叠）
- Qwen2.5-VL 支持
- 推理/生成模式
- 多图/视频样本的固定形状填充（当前假设每样本最多 1 张图）
- 混合 text-only + vision batch 支持（当前要求每个 batch 都有图片）
- 视觉模型 DP 分片（当前视觉模型在所有设备上复制，浪费计算）

---

## 参考资源

- MaxText (Google JAX LLM 训练框架): https://github.com/google/maxtext
- Optax (JAX 优化器库): https://github.com/google-deepmind/optax
- Orbax (JAX checkpoint 管理): https://github.com/google/orbax
- Grain (JAX 数据加载库): https://github.com/google/grain
- Pallas (JAX 自定义 TPU kernel): https://jax.readthedocs.io/en/latest/pallas/
- Flax NNX: https://flax.readthedocs.io/en/latest/

> 详细迁移历史、每步训练数据、各阶段验证结果请见 `jax_qwenvl/RESEARCH_LOG.md`
> 功能路线图与下一步规划请见 `jax_qwenvl/ROADMAP.md`
