# Qwen3-VL 项目笔记 (Gemini 版)

> 所有的回答都要使用中文。

> **重要限制**：只允许修改 `CLAUDE.md`、`.claude/` 文件夹、`.gemini/` 文件夹、`jax_qwenvl/` 文件夹内的文件。其他所有文件和文件夹（如 `qwen-vl-finetune/`、`qwen-vl-utils/`、`evaluation/`、`cookbooks/` 等）**禁止修改**。

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

# ViT 视觉数据分片训练（2026-04-03 验证）
SHARD_VISION_BATCH=1 MAX_PIXELS=345744 BATCH_SIZE=1 \
DATASETS=llava_instruct_150k \
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
| `MAX_PIXELS` | `50176` | 图片最大像素数（多机建议 50176，避免 OOM；大分辨率测试可用 345744） |
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
| `SHARD_VISION_BATCH` | `0` | 是否启用 ViT 视觉数据分片（1 为分片，0 为复制） |

---

## 项目概览

Qwen3-VL 是阿里巴巴的多模态视觉语言模型仓库，支持图像理解、视频理解、OCR、UI 操控等能力。

### 目录结构

```
Qwen3-VL/
├── .gemini/                 # Gemini 专用指令与文档
│   └── GEMINI.md            # 本文件
├── jax_qwenvl/              # JAX/TPU 训练框架（主要开发目标）
│   ├── RESEARCH_LOG.md      # 详细实验日志与验证数据
│   └── ROADMAP.md           # 功能路线图与优先级规划
```
*(详细目录结构请参考 `CLAUDE.md`，此处仅列出 Gemini 相关及核心路径)*

---

## JAX/TPU 框架关键架构

JAX/TPU 框架（`jax_qwenvl/`）从 PyTorch/GPU 完整迁移，零 `import torch` 依赖。

### 关键实现细节

- **固定形状填充**：所有输入张量必须填充到固定形状，否则每步触发 XLA 重编译（~250x 性能损失）
- **混合精度**：训练用 bfloat16；RMSNorm variance、Softmax、RoPE、cross-entropy 保持 float32
- **ViT 视觉数据分片**：通过 `SHARD_VISION_BATCH` 支持在多设备间分片视觉输入，节省显存，但在小 Batch Size 下可能因通信开销导致速度略慢。

### 并行训练模式

| 模式 | 启动方式 | Mesh 示例 (v6e-16) | 稳态速度 | 适用场景 |
|------|---------|-------------------|---------|---------|
| DP（默认） | `bash train_tpu.sh` | dp=16, fsdp=1 | 0.67s | 小模型（2B） |
| FSDP | `FSDP=True` | dp=1, fsdp=16 | 0.35s | 大模型全分片 |
| **Hybrid（推荐）** | `FSDP_DEVICES=4` | dp=4, fsdp=4 | **0.38s** | 大模型（8B+），编译快 |

---

## 常见问题排查

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| **大分辨率 OOM** | `RESOURCE_EXHAUSTED` (超出 10G+) | `BATCH_SIZE=2` 配合 `MAX_PIXELS=345744` 时，注意力矩阵过大导致 OOM。需降低分辨率或减小 Batch Size。 |
| **分片比复制慢** | 稳态 Step Time 变长 | 在小 Batch Size (如 BS=1) 下，分片带来的通信开销超过了计算节省，属正常现象。 |

*(更多通用问题请参考 `CLAUDE.md`)*

---

## 已完成功能（Gemini 验证）

- ✅ **ViT 视觉数据分片支持**：通过 `SHARD_VISION_BATCH` 成功实现并验证了分片与复制模式的性能对比（2026-04-03）。
- ✅ **大模型分布式稳定性**：验证了 8B 模型在 4 节点 TPU Pod 上的稳定训练。

## 下一步：待完成工作

- 推理/生成模式（P0）
- Qwen2.5-VL 完整支持（P2）
- 训练验证管道（P3）
- TP（张量并行）实现（P1）
- 进一步优化大分辨率下的显存占用（如引入重算或更细粒度的分片）

---
> 详细迁移历史、每步训练数据、各阶段验证结果请见 `jax_qwenvl/RESEARCH_LOG.md`
> 功能路线图与下一步规划请见 `jax_qwenvl/ROADMAP.md`
