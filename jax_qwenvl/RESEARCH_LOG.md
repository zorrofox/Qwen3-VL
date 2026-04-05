# Qwen3-VL JAX/TPU 迁移研究日志

> 本文件记录项目详细历史：commit 记录、逐步验证数据、各功能实现细节。
> 日常参考请见 CLAUDE.md。

---

## Commit 历史

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
| `5b83ef0` | 混合 DP+FSDP 模式：`fsdp_devices` 参数，`mode='hybrid'`，`P(('dp','fsdp'))` batch 分片 |
| `20a8862` | 混合 DP+FSDP 验证结果文档化（7/7 测试全部通过 on v6e-16） |
| `cd1c2d1` | 8B 模型 hybrid 验证：HF Hub 自动下载 + batched shard_params（0.69s/step, ~13k tok/s） |
| `50b9f1b` | 8B weight export 修复：SDK 流式上传 + sync_global_devices 防止 shutdown barrier 超时 |
| `2aa753e` | CLAUDE.md 更新：8B 完整 1 epoch 训练结果（4928 步, avg_loss=1.7937, 32.67 GiB → GCS） |
| `d164312` | 修复 tensorboard 日志未同步到 GCS：当 `gcs_output_dir` 设置时自动推导 `logging_dir` |

---

## Layer 0 迁移（CPU 端数据预处理）

**完成时间**：2026-02-07
**Agent**：Agent 1 (RoPE) + Agent 2 (数据管道) + Agent 3 (视觉工具) 并行

| 文件 | 行数 | 说明 |
|------|------|------|
| `jax_qwenvl/__init__.py` | 1 | 包入口 |
| `jax_qwenvl/types.py` | 30 | `Batch`、`CausalLMOutput` NamedTuple，所有字段 `np.ndarray` |
| `jax_qwenvl/config.py` | 31 | token ID、shape 常量、视觉处理默认值 |
| `jax_qwenvl/data/__init__.py` | 65 | 数据集注册表（纯 Python 配置） |
| `jax_qwenvl/data/rope2d.py` | 494 | 3 个 RoPE 函数（`get_rope_index_3/25/2`），纯 numpy |
| `jax_qwenvl/data/data_processor.py` | 727+ | Dataset + 2 个 Collator，返回 `Batch` NamedTuple |
| `jax_qwenvl/utils/vision_process.py` | 492 | 图像/视频加载，无 torch/torchvision 依赖 |

关键设计决策：
1. **纯 NumPy**：Layer 0 是 CPU 端预处理，用纯 numpy 避免动态 shape 问题
2. **`_ensure_numpy()`**：duck typing 处理 HF processor 返回 torch tensor 的情况
3. **删除 torchvision 视频后端**：decord 为主、torchcodec 为备（lazy import）
4. **`_pad_sequence()` 自实现**：替代 `torch.nn.utils.rnn.pad_sequence()`
5. **Collator 返回 `Batch` NamedTuple**：确保下游 `jax.device_put()` 兼容

---

## Layer 1 迁移（模型定义 + 训练基础设施）

**完成时间**：2026-02-07
**Agent**：Agent 4 (模型定义) + Agent 5 (训练循环)

### 模型定义（8 个文件）

| 文件 | 行数 | 内容 |
|---|---|---|
| `jax_qwenvl/model/config.py` | 156 | Qwen3VLConfig / VisionConfig / TextConfig dataclass，含 `from_pretrained` |
| `jax_qwenvl/model/layers.py` | 105 | RMSNorm, SwiGLUMLP, VisionMLP, LoRADense |
| `jax_qwenvl/model/rope.py` | 191 | Vision RoPE（2D 空间位置查找），Text MRoPE（3D 交错频率） |
| `jax_qwenvl/model/vit.py` | 427 | PatchEmbed3D, VisionAttention, VisionBlock, PatchMerger, VisionModel |
| `jax_qwenvl/model/llm.py` | 318 | TextAttention（GQA + q/k_norm）, DecoderLayer, TextModel |
| `jax_qwenvl/model/qwen3_vl.py` | 301 | Qwen3VLForConditionalGeneration, cross_entropy_loss |
| `jax_qwenvl/model/weight_loader.py` | 308 | HF safetensors → Flax params 转换 |

### 训练基础设施（5 个文件）

| 文件 | 行数 | 内容 |
|---|---|---|
| `jax_qwenvl/train/optimizer.py` | 349 | optax.multi_transform 6+1 参数组，warmup+cosine schedule |
| `jax_qwenvl/train/train_state.py` | 45 | 扩展 Flax TrainState |
| `jax_qwenvl/train/train_step.py` | 156 | @jax.jit train_step + train_step_with_accumulation (jax.lax.scan) |
| `jax_qwenvl/train/train.py` | 487+ | 训练入口：参数解析、模型加载、权重合并、训练循环 |

### 关键架构实现

1. **Vision RoPE**：2D 空间位置查找表 + flatten + duplicate，匹配 HF `rot_pos_emb` 方法
2. **Text MRoPE**：3D 交错频率布局 `[THWTHW...]`，匹配 HF `apply_interleaved_mrope`
3. **DeepStack**：ViT 在指定层（如 [8,16,24]）提取中间特征，注入 LLM 早期层
4. **q_norm / k_norm**：Qwen3 特有的 per-head RMSNorm（在 projection 后、RoPE 前）
5. **GQA**：通过 `jnp.repeat` 展开 K/V heads 到 Q heads 数量
6. **3D Conv PatchEmbed**：Flax `nn.Conv` 实现 `(temporal_patch_size, patch_size, patch_size)` 3D 卷积
7. **cu_seqlens → block-diagonal mask**：替代 Flash Attention varlen，用 JAX 实现
8. **LoRADense**：内联 LoRA（lora_rank=0 时退化为普通 Dense），初始化 lora_B=zeros

---

## Layer 2 迁移（分布式训练 + Checkpoint）

**完成时间**：2026-02-07，**Agent**：Agent 6

关键实现：
1. **SPMD Mesh**：3 轴 `('dp', 'fsdp', 'tp')`，DP 参数全复制 `P()`，FSDP/Hybrid 2D kernel 分片 `P('fsdp', None)`
2. **Batch 分片**：`position_ids` shape `(3, B, L)`，batch 在 axis=1：`P(None, 'dp', None)`
3. **梯度累积**：`jax.lax.scan` 在 JIT 内循环累积，平均后 `apply_gradients`
4. **梯度检查点**：`nn.remat(DecoderLayer, policy=nothing_saveable)` 和 `nn.remat(VisionBlock)`

---

## Layer 3 迁移（生产化功能）

**完成时间**：2026-02-07，**Agent**：Agent 7

关键实现：
1. **LoRA 合并**：`kernel = base + (A @ B) * (alpha/rank)`，递归遍历参数树
2. **Flax → PyTorch key 映射**：`blocks_N` → `blocks.N`，`kernel` → `weight`，Dense `(in,out)→(out,in)`
3. **分片保存**：超过 `max_shard_size`（默认 5GB）时自动分片

---

## TPU 硬件验证记录

### 第一轮：float32（6/8 通过，2026-02-07）

**环境**：v6e-4 spot (us-central1-b)，JAX 0.6.2，Qwen3-VL-2B-Instruct (float32)

| 测试 | 结果 | 详情 |
|------|------|------|
| 1. TPU 设备检测 | PASS | backend=tpu, 4 chips (topology 2x2) |
| 2. 模型权重加载 | PASS | 2,127,532,032 params |
| 3. 前向推理 | PASS | logits=(1,32,151936), no NaN/Inf, 5.4s |
| 4. SPMD 分片 (DP) | PASS | dp=4, fsdp=1 |
| 5. 单步训练 | PASS | loss=10.2661 |
| 6. Checkpoint | FAIL | OOM（float32 replicated 2B 模型占满 HBM） |
| 7. 梯度累积 | FAIL | OOM（需 32.35G，仅有 31.25G HBM/chip） |
| 8. FSDP 训练 | PASS | loss=10.2678 |

发现并修复的 Bug：
- `weight_loader.py`：Qwen3-VL safetensors 使用 `model.language_model.` 前缀（非 `model.`），已支持两种命名
- `sharding.py`：参数第一维不能被 FSDP 设备数整除时，自动回退到 replicated
- `config.py`：Qwen3-VL-2B-Instruct 的 `tie_word_embeddings=True`，无独立 lm_head

### 第二轮：bfloat16（8/8 通过，2026-02-07）

| 项目 | float32 | bfloat16 | 节省 |
|------|---------|----------|------|
| 参数 | 8.4GB | 4.2GB | 50% |
| Adam 优化器状态 | 16.8GB | 6.88GB | 59% |
| 总计 | ~25GB | ~11GB | 56% |

混合精度策略：RMSNorm variance、Softmax、RoPE、cross-entropy 保持 float32。

### 第三轮：端到端真实图片训练（2026-02-07）

**配置**：v6e-4, LLaVA-Instruct-150K, per_device_batch=4, model_max_length=1024

修复前（XLA 每步重编译）：step time 60-106s，~50 tokens/s
修复后（固定形状填充）：step 3+ 为 0.30s，~14,000 tokens/s（**~250x 加速**）

---

## XLA 重编译修复详情

### 问题根因

JAX/XLA 为每个唯一张量形状编译一次。以下张量形状在 batch 间变化：

**视觉张量**：`pixel_values (N, 6, 16, 16)`、`image_grid_thw (num_images, 3)`、`image_pos_ids_*`、`image_cu_seqlens`
**文本张量**：`input_ids/labels/position_ids/attention_mask` 填充到 batch 内最大长度

### 解决方案

在 DataCollator 中填充到固定形状：
- 视觉：`max_total_patches = batch_size × (max_pixels // patch_size²)`
- 文本：填充到 `training_args.model_max_length`（设为 1024，非 tokenizer 默认 262K）

填充安全性：
- `PatchEmbed3D`：逐 patch 独立处理，padding 0 → ~0 embedding
- `_build_block_diagonal_mask`：padding tokens 形成独立 segment
- `_scatter_embeddings`：cumsum+clip 只选前 `count(mask)` 个 embedding

---

## 多机训练（v6e-16, 4 hosts，2026-02-09）

完整 1 epoch（2464 步，~61 分钟），2B DP 模式，global_batch=64：

| 指标 | 值 |
|------|-----|
| XLA 编译（step 1-2） | ~100s |
| 稳态 step time | 0.68s |
| Throughput | ~25,000 tokens/s |
| 最终 avg_loss | **1.2727** |

多机注意事项：
- `jax.distributed.initialize()` 必须调用（Orbax 多机 checkpoint 需要）
- `max_pixels=50176`（默认 451584 + batch=64 → OOM）
- Orbax checkpoint 需要绝对路径
- GCS tensorboard 不支持追加写入，写本地后 SDK 同步

---

## Checkpoint 断点续训实现（2026-02-10）

### Opt_state sharding 问题（多机模式）

Orbax `StandardRestore` 正确恢复大数组（params、Adam mu/nu），但 opt_state 标量（如 Adam `count`）可能只在单个 host 的设备上，导致 `ValueError: Received incompatible devices`。

**修复尝试**：
1. ~~`jax.device_put(x, replicate)` 直接 re-shard~~ — `CopyArrays only supports destination device list of the same size`
2. ~~`jax.device_put(np.asarray(x), replicate)` 全部转 numpy~~ — OOM（mu/nu ~2.7GB each）
3. **`_ensure_global()` 选择性 re-shard** — 只对 `len(x.devices()) < global_device_count` 的叶节点处理（`8d4a002`）

### 验证结果（v6e-16）

Test 1（20 步训练）：checkpoint step 10/20 PASS，avg_loss=1.6697，0.67s/step
Test 2（从 step 20 恢复到 step 30）：loss 连续（未跳回初始值），checkpoint step 30 PASS

---

## JAX 0.6.2 → 0.9.0 升级（2026-02-10）

| 指标 | JAX 0.6.2 | JAX 0.9.0 |
|------|-----------|-----------|
| XLA 编译 (step 1-2) | ~100s | ~65s（**35% 加速**） |
| 稳态 step time | 0.68s | 0.67s |

升级要点：Python 3.10（runtime 默认）不支持 JAX 0.7.0+，需安装 Python 3.11 venv。
所有训练代码零修改，API 完全向后兼容。

---

## Orbax 原生 GCS Checkpoint 迁移（2026-02-10）

从 process-0-only `flax.serialization` bypass 迁移到 Orbax 原生 `CheckpointManager` 直接指向 `gs://`。

**删除的代码**（~185 行）：`_save_multihost`、`_restore_multihost`、`download_from_gcs`、`_sync_to_gcs`、`_cleanup_old`、`_latest_step_multihost` 及单机/多机分支逻辑。

存储格式从 msgpack 变为 tensorstore/zarr（Orbax 原生）。

---

## GCS 上传改用 google-cloud-storage SDK（2026-02-11）

替代原先的 `subprocess.run(["gcloud", "storage", "cp", ...])`，消除对外部 CLI 依赖。

新增 `jax_qwenvl/utils/gcs.py`：
- `upload_files_to_gcs(local_dir, gcs_uri, extensions)` — 按扩展名过滤上传
- `sync_dir_to_gcs(local_dir, gcs_uri)` — 全目录同步
- `upload_file_to_gcs(local_path, gcs_uri, filename)` — 单文件上传

验证（v6e-16，20+20 步训练+续训）：7 files via SDK ✓，1 TB file via SDK ✓，exit code 0 ✓

---

## FSDP 模式修复与验证（2026-02-11）

### 修复的 Bug

1. `shard_batch()` 硬编码 `'dp'` 轴 → FSDP 下 batch 从未被分片
2. `batch_size` 使用 `mesh.shape['dp']`（FSDP 下为 1）→ global_batch 错误
3. `export_hf_weights` 仅 process 0 调用 → `process_allgather` 集合操作死锁

### FSDP vs DP 对比（v6e-16, 16 chips）

| 模式 | 稳态 step time | Throughput |
|------|---------------|------------|
| DP | 0.67s | ~25,000 tok/s |
| FSDP | **0.35s** | **~49,000 tok/s** |

---

## 混合 DP+FSDP 模式（2026-02-12）

### 三种并行模式性能（v6e-16, 16 chips, Qwen3-VL-2B）

| 模式 | Mesh | 稳态 Step Time | Throughput | XLA 编译 |
|------|------|---------------|------------|---------|
| DP | dp=16, fsdp=1 | 0.67s | ~26,000 tok/s | ~65s |
| FSDP | dp=1, fsdp=16 | 0.35s | ~49,000 tok/s | ~133s |
| **Hybrid** | dp=4, fsdp=4 | **0.38s** | **~45,000 tok/s** | ~65s |

全部 7/7 测试通过（基本训练、断点续训、FSDP 回归、DP 回归、边界退化、优先级测试）。

Hybrid 关键设计：`P(('dp', 'fsdp'))` batch 分片；参数分片与纯 FSDP 相同；`num_data_devices = dp * fsdp` 统一公式。

---

## 8B 模型 Hybrid 验证（2026-02-13）

**环境**：v6e-16, FSDP_DEVICES=4 (dp=4, fsdp=4), per_device_batch=2, global_batch=32

修复的问题：
1. **HF Hub 模型 ID**：`snapshot_download` 自动下载到 `~/.cache/huggingface/`
2. **TPU watchdog 超时**：8B 分片需 ~325s，改用 batched `jax.device_put(params, sharding_tree)` 修复
3. **磁盘空间不足**：8B safetensors ~32GB，设 `GCS_OUTPUT_DIR` 后逐 shard 流式上传修复

内存分析（hybrid dp=4, fsdp=4）：参数 4GB + Adam 8GB + 梯度 4GB + 激活 1GB ≈ 17GB / 31.25GB HBM

20 步训练：avg_loss=1.8565，0.69s/step，~13k tok/s；Checkpoint 恢复：27.7s（1.76 GiB/s）

---

## 8B Weight Export 修复：SDK 流式上传（2026-02-13）

**问题**：TPU VM 100GB boot disk，8B float32 safetensors ~32GB + HF cache + 数据集 + venv > 100GB

**方案演进**：
1. ~~gcsfuse 挂载~~ — ~16 MB/s，32GB 需 33min，超过 JAX shutdown barrier 5min 超时
2. ~~SDK 流式上传（无 barrier）~~ — 非主进程提前退出触发 shutdown barrier
3. **SDK 流式上传 + `sync_global_devices`** — 每 shard 写临时文件 → 上传 → 删除，磁盘需求降到 ~5GB

验证（v6e-16，europe-west4-a）：7/7 shards (32.67 GiB) 上传 GCS，~27 MB/s，~20min，exit code 0 ✓

---

## 8B 模型完整 1 Epoch 训练（2026-02-14）

**配置**：v6e-16, Hybrid dp=4 fsdp=4, per_device_batch=2, global_batch=32, LLaVA-Instruct-150K

| 指标 | 值 |
|------|-----|
| 总步数 | 4,928（1 epoch） |
| 初始 loss | 1.9276 |
| 最终 avg_loss | **1.7937** |
| 稳态 step time | 0.69s |
| 平均 throughput | ~12,500 tokens/s |
| XLA 编译（step 1-2） | ~248s |
| 参数分片时间 | ~313s |
| Checkpoint 保存 | ~35-41s/次（每 100 步） |
| Weight export | ~8.5 min（7 shards, 32.67 GiB → GCS） |
| 总 wall time | ~2h 19min |

关键节点 loss 曲线：step 3 → 1.8781（0.69s），step 1000 → avg 1.8117，step 4928 → avg **1.7937**

2B vs 8B 对比：8B 初始 loss 更高（1.93 vs 1.66），收敛更慢（降幅 7% vs 23.5%）；建议多 epoch 训练或更高学习率。

---

## GKE v6e-16 验证（2026-04-05）

首次在 GKE（非 GCE TPU VM）上验证训练，结果与 GCE 基准完全一致。

### 集群配置

| 项目 | 值 |
|------|-----|
| 集群 | tpu-v6e-cluster，asia-northeast1-b，GKE 1.32 |
| 节点池 | ct6e-standard-4t × 4，4x4 拓扑，Spot |
| 辅助节点 | n2-standard-4 × 1（default-pool） |

### 验证结果（8B Hybrid, 30步基准）

| 指标 | 值 |
|------|-----|
| step_time（稳态） | **0.78s** |
| tokens/s | **~12.4k** |
| avg_loss（30步） | **1.8243** |
| XLA 编译（step 1-2） | ~160s |
| 参数分片时间 | ~488s |

与 GCE TPU VM 基准完全一致。

### 踩坑记录

**1. GCS 写权限 403**
- 现象：Orbax CheckpointManager 初始化时写 GCS 报 `Provided scope(s) are not authorized`
- 根因：GKE 节点 SA 默认 scope 只有 `deepstorage.read_only`
- 解决：Workload Identity —— 集群启用 `--workload-pool`，节点池加 `--workload-metadata=GKE_METADATA`，创建 KSA 绑定到具有 objectAdmin 权限的 GSA

**2. XLA 编译静默崩溃**
- 现象：Sharding 完成后 5-26 分钟，所有 pod 静默崩溃（无 Python 异常，只有 JAX Shutdown barrier 错误）
- 根因：容器默认 Python 3.12，与 `torch`（CPU-only）+ `libtpu` 共存时 XLA 编译层崩溃
- 解决：在容器内创建 Python 3.11 venv，用 `pip install -r requirements.txt` 完整安装依赖

**3. node selector 值**
- 正确：`cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice`
- 错误：`tpu-v6e-podslice`（GKE 文档示例，实际节点 label 不同）

### Kubernetes 资源文件

见 `jax_qwenvl/gke/`：
- `verify-tpu-v6e-16.yaml`：TPU 设备验证 Job
- `qwen3vl-8b-train-job.yaml`：8B 训练 Job（含 Workload Identity、Python 3.11、完整注释）

GCS 输出（`gs://grhuang-02-vertex-ai/qwen3vl-8b-full-epoch/`）：7 safetensors + index.json + processor files + checkpoints/4600~4928
