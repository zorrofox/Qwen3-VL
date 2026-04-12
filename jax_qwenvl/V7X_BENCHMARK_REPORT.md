# TPU v7x 训练基准测试报告

> 日期：2026-04-12  
> 模型：Qwen3-VL-8B-Instruct  
> 数据集：llava_instruct_150k（157,712 样本）  
> 集群：GKE `bodaborg-tpu7x-auto-nap2`，us-central1-c  

---

## 1. TPU v7x 硬件规格

| 参数 | 值 |
|------|-----|
| 机器类型 | `tpu7x-standard-4t` |
| 物理芯片数/节点 | 4 |
| TensorCore 数/芯片 | 2（独立 chiplet） |
| JAX devices/节点 | 4 × 2 = **8** |
| HBM/芯片 | 192 GiB |
| HBM/JAX device | 192 / 2 = **96 GiB** |
| HBM 带宽/芯片 | 7,380 GiBps |
| Peak BF16 算力/芯片 | 2,307 TFLOPs |
| VMEM/TensorCore | 64 MB |

**关键发现**：`tpu7x-standard-4t` 每节点有 8 个 JAX 逻辑设备（非 4 个），因为每个物理芯片包含 2 个独立 TensorCore，各自拥有独立的 96 GiB HBM。

### 测试配置（tpu7x-16）

```
2 nodes × 4 chips × 2 TensorCore = 16 JAX devices
topology: 2x2x2
xpk 命名: tpu7x-16
GKE nodeSelector: gke-tpu-accelerator=tpu7x, gke-tpu-topology=2x2x2
```

FSDP 配置：

```
FSDP_DEVICES=4 → dp=4, fsdp=4, hybrid 模式
每 JAX device 本地 batch = 2
全局 batch = 16 devices × 2 = 32
```

---

## 2. 训练性能测试结果

### 2.1 稳态 step time 对比

| 配置 | step_time | tokens/s（实际） | HBM 占用 | 状态 |
|------|-----------|----------------|---------|------|
| L=1024, batch=2, XLA | **0.49s** | **~20k** | 低 | ✅ 最优 |
| L=8192, batch=2, XLA | 5.22s | ~1,900 | **95.8%** | ⚠️ HBM 紧张 |
| L=8192, batch=2, Pallas shard_map | 29.6s | ~300 | 未测 | ❌ 已回退 |

> 注：tokens/s 为实际 token 数（非 padding），由训练日志 metrics_logger 计算。

### 2.2 HBM 监控数据

通过 Cloud Monitoring API（`kubernetes.io/node/accelerator/memory_used`）实测：

```
L=8192, batch=2, XLA：
  每芯片 HBM_used  = 181.58 GiB
  每芯片 HBM_total = 189.49 GiB（≈ 192 GiB，余量为 runtime 保留）
  利用率           = 95.8%
```

L=8192 时 HBM 接近上限的原因：`jax.nn.dot_product_attention`（XLA 实现）在每层计算完整的 `(B, H, L, L)` 注意力矩阵，复杂度 O(L²)。

### 2.3 XLA 编译时间

| 配置 | 编译步数 | 每步编译时间 | 稳态 step_time |
|------|----------|-------------|---------------|
| L=1024 (tpu7x-16) | 2步 | ~96s/步 | 0.49s |
| L=8192 (tpu7x-16) | 2步 | ~148s/步 | 5.22s |

编译步数为 2（而非 1）是因为 FSDP 的前向和反向分别编译。

---

## 3. 与 H200 ×8 的横向对比

### 3.1 测试配置

| 参数 | H200 ×8 | v7x-16 (L=1024) | v7x-16 (L=8192) |
|------|---------|----------------|----------------|
| 框架 | PyTorch + DeepSpeed ZeRO-3 | JAX + Flax Hybrid FSDP | JAX + Flax Hybrid FSDP |
| 序列长度 | 8,192 | 1,024 | 8,192 |
| per-device batch | 2 | 2 | 2 |
| 梯度累积 | 4 | 1 | 1 |
| 有效 batch（全局 token 数/步） | 64 × 8192 = **524k tokens** | 32 × 1024 = **33k tokens** | 32 × 8192 = **262k tokens** |

### 3.2 性能指标

| 指标 | H200 ×8 | v7x (L=1024) | v7x (L=8192) |
|------|---------|-------------|-------------|
| step_time | 3.3s | 0.49s | 5.22s |
| 有效 tokens/s（padded） | **~159k** | ~67k | ~50k |
| samples/s | 19.4 | 65.3 | 6.1 |
| vs H200（tokens/s） | 1× | 0.42× ❌ | 0.31× ❌ |
| 最终 loss（步后） | **0.875**（1233步） | —（仅30步基准）| —（仅30步基准）|

### 3.3 对比结论

**以有效 token 吞吐量（真实训练效率指标）衡量**：

- v7x 在 L=1024 时样本吞吐最高（65 samples/s vs H200 的 19），但每步只处理 33k tokens，而 H200 每步处理 524k tokens。
- **实际训练 token 效率**：H200 是 v7x L=1024 的 2.4×，是 v7x L=8192 的 3.2×。
- 根本差距：H200 使用 cuDNN Flash Attention（O(L) 内存），可以高效跑 L=8192 大 batch；v7x 当前 XLA attention 是 O(L²)，无法在 L=8192 上扩大 batch。

---

## 4. Flash Attention 研究历程

### 4.1 尝试一：`jax.nn.dot_product_attention`（无效）

**做法**：将原始手动 matmul+softmax 替换为 `jax.nn.dot_product_attention`。

**结果**：
- HBM：无变化（仍 95.8%）
- step_time：5.22s → 4.99s（仅快 4%）

**原因**：`jax.nn.dot_product_attention` 在 TPU 上默认走 XLA 实现（`implementation=None`），与手动计算等价，仍是 O(L²) 内存。JAX 0.9.0 中没有 TPU 的 `implementation='flash'` 或 `implementation='pallas'` 选项。

---

### 4.2 尝试二：Pallas `flash_attention` + `shard_map`（更慢）

**做法**：
```python
from jax.experimental.pallas.ops.tpu import flash_attention as tpu_fa
from jax import shard_map

def _pallas_attn(q, k, v, ab):
    k = jnp.repeat(k, num_kv_groups, axis=1)  # GQA 展开
    ab_full = jnp.broadcast_to(ab, (B_l, H_q, L, L))
    return tpu_fa.flash_attention(q, k, v, ab=ab_full, sm_scale=scaling)

attn_output = shard_map(
    _pallas_attn, mesh=global_mesh,
    in_specs=(...), out_specs=..., check_vma=False
)(q, k, v, ab)
```

**结果（L=8192）**：
- step_time：**29.6s**（XLA 5.22s 的 **6× 慢**）
- tokens/s：~300（降至 XLA 的 1/6）

**已经修复的 API 问题**（在调试过程中发现）：

| 问题 | 错误现象 | 修复 |
|------|---------|------|
| CPU 单元测试未覆盖 dtype 差异 | `value dtype should be float32, but got bfloat16` | RoPE 后 q/k 为 float32，v 为 bfloat16，统一 cast |
| Pallas bias 不接受广播形状 | `Attention bias shape mismatch` | `jnp.broadcast_to(ab, (B, H, L, L))` |
| `shard_map` 不接受 `check_rep` | `got unexpected keyword argument 'check_rep'` | 改为 `check_vma=False` |
| `jax.experimental.shard_map` 已弃用 | DeprecationWarning | 改为 `jax.shard_map` |
| Pallas 不支持 GQA | `Head count mismatch: 32, 8, 8` | shard_map 内部做 `jnp.repeat` |

**6× 慢的根本原因**：
1. shard_map 内部 `jnp.repeat` 每步展开 K/V（8→32 heads），4× 数据量
2. Pallas `flash_attention` kernel 非最优（应用 `splash_attention`）
3. shard_map dispatch 开销在 JAX 0.9.0 较高
4. block_size 默认 128 对 Ironwood 64MB VMEM 来说过小

---

### 4.3 研究结论：正确的方向（未实现）

#### 应使用 `splash_attention`，不是 `flash_attention`

| | `flash_attention` | `splash_attention`（推荐） |
|--|-----------------|-------------------------|
| 优化程度 | 早期版本 | 专为 Ironwood 优化 |
| DMA pipelining | 否 | 是 |
| VMEM 利用 | 低 | 高（64MB 充分利用）|
| L<4K | 无优势 | 无优势（XLA 已够）|
| L>4K | 有优势 | **更大优势** |

#### MaxText 的正确实现模式

```python
# MaxText 在 Ironwood 上的正式做法：vmap 必须在 shard_map 内部
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel, make_splash_mha, SegmentIds, BlockSizes
)

# 构造 splash kernel（只做一次）
kernel = make_splash_mha(
    block_sizes=BlockSizes(
        block_q=512,         # 推荐：Ironwood 大 VMEM 适合大 block
        block_kv_compute=512,
        block_kv=512,
    )
)

@functools.partial(
    shard_map, mesh=mesh,
    in_specs=(batch_spec, batch_spec, batch_spec, seg_spec),
    out_specs=batch_spec,
    check_rep=False,
)
def wrapped(q, k, v, seg_ids):
    # vmap 在 shard_map 内部处理 batch 维度
    return jax.vmap(kernel)(q, k, v, segment_ids=seg_ids)
```

**注意**：`splash_attention` 用 `segment_ids` 处理 packed sequences（替代我们的 block-diagonal additive mask），需要修改数据管道。

#### 现有库调研（2026 年 4 月）

| 库 | TPU 支持 | SPMD 自动 | 结论 |
|----|---------|-----------|----|
| `flash-attn-jax 0.6.2` | ❌ CUDA 专用 | — | 无用 |
| `kvax`（Nebius） | ❌ GPU/Triton | — | 无用 |
| `jax-flash-attn2` | ✅ Pallas 后端 | ❌ 需 shard_map | 可参考 |
| JAX 内置 `splash_attention` | ✅ Ironwood 优化 | ❌ 需 shard_map | **推荐路径** |

JAX 0.9.2（2026-03-18 发布）未新增 TPU Flash Attention 自动支持，shard_map 仍是必须的。

---

## 5. 测试基础设施

### 5.1 GKE 相关配置（v7x 专项）

详见 `jax_qwenvl/gke/` 目录：

| 文件 | 用途 |
|------|------|
| `smoke-test-v7x.yaml` | 提交训练前必须通过的 smoke test |
| `qwen3vl-8b-v7x-train-job.yaml` | 正式训练 Job（L=8192，batch=2）|
| `verify-tpu-v7x-8chips.yaml` | 单 host 设备验证（8 JAX devices）|

**提交规范**：
```bash
# 1. CPU 单元测试
JAX_PLATFORM_NAME=cpu python3 -m pytest jax_qwenvl/tests/test_flash_attention.py -v

# 2. TPU Smoke Test（必须 PASSED 才能提交训练）
kubectl apply -f jax_qwenvl/gke/smoke-test-v7x.yaml
kubectl wait --for=condition=complete job/v7x-smoke-test --timeout=600s

# 3. 训练 Job
kubectl apply -f jax_qwenvl/gke/qwen3vl-8b-v7x-train-job.yaml
```

### 5.2 GKE 踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 调度失败（nodeSelector 注入错误） | `optimize-utilization-scheduler` 把 cpu/memory 请求映射到 `cpu-np` 节点池 | pod spec 中**不要**请求 cpu/memory，只请求 `google.com/tpu: 4` |
| multi-host 需要 workload policy | v7x Ironwood 要求 `HIGH_THROUGHPUT` workload policy | `gcloud beta compute resource-policies create workload-policy NAME --type=HIGH_THROUGHPUT --accelerator-topology=2x2x2` |
| GCS 403 | 节点 SA 项目号混用（`706422770546` vs `735972712744`） | 授权正确的 SA：`735972712744-compute@developer.gserviceaccount.com` |
| HuggingFace 下载挂起 | Xet CDN 在 v7x pod 网络环境不可访问 | 预先下载到 GCS：`gs://grhuang-02-vertex-ai/models/Qwen3-VL-8B-Instruct/qwen3vl-8b/` |
| Checkpoint 保存崩溃 | `FSDP_DEVICES=8` 下 Orbax 2-process allgather 有 bug | 使用 `FSDP_DEVICES=4`（dp=4, fsdp=4），与 v6e 相同 mesh 结构 |

---

## 6. 当前状态与后续建议

### 6.1 当前代码状态

```
jax_qwenvl/model/llm.py TextAttention:
  使用 jax.nn.dot_product_attention（XLA，SPMD 兼容，O(L²)）
  + GQA 展开（jnp.repeat）
  + dtype 统一 cast（处理 RoPE upcast）
```

已通过的测试（`jax_qwenvl/tests/`）：
- 8 个 CPU 单元测试（test_flash_attention.py）
- TPU Smoke Test（使用真实 mesh + jit + grad）

### 6.2 实现 Splash Attention 的前提条件

1. **数据管道改造**：将 block-diagonal additive mask 改为 `segment_ids`（每 token 所属序列编号），适配 Splash Attention API
2. **Block size 调参**：从默认 128 改为 512，充分利用 Ironwood 64MB VMEM
3. **Smoke test 更新**：测试 `splash_attention + vmap inside shard_map` 路径
4. **JAX 版本考虑**：升级到 JAX 0.9.2（最新，2026-03-18）

### 6.3 是否继续追 Flash Attention

| 场景 | 建议 |
|------|------|
| 训练长上下文（L>4K） | 必须实现 Splash Attention，否则无法与 H200 竞争 |
| 训练短上下文（L≤1024） | 现状 XLA 已足够，v7x 在 samples/s 上有优势 |
| 与 H200 有效 token 吞吐对比 | 需要 L=8192 + Flash Attention，否则差距 2.4-3× |

---

## 7. 已验证性能基准汇总

| 配置 | 硬件 | step_time | tokens/s（实际） | HBM 利用率 | loss |
|------|------|-----------|----------------|-----------|------|
| 8B Hybrid dp=4,fsdp=4 | GKE v7x-16 | **0.49s** | ~20k | 低 | 1.83（30步）|
| 8B Hybrid dp=4,fsdp=4 | GKE v6e-16 | 0.78s | ~12.4k | — | 1.82（30步）|
| 8B ZeRO-3, L=8192 | GKE H200×8 | 3.3s | —（padded ~159k）| — | **0.875**（1233步）|
| 8B XLA, L=8192 | GKE v7x-16 | 5.22s | ~1.9k | **95.8%** | 1.82（30步）|
| 8B Pallas shard_map, L=8192 | GKE v7x-16 | 29.6s | ~300 | — | 1.82（30步）|

---

*参考资料：*
- *MaxText Flash Attention 实现：https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/layers/attentions.py*
- *JAX Splash Attention：https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/*
- *Google Ironwood 性能指南：https://docs.cloud.google.com/tpu/docs/ironwood-performance*
- *JAX 0.9.2 Changelog：https://docs.jax.dev/en/latest/changelog.html*
