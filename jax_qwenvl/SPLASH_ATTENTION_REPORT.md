# Splash Attention 实现报告

> 日期：2026-04-17  
> 分支：`feat/splash-attention`  
> 模型：Qwen3-VL-8B-Instruct，llava_instruct_150k，30步基准  

---

## 1. 背景

之前 L=8192 使用 `jax.nn.dot_product_attention`（XLA，O(L²) 内存）：
- step_time：5.22s
- HBM：**95.8%**（接近上限）
- 原因：每层计算完整 `(B, H, L, L)` 注意力矩阵

目标：改用 Pallas Splash Attention（O(L) 内存）降低 HBM，提升速度。

---

## 2. 实现方案

### 2.1 正确模式：MaxText 的 `shard_map(vmap(splash_kernel))`

之前尝试的错误方式（6× 慢）：
```python
# ❌ 错误：直接传全 batch 给 flash_attention
shard_map(lambda q, k, v, ab: flash_attention(q, k, v, ab=ab, ...))(q, k, v, ab)
```

MaxText 的正确方式：
```python
# ✅ 正确：vmap 在 shard_map 内部对 batch 维度遍历
shard_map(
    lambda q, k, v: jax.vmap(splash_kernel)(q, k, v),
    mesh=mesh, in_specs=(...), out_specs=..., check_vma=False,
)(q, k, v)
```

### 2.2 CausalMask 内嵌到 kernel

不传 `ab`（additive float mask），而是将 CausalMask 构建到 kernel 内部：

```python
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as sak,
    splash_attention_mask as sam,
)

causal_mask = sam.CausalMask(shape=(L, L))
multi_head_mask = sam.MultiHeadMask(masks=(causal_mask,) * H_q)

splash_kernel = sak.make_splash_mha(
    mask=multi_head_mask,
    block_sizes=sak.BlockSizes(
        block_q=512, block_kv=512, block_kv_compute=512,  # 前向
        block_q_dkv=512, block_kv_dkv=512,                # 反向（必须指定）
        block_kv_dkv_compute=512,
        block_q_dq=512, block_kv_dq=512,
    ),
    head_shards=1, q_seq_shards=1,
)
```

**注意**：不指定反向 block sizes 会报错 `"Need to specify backward blocks."`

### 2.3 GQA 处理

Splash Attention 不原生支持 GQA（Qwen3-VL-8B：32 Q heads，8 KV heads），在 `_splash_fn` 内部展开：

```python
def _splash_fn(q_l, k_l, v_l):
    # Splash 不支持 GQA，在 shard_map 内展开 K/V
    k_l = jnp.repeat(k_l, num_kv_groups, axis=1)  # 8 → 32 heads
    v_l = jnp.repeat(v_l, num_kv_groups, axis=1)
    return jax.vmap(splash_kernel)(q_l, k_l, v_l)
```

### 2.4 全局 mesh 传入模型

在 `sharding.py` 添加全局 mesh 注册，供模型层调用：

```python
def register_global_mesh(mesh, sharding_mode='dp'): ...
def get_global_mesh(): ...  # 返回 (mesh, sharding_mode)
```

在 `train.py` 创建 mesh 后立即注册：
```python
register_global_mesh(mesh, sharding_mode)
```

---

## 3. 测试流程中遇到的问题

### 3.1 Kueue 队列挡住

集群使用 Kueue 管理资源，`cluster-queue` 配额 256 TPU slots 被占满（其他用户），即使我们有专属预留节点也无法运行。

**解决方案**：使用 `poc-dev` namespace，其 `localqueue: lq` → `poc-dev` clusterqueue，有独立配额（16 slots），pending=0。

```bash
# 在 poc-dev namespace 提交，指定 poc-dev localqueue
kubectl apply -n poc-dev -f job.yaml  # job 需要 label: kueue.x-k8s.io/queue-name: lq
```

需要同时在 poc-dev namespace 创建 KSA：
```bash
kubectl create serviceaccount grhuang-trainer-ksa -n poc-dev
kubectl annotate serviceaccount grhuang-trainer-ksa -n poc-dev \
  iam.gke.io/gcp-service-account=YOUR_GSA@YOUR_GCP_PROJECT.iam.gserviceaccount.com
```

### 3.2 Kueue 配额说明

`cluster-queue` 配额 256 是**管理员设置的策略配额**，不是物理上限。集群各 queue 分配：

| Cohort | Queue | TPU 配额 |
|--------|-------|---------|
| `tpu-pool` | `cluster-queue` | 256（当前全满）|
| `poc-cohort` | poc-dev | 16 |
| `poc-cohort` | poc-gsc/ml-perf/… | 16-64 |

---

## 4. 测试结果

### 4.1 性能指标（L=8192, batch=2, FSDP_DEVICES=4）

| 配置 | step_time | tokens/s（实际）| HBM 利用率 |
|------|-----------|----------------|-----------|
| XLA `jax.nn.dot_product_attention` | 5.22s | ~1,900 | 95.8% |
| **Splash Attention shard_map(vmap)** | **3.31s** | **~2,900** | **93.0%** |
| 提升 | **37% 更快** | **53% 更多** | 降低 2.8pp |

### 4.2 Smoke Test 通过

```
Splash Attention shard_map(vmap) forward+backward: PASSED ✓
VisionAttention dot_product_attention fallback: PASSED ✓
```

### 4.3 与 H200 横向对比

| | H200 ×8 | v7x-16 XLA | v7x-16 Splash |
|--|---------|-----------|--------------|
| 物理加速器 | 8 块 GPU | **8 块物理芯片** | **8 块物理芯片** |
| 逻辑设备 | 8 | 16（每芯片 2 TensorCore）| 16 |
| step_time | 3.3s | 5.22s | **3.31s** |
| 有效 batch | 64 × 8192 = 524k tok | 32 × 8192 = 262k tok | 32 × 8192 = 262k tok |
| padded tok/s | **~159k** | ~50k | **~79k** |
| vs H200 | 1× | 0.31× | **0.50×** |

> 注：H200 使用 grad_accum=4，每步有效 batch=64；v7x 为 batch=2，有效 batch=32。物理芯片数相同（8块），但 H200 每步处理 2× 的 token。

---

## 5. HBM 分析

HBM 从 95.8% 降至 93%，改善有限的原因：

**GQA 展开抵消了 Flash Attention 的内存节省：**

```
Splash Attention 节省：注意力矩阵 O(L²) → O(L)，约减少 7.5 GiB/层

GQA 展开增加：K/V 从 8 heads 展开到 32 heads（4× 数据量）
  展开后 K/V 内存：32 × L × D × 2（bf16）× 2（K+V）= 每设备额外几 GB
```

净效果：仅降低 2.8pp（95.8% → 93%）。

---

## 6. GQA 支持现状

### 6.1 Splash Attention API 现状

| API | 支持场景 |
|-----|---------|
| `make_splash_mha()` | MHA：Q/K/V heads 相同 |
| `make_splash_mqa()` | MQA：1 个 KV head（极端情形）|
| **GQA（8 KV heads）** | **❌ 无原生支持** |

### 6.2 社区现状（2026-04 调研）

- 无任何公开 Issue 或 PR 专门针对 GQA 支持
- MaxText 通过 `jnp.repeat` 展开解决（与我们相同）
- 这是**社区尚未填补的空缺**，不是已规划的 roadmap 功能

### 6.3 可能的解法

要真正支持 GQA 需要自行在 Pallas kernel 层实现 KV head broadcasting，工作量较大（需要修改 Mosaic IR 层的 kernel 代码）。

---

## 7. 待完成工作

| 优先级 | 方向 | 说明 |
|--------|------|------|
| 高 | 原生 GQA Splash Attention | 向 JAX 提 PR 或等待社区实现 |
| 中 | 增大 batch size | HBM 93%，需先解决 GQA 才有空间 |
| 低 | `feat/splash-attention` 合并到主分支 | 需充分测试收敛性 |

---

## 8. 代码变更清单

| 文件 | 变更内容 |
|------|---------|
| `jax_qwenvl/train/sharding.py` | 新增 `register_global_mesh` / `get_global_mesh` |
| `jax_qwenvl/train/train.py` | 调用 `register_global_mesh(mesh, sharding_mode)` |
| `jax_qwenvl/model/llm.py` | TextAttention 替换为 Splash Attention + shard_map(vmap) |
| `jax_qwenvl/tests/smoke_test_tpu.py` | 使用真实 mesh + shard_map(vmap(splash_kernel)) |
| `jax_qwenvl/tests/test_flash_attention.py` | 8 个 CPU 单元测试（CPU 走 dot_product_attention 回退）|

---

*参考：MaxText attentions.py，JAX Pallas splash_attention_kernel.py*
