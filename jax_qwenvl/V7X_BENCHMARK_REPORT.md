# TPU v7x — Qwen3-VL 8B 训练基准报告

> **模型**：Qwen3-VL-8B-Instruct（自定义 JAX/Flax 多模态训练栈）
> **数据**：LLaVA-Instruct-150K (157,712 样本)
> **首版**：2026-04-12 ｜ **最近更新**：2026-04-21

---

## 1. 硬件与集群

### 1.1 v7x (Ironwood) 规格

| 参数 | 值 |
|------|-----|
| 机器类型 | `tpu7x-standard-4t` |
| 物理芯片数 / 节点 | 4 |
| TensorCore / 芯片 | 2（独立 chiplet） |
| **JAX devices / 节点** | **8**（4 chips × 2 cores） |
| HBM / 物理芯片 | 192 GiB（2 个 JAX dev 共享） |
| HBM 带宽 / 芯片 | 7,380 GiB/s |
| BF16 峰值 / 芯片 | 2,307 TFLOP/s |
| BF16 峰值 / JAX dev | **1,153 TFLOP/s** |
| VMEM / TensorCore | 64 MB |

### 1.2 测试集群

| 集群 | 区域 | 拓扑 | 总 dev | 用途 |
|------|------|------|--------|------|
| `YOUR_GKE_CLUSTER` | asia-northeast1 | 2x2x2 (2 host) | 16 | 历史 multi-host 基准 |
| `bodaborg-tpu7x-auto-nap2` | us-central1-c | 2x2x1 (1 host) | 8 | 当前活跃基准（lustre PVC） |

---

## 2. 性能基准

### 2.1 完整对照表

> tokens/s 为 metrics_logger 实算（基于 attention_mask sum，非 padding）；MFU = actual_FLOPs / peak。

| # | 平台 | quant | per_dev_bs | global_bs | L | step_time | tokens/s | HBM peak | MFU | loss(30步) | 备注 |
|---|------|-------|-----------|-----------|---|-----------|----------|----------|-----|-----------|------|
| 1 | **H200×8** PyTorch ZeRO-3 | bf16 | 2 (grad×4) | 64 | 8192 | 3.3s | ~159k pad | — | — | **0.875** (1233步) | cuDNN Flash Attn |
| 2 | v7x-16 hybrid dp=4/fsdp=4 | bf16 | 2 | 32 | 1024 | 0.49s | ~20k | 低 | — | 1.83 | XLA attention |
| 3 | v7x-16 hybrid dp=4/fsdp=4 | bf16 | 2 | 32 | 8192 | 5.22s | ~1.9k | 95.8% | — | 1.82 | XLA O(L²) |
| 4 | v7x-16 hybrid dp=4/fsdp=4 | bf16 | 2 | 32 | 8192 | **3.31s** | ~2.9k | 93% | — | — | **Splash Attention** ✅ |
| 5 | v6e-16 hybrid dp=4/fsdp=4 | bf16 | 2 | 32 | 1024 | 0.78s | ~12.4k | — | — | 1.82 | 参照 |
| 6 | v7x-8 hybrid dp=2/fsdp=4 | bf16 | 2 | 16 | 1024 | 0.56s | ~8.5k | — | — | 2.537 | single-host A |
| 7 | v7x-8 hybrid dp=2/fsdp=4 | bf16 | 4 | 32 | 1024 | 0.75s | ~12.5k | 67% | 25.6% | 2.525 | **BF16 sweet spot** |
| 8 | v7x-8 hybrid dp=2/fsdp=4 | bf16 | 8 | 64 | 1024 | 1.45s | ~12.5k | 67% | 26.5% | 2.534 | compute bound |
| 9 | v7x-8 hybrid dp=2/fsdp=4 | **fp8 e4m3fn** | 4 | 32 | 1024 | **0.71s** | **~13.3k** | **45.5%** | **27.0%** | 2.549 | **FP8 sweet spot** ✅ |
| 10 | v7x-8 hybrid dp=2/fsdp=4 | fp8 e4m3fn | 8 | 64 | 1024 | 1.37s | ~13.5k | 66% | 28.0% | 2.572 | compute bound |

> H200 用 PyTorch + DeepSpeed ZeRO-3，跑足 1233 步，loss 收敛到 0.875。其他 v7x/v6e 行均为 30 步基准，loss 仅作健康度参考，不可与 H200 直接比较。
> 行 6-10 数据来自 `bodaborg-tpu7x-auto-nap2`，挂载 lustre PVC `/data/qwen3vl/llava_data/`，全部启用 MaxText 推荐 XLA flags（`scoped_vmem_limit_kib=98304` + `sparse_core_collective_offload`）。

### 2.2 关键 deltas

**FP8 vs BF16（同 batch=4，single-host）**：

| 维度 | BF16 | FP8 | Δ |
|------|------|-----|---|
| step_time | 0.75s | 0.71s | -5.3% |
| tokens/s | 12.5k | 13.3k | **+6.4%** |
| HBM peak | 128 GiB (67%) | 87 GiB (45.5%) | **-32%** |
| MFU | 25.6% | 27.0% | +1.4pp |
| loss(30步) | 2.525 | 2.549 | +0.024（收敛健康） |

> **HBM -32% 是 FP8 真生效的硬证据**——这部分压缩不可能由其他因素解释。但 step_time 增益小，因为量化只覆盖 LLM Dense（68% kernel），剩下 32% 的 vision tower 仍是 BF16 + 主要瓶颈。

**Splash Attention vs XLA（同 v7x-16, L=8192, batch=2）**：

| 维度 | XLA `dot_product_attention` | Splash via `shard_map(vmap)` | Δ |
|------|----------------------------|------------------------------|---|
| step_time | 5.22s | 3.31s | -37% |
| tokens/s | ~1,900 | ~2,900 | +53% |
| HBM | 95.8% | 93.0% | -2.8pp |

> Splash 在 L≥4K 才有显著收益；L=1024 时 `block_q=min(512, L)` 退化为 2 个 block，行为接近 XLA（这就是 v7x-8 行 7 仍叫"BF16"而非"Splash"的原因）。

### 2.3 Compute bound 现象

batch=4 → batch=8 (BF16 同样规律 FP8)：
- step_time 1.93×（0.75→1.45 / 0.71→1.37），几乎线性
- tokens/s 几乎持平（~12.5k / ~13.5k）
- HBM 还有余量（FP8 bs4 仅 45.5%）
- 含义：v7x TC 在当前实现下已"无空隙"被打满，加 batch 只是等比例延长时间，throughput 边际为 0

### 2.4 v7x 算力利用率（MFU）天花板

| 实现栈 | TFLOP/s/dev | MFU | 备注 |
|--------|-------------|-----|------|
| H200 PyTorch ZeRO-3 | ~480 (peak 989) | ~48% | cuDNN FA + Apex 融合 |
| MaxText text-only SFT (v7x) | ~232 | ~20% | Google 官方栈 |
| 本仓库 BF16 bs4 (v7x-8) | 295 | **25.6%** | 自定义 + Splash |
| 本仓库 FP8 bs4 (v7x-8) | 312 | **27.0%** | + Qwix QT |

**为何 25-27% 而非 100%**（按影响排序）：
1. **Vision tower 调度开销**：27 层 ViT × hidden=1152，大量小 matmul + LayerNorm，kernel launch 摊不开
2. **LLM Splash 在 L=1024 退化**：block_q=512 时只切 2 个 block，与 XLA 等价
3. **FFN 没接 Pallas 融合 kernel**：SwiGLU = 3 matmul + activation + element-wise，分开走 XLA 多 4 次 HBM 读写
4. **GQA 用 `jnp.repeat`**：K/V 8→32 heads，多 75% HBM 流量
5. **小全局 batch (32-64)**：collective reduction 摊销不充分
6. **FP8 只覆盖 LLM Dense**：vision tower / lm_head 仍 BF16，约束整体上限

### 2.5 HBM 监控方法

```bash
NODE=$(kubectl get pod -n default -l app=qwen3vl-v7x-fast -o jsonpath='{.items[0].spec.nodeName}')
NOW=$(date -u +%s); START=$((NOW - 600))
curl -sS -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://monitoring.googleapis.com/v3/projects/cloud-tpu-multipod-dev/timeSeries?filter=metric.type%3D%22kubernetes.io%2Fnode%2Faccelerator%2Fmemory_used%22%20AND%20resource.labels.node_name%3D%22${NODE}%22&interval.startTime=$(date -u -d @$START +%Y-%m-%dT%H:%M:%SZ)&interval.endTime=$(date -u -d @$NOW +%Y-%m-%dT%H:%M:%SZ)&aggregation.alignmentPeriod=60s&aggregation.perSeriesAligner=ALIGN_MEAN"
```

返回 4 路 series（4 物理芯片，每芯片 192 GiB）。每 JAX dev 看到 96 GiB 上限。

---

## 3. 实现现状

### 3.1 LLM TextAttention — Splash Attention（已上线）

**文件**：`jax_qwenvl/model/llm.py:113-198`

```python
if global_mesh is not None and jax.default_backend() == "tpu":
    # Splash Attention via shard_map(vmap(splash_kernel)) — MaxText 模式
    causal_mask = _sam.CausalMask(shape=(L, L))
    multi_head_mask = _sam.MultiHeadMask(masks=(causal_mask,) * num_heads)
    splash_kernel = _sak.make_splash_mha(
        mask=multi_head_mask,
        block_sizes=_sak.BlockSizes(
            block_q=min(512, L), block_kv=min(512, L), block_kv_compute=min(512, L),
            block_q_dkv=min(512, L), block_kv_dkv=min(512, L), block_kv_dkv_compute=min(512, L),
            block_q_dq=min(512, L), block_kv_dq=min(512, L),
        ),
    )
    def _splash_fn(q_l, k_l, v_l):
        if num_kv_groups > 1:                      # GQA: Splash 不原生支持
            k_l = jnp.repeat(k_l, num_kv_groups, axis=1)
            v_l = jnp.repeat(v_l, num_kv_groups, axis=1)
        return jax.vmap(splash_kernel)(q_l, k_l, v_l)
    attn_output = shard_map(_splash_fn, mesh=global_mesh,
                            in_specs=(batch_spec,)*3, out_specs=batch_spec, check_vma=False)(q, k, v)
else:
    # CPU/GPU 回退：jax.nn.dot_product_attention（XLA O(L²)）
```

| 长度 | 行为 | 收益 |
|------|------|------|
| L=1024 | block_q=512，仅 2 block，退化为类 XLA | 几乎 0 |
| L=4096 | 8 block，开始有 O(L) 优势 | 中等 |
| L=8192 | 16 block，充分 pipeline | **+53% tok/s**（行 3 vs 4） |

### 3.2 Vision ViT Attention — XLA（待优化）

**文件**：`jax_qwenvl/model/vit.py:200-230`

```python
attn_bias = block_diagonal_mask(cu_seqlens, ...)
attn_output = jax.nn.dot_product_attention(q, k, v, bias=attn_bias, scale=scaling)
```

未接 Splash 的原因：vision token 数随 batch 与图片数动态变化，需要 `segment_ids` 改造数据管道。

### 3.3 Quantization — Qwix QT FP8（已上线）

**文件**：`jax_qwenvl/train/train.py:344-372`

```python
if training_args.enable_fp8:
    rules = [
        qwix.QuantizationRule(
            module_path=r'.*(q_proj|k_proj|v_proj|o_proj)$',
            weight_qtype=jnp.float8_e4m3fn, act_qtype=jnp.float8_e4m3fn,
        ),
        qwix.QuantizationRule(
            module_path=r'.*(gate_proj|up_proj|down_proj)$',
            weight_qtype=jnp.float8_e4m3fn, act_qtype=jnp.float8_e4m3fn,
        ),
    ]
    model = qwix.quantize_model(model, qwix.QtProvider(rules))
```

| 项 | 值 |
|----|-----|
| API | `qwix.quantize_model(model, qwix.QtProvider(rules))` |
| 模式 | QT（forward 时 cast，weights 保持 BF16 便于梯度更新） |
| 覆盖 | 36 层 LLM × 7 Dense kernel = **252 / 370**（68%） |
| 不量化 | visual ViT 27 层、deepstack mergers、patch_embed、lm_head |
| dtype | **必须** `jnp.float8_e4m3fn`（dtype 对象），不能用字符串 `'fp8'` |
| 验证 | `PRINT_MODULE_PATHS=1` dry-run 拿真实 path + `jax.make_jaxpr` 扫 forward 图 |

### 3.4 Parallelism — Hybrid FSDP

| 集群 | mesh | 备注 |
|------|------|------|
| Single host (8 dev) | `dp=2, fsdp=4` | 当前 v7x-8 |
| Multi host v7x-16 (2 host) | `dp=4, fsdp=4` | 跨 host FSDP 受 ICI 带宽限制；checkpoint 用 `FSDP_DEVICES=4` 避开 Orbax 2-process allgather bug |

---

## 4. 失败记录

### 4.1 Pallas `flash_attention` + shard_map（弃用）

**结果**：L=8192 step_time 29.6s（XLA 5.22s 的 6× 慢）。已经过 5 轮 API 修复仍跑不出收益，最终改用 `splash_attention`（§ 3.1）。

| 修复过的 API 问题 | 错误 | 修复 |
|------------------|------|------|
| RoPE 后 dtype 不一致 | `value dtype should be float32, but got bfloat16` | q/k/v 统一 cast 到 hidden_states.dtype |
| Pallas bias 不接受广播 | `Attention bias shape mismatch` | `jnp.broadcast_to(ab, (B, H, L, L))` |
| `shard_map` 参数名变更 | `unexpected keyword 'check_rep'` | 改为 `check_vma=False` |
| `jax.experimental.shard_map` 已弃用 | DeprecationWarning | 改为 `jax.shard_map` |
| Pallas 不支持 GQA | `Head count mismatch: 32, 8, 8` | shard_map 内 `jnp.repeat` |

**6× 慢的根本原因**：shard_map dispatch 开销 + block_size=128 过小 + Pallas FA kernel 非最优。换 `splash_attention` 后这些问题都消失。

### 4.2 Qwix FP8 第一次尝试 — `f30026f`（silent fail，已修复）

| 错误 | 后果 | 修复 |
|------|------|------|
| `module_path=r'.*Dense.*'` 匹配 Flax 类名 | Qwix 实际匹配 instance name (`q_proj`/`Dense_0`)，0 layer 命中 | `r'.*(q_proj\|k_proj\|v_proj\|o_proj)$'` 等真实命名 |
| `weight_qtype='fp8'` 字符串 | Qwix 0.1.6+ 要求 dtype 对象 | `jnp.float8_e4m3fn` |
| 无任何验证 | log 写"FP8 enabled"但实际 BF16 跑 | 加 `PRINT_MODULE_PATHS=1` dry-run + `jax.make_jaxpr` 扫 forward |

### 4.3 GQA via MQA + 双 vmap（无收益已 revert）

尝试用 MQA 单 KV head + 在 KV groups 上额外 vmap，避免 `jnp.repeat`。实测 HBM/step_time 与 jnp.repeat 完全相同——K/V 展开占总 HBM 仅 0.05%，FFN intermediate 才是大头。

---

## 5. GKE 部署踩坑

| 问题 | 原因 | 解决 |
|------|------|------|
| nodeSelector 注入错误（调度失败） | `optimize-utilization-scheduler` 把 cpu/memory request 映射到 `cpu-np` 节点池 | pod spec 中**不要**请求 cpu/memory，只 `google.com/tpu: 4` |
| multi-host 调度失败 | v7x 多 host 必须用 `HIGH_THROUGHPUT` workload policy | `gcloud beta compute resource-policies create workload-policy NAME --type=HIGH_THROUGHPUT --accelerator-topology=2x2x2` |
| GCS 403 | 节点 SA 项目号不对 | 授权 `YOUR_COMPUTE_SA@developer.gserviceaccount.com` 到 bucket |
| HuggingFace 下载挂起 | Xet CDN 在 v7x pod 网络不可达 | 预下到 GCS 或 lustre PVC |
| Checkpoint allgather 死锁 | `FSDP_DEVICES=8` 下 Orbax 2-process bug | 用 `FSDP_DEVICES=4`（与 v6e 同 mesh） |
| 数据集 zip 启动慢（5-10 min） | 每次训练 Job 重新 unzip 118k 张图 | 一次性 `lustre-stage-llava.yaml` 把数据放到 lustre PVC，训练 Job 直挂 |
| `jax.make_jaxpr` 验证 FP8 失败 | dummy batch=1 无法 shard_map | 验证降级为 warning，靠 step_time + HBM 后置确认 |

---

## 6. 复现命令

```bash
# 一次性：把 LLaVA 数据从 GCS 拷到 lustre PVC（约 7 min, 18 GiB）
kubectl apply -f jax_qwenvl/gke/lustre-stage-llava.yaml
kubectl wait --for=condition=complete --timeout=900s job/lustre-stage-llava -n default

# 训练（编辑 yaml 内 BATCH_SIZE / RUN_NAME / ENABLE_FP8）
kubectl apply -f jax_qwenvl/gke/qwen3vl-8b-v7x-train-fast.yaml
kubectl logs -f -n default -l app=qwen3vl-v7x-fast

# 修 / 调整 Qwix pattern 时的 dry-run（拿真实 module path）
# 在 yaml 里设：PRINT_MODULE_PATHS=1, ENABLE_FP8=False，跑一次取 path 命名后退出
```

GKE manifest 清单（`jax_qwenvl/gke/`）：

| 文件 | 用途 |
|------|------|
| `lustre-stage-llava.yaml` | 一次性数据 staging（GCS → lustre PVC） |
| `qwen3vl-8b-v7x-train-fast.yaml` | 当前活跃训练 Job（v7x-8 single-host + lustre） |
| `qwen3vl-8b-v7x-train-job.yaml` | 历史 v7x-16 multi-host 训练 Job |
| `smoke-test-v7x.yaml` | TPU + 模型加载 smoke test |
| `verify-tpu-v7x-{4,8}chips.yaml` | 单 host 设备验证 |
| `verify-tpu-v6e-16.yaml` | v6e 对照参考 |

---

## 7. 下一步推 throughput 的剩余杠杆

| 方向 | 预估 step_time/MFU 收益 | 工作量 |
|------|----------------------|--------|
| **多 host 拼 16 dev**（FP8 + dp=4/fsdp=4） | tok/s ~25k（线性 scale） | 中（需 cluster 释放节点 + workload policy） |
| **FP8 扩到 vision ViT**（`visual/blocks_*/{linear_fc1,linear_fc2,attn/qkv,attn/proj}` 加规则） | step_time -10%~15%（覆盖剩 32% kernel） | 小（pattern + dry-run + 数值稳定性验证） |
| **vision ViT attention 接 Splash**（segment_ids 改造数据管道） | 长 patches 时显著 | 大 |
| **Pallas FFN 融合 kernel**（SwiGLU 一次 dispatch） | MFU +5-8 pp | 大 |
| **Splash block size 调大**（L≥4K 时 block_q=1024） | 长 L 训练时显著 | 小 |

---

## 附：参考资料

- MaxText attention 实现：https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/layers/attentions.py
- JAX Splash Attention：https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/
- Qwix 量化：https://github.com/google/qwix
- Google Ironwood 性能指南：https://docs.cloud.google.com/tpu/docs/ironwood-performance
- Splash Attention 详细实现报告：[`SPLASH_ATTENTION_REPORT.md`](./SPLASH_ATTENTION_REPORT.md)
