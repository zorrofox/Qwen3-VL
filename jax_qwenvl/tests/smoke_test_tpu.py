"""TPU Smoke Test — 提交训练 Job 前必须通过此测试。

在真实 v7x TPU 上，使用与训练完全相同的 mesh + shard_map + Pallas 配置，
验证 Flash Attention forward+backward 正确运行。

运行：
    source /opt/venv311/bin/activate && python3 -m jax_qwenvl.tests.smoke_test_tpu

退出码：0 = PASSED，1 = FAILED
"""
from __future__ import annotations
import logging
import sys
import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_pallas_under_shard_map():
    """用与训练完全相同的 mesh + shard_map 路径测试 Pallas Flash Attention。

    配置：Qwen3-VL-8B，FSDP_DEVICES=4，hybrid 模式
    - mesh: dp=4, fsdp=4（16 设备）
    - 全局 batch=32，每设备 2 samples
    - H_q=32, H_kv=8, L=1024, D=128
    """
    from jax.experimental.pallas.ops.tpu import flash_attention as tpu_fa
    from jax import shard_map  # jax.experimental.shard_map 在 JAX 0.8+ 弃用
    from jax.sharding import PartitionSpec as P, NamedSharding
    from jax_qwenvl.train.sharding import create_device_mesh, register_global_mesh

    # 1. 创建与训练相同的 mesh（FSDP_DEVICES=4）
    mesh = create_device_mesh(dp=-1, fsdp=4)
    register_global_mesh(mesh, 'hybrid')
    logger.info("Mesh: dp=%d fsdp=%d total=%d", mesh.shape['dp'], mesh.shape['fsdp'],
                jax.device_count())

    # 2. 训练参数
    B_global = jax.device_count() * 2   # 每设备 2 samples，global=32
    H_q, H_kv, L, D = 32, 8, 1024, 128
    scaling = D ** -0.5
    batch_spec = P(('dp', 'fsdp'), None, None, None)

    # 3. 构造全局 sharded 张量（与 shard_batch 中 P(('dp','fsdp')) 一致）
    key = jax.random.PRNGKey(0)
    sharding = NamedSharding(mesh, batch_spec)
    q  = jax.device_put(jax.random.normal(key, (B_global, H_q, L, D), dtype=jnp.bfloat16), sharding)
    k  = jax.device_put(jax.random.normal(jax.random.fold_in(key,1), (B_global, H_kv, L, D), dtype=jnp.bfloat16), sharding)
    v  = jax.device_put(jax.random.normal(jax.random.fold_in(key,2), (B_global, H_kv, L, D), dtype=jnp.bfloat16), sharding)
    # bias: (B_global, 1, L, L) — 与生产代码一致
    ab = jax.device_put(jnp.zeros((B_global, 1, L, L), dtype=jnp.bfloat16), sharding)

    # 4. shard_map + Pallas（与 llm.py TextAttention 完全相同路径）
    def _pallas_attn(q_l, k_l, v_l, ab_l):
        B_l, H_q_l, L_q, _ = q_l.shape
        # Pallas 不支持 GQA，展开 K/V（H_q=32, H_kv=8, groups=4）
        num_kv_groups = H_q // H_kv
        if num_kv_groups > 1:
            k_l = jnp.repeat(k_l, num_kv_groups, axis=1)
            v_l = jnp.repeat(v_l, num_kv_groups, axis=1)
        ab_full = jnp.broadcast_to(ab_l, (B_l, H_q_l, L_q, L_q))
        return tpu_fa.flash_attention(q_l, k_l, v_l, ab=ab_full, sm_scale=scaling)

    logger.info("运行 shard_map + Pallas forward (B=%d H_q=%d H_kv=%d L=%d D=%d) ...",
                B_global, H_q, H_kv, L, D)
    out = shard_map(
        _pallas_attn,
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, batch_spec),
        out_specs=batch_spec,
        check_vma=False,
    )(q, k, v, ab)

    assert out.shape == (B_global, H_q, L, D), f"shape 错误: {out.shape}"
    assert out.dtype == jnp.bfloat16, f"dtype 错误: {out.dtype}"
    assert np.isfinite(np.array(out.sum())), "输出含 NaN/Inf"
    logger.info("Forward: ✓  shape=%s  dtype=%s", out.shape, out.dtype)

    # 5. Backward（梯度计算）— 最关键的：确认 shard_map + Pallas 在 grad 下正常
    def fn(q, k, v, ab):
        return shard_map(
            _pallas_attn,
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec, batch_spec),
            out_specs=batch_spec,
            check_vma=False,
        )(q, k, v, ab).sum()

    logger.info("运行 shard_map + Pallas backward (grad) ...")
    dq, dk, dv = jax.grad(fn, argnums=(0, 1, 2))(q, k, v, ab)
    assert dq.shape == (B_global, H_q, L, D)
    assert np.isfinite(np.array(dq.sum())), "dq 含 NaN/Inf"
    assert np.isfinite(np.array(dk.sum())), "dk 含 NaN/Inf"
    logger.info("Backward: ✓  dq/dk/dv 均有限")


def test_vit_attention_fallback():
    """VisionAttention 使用 dot_product_attention（无 GQA，简单 mask）。"""
    S, H, D = 512, 16, 64
    scaling = D ** -0.5
    key = jax.random.PRNGKey(1)
    q = jax.random.normal(key, (1, S, H, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (1, S, H, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (1, S, H, D), dtype=jnp.bfloat16)
    mask = jnp.zeros((1, 1, S, S), dtype=jnp.bfloat16)

    def vit_attn(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, bias=mask, scale=scaling).sum()

    dq, dk, dv = jax.jit(jax.grad(vit_attn, argnums=(0, 1, 2)))(q, k, v)
    assert dq.shape == q.shape
    assert np.isfinite(np.array(dq.sum()))
    logger.info("VisionAttention fallback: ✓")


def main():
    logger.info("=== TPU Smoke Test ===")
    logger.info("backend=%s  devices=%d", jax.default_backend(), jax.device_count())

    if jax.default_backend() != "tpu":
        logger.error("必须在 TPU 上运行（当前：%s）", jax.default_backend())
        sys.exit(1)

    tests = [
        ("Pallas FA via shard_map (hybrid mesh, forward+backward)", test_pallas_under_shard_map),
        ("VisionAttention dot_product_attention fallback", test_vit_attention_fallback),
    ]

    passed = True
    for name, fn in tests:
        logger.info("--- %s ---", name)
        try:
            fn()
            logger.info("%s: PASSED ✓", name)
        except Exception:
            logger.exception("%s: FAILED ✗", name)
            passed = False

    if passed:
        logger.info("=== Smoke Test PASSED ✓ 可以提交训练 Job ===")
        sys.exit(0)
    else:
        logger.error("=== Smoke Test FAILED ✗ 禁止提交训练 Job ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
