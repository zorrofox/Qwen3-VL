"""TPU Smoke Test — 提交训练 Job 前必须通过此测试。

使用与训练完全相同的 mesh + shard_map(vmap(splash_kernel)) 路径，
验证 Splash Attention forward+backward 在真实 v7x 上正确运行。

运行：
    source /opt/venv311/bin/activate && python3 -m jax_qwenvl.tests.smoke_test_tpu

退出码：0 = PASSED，1 = FAILED
"""
from __future__ import annotations
import logging, sys, functools
import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_splash_attention_under_shard_map():
    """用与训练完全相同的 mesh + shard_map(vmap(splash_kernel)) 测试 Splash Attention。

    配置与 Qwen3-VL-8B 训练一致：
    - mesh: dp=4, fsdp=4（hybrid，16 设备）
    - H_q=32, H_kv=8, L=1024, D=128（GQA 展开在 _splash_fn 内完成）
    - 无 segment_ids（标准非 packed 序列）
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sak,
        splash_attention_mask as sam,
    )
    from jax import shard_map
    from jax.sharding import PartitionSpec as P, NamedSharding
    from jax_qwenvl.train.sharding import create_device_mesh, register_global_mesh

    # 1. 创建与训练相同的 mesh
    mesh = create_device_mesh(dp=-1, fsdp=4)
    register_global_mesh(mesh, 'hybrid')
    logger.info("Mesh: dp=%d fsdp=%d total=%d", mesh.shape['dp'], mesh.shape['fsdp'],
                jax.device_count())

    # 2. 训练参数（Qwen3-VL-8B 真实形状）
    B_global = jax.device_count() * 2   # 32
    H_q, H_kv, L, D = 32, 8, 1024, 128
    num_kv_groups = H_q // H_kv        # 4
    batch_spec = P(('dp', 'fsdp'), None, None, None)

    # 3. 构造 Splash MHA kernel（GQA 通过 jnp.repeat 展开 K/V）
    num_kv_groups = H_q // H_kv  # 4
    block_q = min(512, L)
    block_kv = min(512, L)
    causal_mask = sam.CausalMask(shape=(L, L))
    multi_head_mask = sam.MultiHeadMask(masks=(causal_mask,) * H_q)
    splash_kernel = sak.make_splash_mha(
        mask=multi_head_mask,
        block_sizes=sak.BlockSizes(
            block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv,
            block_q_dkv=block_q, block_kv_dkv=block_kv,
            block_kv_dkv_compute=block_kv,
            block_q_dq=block_q, block_kv_dq=block_kv,
        ),
        head_shards=1, q_seq_shards=1,
    )

    # 4. 生成全局 sharded 张量
    sharding = NamedSharding(mesh, batch_spec)
    key = jax.random.PRNGKey(0)
    q = jax.device_put(jax.random.normal(key, (B_global, H_q, L, D), dtype=jnp.bfloat16), sharding)
    k = jax.device_put(jax.random.normal(jax.random.fold_in(key,1), (B_global, H_kv, L, D), dtype=jnp.bfloat16), sharding)
    v = jax.device_put(jax.random.normal(jax.random.fold_in(key,2), (B_global, H_kv, L, D), dtype=jnp.bfloat16), sharding)

    # 5. GQA via jnp.repeat + vmap over batch（与生产代码一致）
    def _splash_fn(q_l, k_l, v_l):
        if num_kv_groups > 1:
            k_l = jnp.repeat(k_l, num_kv_groups, axis=1)
            v_l = jnp.repeat(v_l, num_kv_groups, axis=1)
        return jax.vmap(splash_kernel)(q_l, k_l, v_l)

    logger.info("运行 shard_map(vmap(splash_kernel)) forward (B=%d H_q=%d H_kv=%d L=%d D=%d) ...",
                B_global, H_q, H_kv, L, D)
    out = shard_map(
        _splash_fn, mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec),
        out_specs=batch_spec,
        check_vma=False,
    )(q, k, v)

    assert out.shape == (B_global, H_q, L, D), f"shape 错误: {out.shape}"
    assert out.dtype == jnp.bfloat16, f"dtype 错误: {out.dtype}"
    assert np.isfinite(np.array(out.sum())), "输出含 NaN/Inf"
    logger.info("Forward: ✓  shape=%s", out.shape)

    # 6. Backward（梯度计算）— shard_map(vmap) 下正常
    def fn(q, k, v):
        return shard_map(
            _splash_fn, mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec),
            out_specs=batch_spec,
            check_vma=False,
        )(q, k, v).sum()

    logger.info("运行 backward (grad) ...")
    dq, dk, dv = jax.grad(fn, argnums=(0, 1, 2))(q, k, v)
    assert dq.shape == (B_global, H_q, L, D)
    assert np.isfinite(np.array(dq.sum())), "dq 含 NaN/Inf"
    assert np.isfinite(np.array(dk.sum())), "dk 含 NaN/Inf"
    logger.info("Backward: ✓")


def test_vit_attention_fallback():
    """VisionAttention 使用 dot_product_attention（无 GQA）。"""
    S, H, D = 512, 16, 64
    key = jax.random.PRNGKey(1)
    q = jax.random.normal(key, (1, S, H, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (1, S, H, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (1, S, H, D), dtype=jnp.bfloat16)
    mask = jnp.zeros((1, 1, S, S), dtype=jnp.bfloat16)

    dq, dk, dv = jax.jit(jax.grad(
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, bias=mask, scale=D**-0.5).sum(),
        argnums=(0, 1, 2)
    ))(q, k, v)
    assert dq.shape == q.shape and np.isfinite(np.array(dq.sum()))
    logger.info("VisionAttention fallback: ✓")


def main():
    logger.info("=== TPU Smoke Test (Splash Attention) ===")
    logger.info("backend=%s  devices=%d", jax.default_backend(), jax.device_count())

    if jax.default_backend() != "tpu":
        logger.error("必须在 TPU 上运行（当前：%s）", jax.default_backend())
        sys.exit(1)

    tests = [
        ("Splash Attention shard_map(vmap) forward+backward", test_splash_attention_under_shard_map),
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
