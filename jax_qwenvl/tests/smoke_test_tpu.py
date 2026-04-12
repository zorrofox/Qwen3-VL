"""TPU Smoke Test — 提交训练 Job 前必须通过此测试。

测试实际使用的 attention 实现在 TPU 上能正确运行，
包括在 jax.jit + jax.grad 下（模拟 SPMD 分区检查）。

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


def test_dot_product_attention_spmd():
    """测试 jax.nn.dot_product_attention 在 jit+grad 下（模拟 SPMD）。

    Qwen3-VL-8B 实际参数：H=32，H_kv=8，D=128，L=1024（缩小版）。
    关键：用 jax.jit + jax.grad 触发 XLA 编译和分区检查。
    如果 attention 实现在 SPMD 下不兼容，这里会报错。
    """
    B, H, H_kv, L, D = 2, 32, 8, 1024, 128
    scaling = D ** -0.5

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H_kv, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H_kv, L, D), dtype=jnp.bfloat16)

    # Causal mask (B, 1, L, L)
    pos = jnp.arange(L)
    causal = (pos[:, None] >= pos[None, :]).astype(jnp.bfloat16)
    additive = jnp.where(causal, 0.0, jnp.finfo(jnp.float32).min).astype(jnp.bfloat16)
    mask = additive[None, None, :, :]  # (1, 1, L, L)

    # 模拟生产代码：GQA 展开 + dtype cast + dot_product_attention
    def attention_fn(q, k, v):
        # GQA 展开
        groups = H // H_kv
        k_exp = jnp.repeat(k, groups, axis=1)
        v_exp = jnp.repeat(v, groups, axis=1)
        # RoPE upcast 模拟：k/q 变成 float32，v 仍 bfloat16，统一 cast
        dtype = jnp.bfloat16
        q_t = jnp.transpose(q.astype(dtype), (0, 2, 1, 3))   # (B, L, H, D)
        k_t = jnp.transpose(k_exp.astype(dtype), (0, 2, 1, 3))
        v_t = jnp.transpose(v_exp.astype(dtype), (0, 2, 1, 3))
        out = jax.nn.dot_product_attention(q_t, k_t, v_t, bias=mask, scale=scaling)
        return out.sum()

    # 关键：在 jax.jit + jax.grad 下运行（触发 XLA 编译和 SPMD 分区检查）
    jit_grad_fn = jax.jit(jax.grad(attention_fn, argnums=(0, 1, 2)))
    logger.info("运行 attention forward+backward (jit+grad, B=%d H=%d L=%d D=%d) ...", B, H, L, D)
    dq, dk, dv = jit_grad_fn(q, k, v)

    assert dq.shape == (B, H, L, D), f"dq shape 错误: {dq.shape}"
    assert dk.shape == (B, H_kv, L, D), f"dk shape 错误: {dk.shape}"
    assert np.isfinite(np.array(dq).sum()), "dq 含 NaN/Inf"
    assert np.isfinite(np.array(dk).sum()), "dk 含 NaN/Inf"
    logger.info("attention jit+grad: ✓")


def test_vit_attention_spmd():
    """测试 VisionAttention（无 GQA，block-diagonal mask）在 jit+grad 下。"""
    S, H, D = 512, 16, 64  # 缩小版视觉 attention
    scaling = D ** -0.5
    key = jax.random.PRNGKey(1)
    q = jax.random.normal(key, (1, S, H, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (1, S, H, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (1, S, H, D), dtype=jnp.bfloat16)
    # 简单全 attend mask
    mask = jnp.zeros((1, 1, S, S), dtype=jnp.bfloat16)

    def vit_attn(q, k, v):
        out = jax.nn.dot_product_attention(q, k, v, bias=mask, scale=scaling)
        return out.sum()

    logger.info("运行 VisionAttention jit+grad (S=%d H=%d) ...", S, H)
    dq, dk, dv = jax.jit(jax.grad(vit_attn, argnums=(0, 1, 2)))(q, k, v)
    assert dq.shape == q.shape
    assert np.isfinite(np.array(dq).sum())
    logger.info("VisionAttention jit+grad: ✓")


def main():
    logger.info("=== TPU Smoke Test ===")
    logger.info("backend=%s  devices=%d", jax.default_backend(), jax.device_count())

    if jax.default_backend() != "tpu":
        logger.error("必须在 TPU 上运行（当前：%s）", jax.default_backend())
        sys.exit(1)

    tests = [
        ("TextAttention (GQA, causal mask, jit+grad)", test_dot_product_attention_spmd),
        ("VisionAttention (full mask, jit+grad)", test_vit_attention_spmd),
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
