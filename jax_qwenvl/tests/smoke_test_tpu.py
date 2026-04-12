"""TPU Smoke Test — 提交训练 Job 前必须通过此测试。

直接用真实训练形状测试 Pallas Flash Attention，不加载模型和数据集。
完成时间 < 2 分钟。

运行：
    source /opt/venv311/bin/activate
    python3 -m jax_qwenvl.tests.smoke_test_tpu

退出码：0 = PASSED（可提交训练），1 = FAILED（禁止提交）
"""
from __future__ import annotations
import logging
import sys
import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_pallas_flash_attention():
    """测试 Pallas Flash Attention 的真实训练形状。

    Qwen3-VL-8B：H=32, H_kv=8, D=128。
    用 L=1024（比 L=8192 快，但覆盖所有代码路径）。
    """
    from jax.experimental.pallas.ops.tpu import flash_attention as tpu_fa

    # 真实训练参数（GQA 展开后）
    B, H, L, D = 2, 32, 1024, 128
    scaling = D ** -0.5

    logger.info("测试形状：B=%d H=%d L=%d D=%d", B, H, L, D)

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.bfloat16)

    # Causal mask — 精确形状 (B, H, L, L)（与生产代码一致）
    pos = jnp.arange(L)
    causal = (pos[:, None] >= pos[None, :]).astype(jnp.bfloat16)
    additive = jnp.where(causal, jnp.zeros_like(causal),
                         jnp.full_like(causal, jnp.finfo(jnp.float32).min))
    ab = jnp.broadcast_to(additive[None, None], (B, H, L, L))

    logger.info("运行 Pallas flash_attention (forward) ...")
    out = tpu_fa.flash_attention(q, k, v, ab=ab, sm_scale=scaling)

    assert out.shape == (B, H, L, D), f"形状错误: {out.shape}"
    assert out.dtype == jnp.bfloat16, f"dtype 错误: {out.dtype}"
    assert np.isfinite(np.array(out).sum()), "输出含 NaN/Inf"
    logger.info("Forward pass: ✓  shape=%s  dtype=%s", out.shape, out.dtype)

    # 测试 backward（梯度计算）
    logger.info("运行 Pallas flash_attention (backward) ...")
    def fn(q, k, v):
        return tpu_fa.flash_attention(q, k, v, ab=ab, sm_scale=scaling).sum()

    grads = jax.grad(fn, argnums=(0, 1, 2))(q, k, v)
    for name, g in zip(["dq", "dk", "dv"], grads):
        assert g.shape == (B, H, L, D), f"{name} 形状错误: {g.shape}"
        assert np.isfinite(np.array(g).sum()), f"{name} 含 NaN/Inf"
    logger.info("Backward pass: ✓  dq/dk/dv 均有限")


def test_pallas_with_none_mask():
    """测试 ab=None（无 mask）时 Pallas 正常运行。"""
    from jax.experimental.pallas.ops.tpu import flash_attention as tpu_fa

    B, H, L, D = 1, 32, 512, 128
    key = jax.random.PRNGKey(99)
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.bfloat16)

    out = tpu_fa.flash_attention(q, k, v, ab=None, sm_scale=D**-0.5)
    assert out.shape == (B, H, L, D)
    logger.info("ab=None test: ✓")


def main():
    logger.info("=== TPU Smoke Test ===")
    logger.info("backend=%s  devices=%d", jax.default_backend(), jax.device_count())

    if jax.default_backend() != "tpu":
        logger.error("必须在 TPU 上运行（当前：%s）", jax.default_backend())
        sys.exit(1)

    tests = [
        ("Pallas Flash Attention forward+backward", test_pallas_flash_attention),
        ("Pallas Flash Attention ab=None", test_pallas_with_none_mask),
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
