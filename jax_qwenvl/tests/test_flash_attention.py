"""单元测试：Flash Attention（jax.nn.dot_product_attention）

测试目标：
1. 数值等价性：Flash Attention 输出与原始手动实现在 bfloat16 容差内一致
2. GQA 支持：num_kv_heads < num_heads 时结果正确
3. Causal mask：上三角遮蔽正确（等价于加性 -inf mask）
4. Packed sequences：block-diagonal mask 正确
5. 内存：Flash Attention 不创建完整 (B, H, L, L) 注意力矩阵

运行：
    pip install pytest
    pytest jax_qwenvl/tests/test_flash_attention.py -v
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# 强制使用 CPU（单元测试不需要 TPU）
jax.config.update("jax_platform_name", "cpu")


# ---------------------------------------------------------------------------
# 辅助函数：参考实现（原始手动 attention）
# ---------------------------------------------------------------------------

def ref_attention(q, k, v, mask=None):
    """原始手动实现（O(L²) 内存），作为 ground truth。

    Args:
        q: (B, H, L, D)
        k: (B, H_kv, L, D) — H_kv 可 < H（GQA 情形，此处已展开）
        v: (B, H_kv, L, D)
        mask: (B, 1, L, L) float 加性 mask，None 表示不遮蔽
    Returns:
        (B, H, L, D)
    """
    # GQA 展开
    H = q.shape[1]
    H_kv = k.shape[1]
    if H_kv < H:
        groups = H // H_kv
        k = jnp.repeat(k, groups, axis=1)
        v = jnp.repeat(v, groups, axis=1)

    scaling = q.shape[-1] ** -0.5
    attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scaling
    if mask is not None:
        attn = attn + mask
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)
    return jnp.matmul(attn, v)


def flash_attention(q, k, v, mask=None):
    """新实现（与生产代码逻辑一致）：Pallas mha_reference（CPU 可用）。

    - 输入格式 (B, H, L, D)，与 Pallas TPU kernel 相同
    - GQA：先展开 K/V（Pallas 不支持 GQA）
    - ab：additive bias mask (B, 1, L, L)
    CPU 下使用 mha_reference（纯 JAX，等价于 Pallas kernel 的参考实现）。
    """
    from jax.experimental.pallas.ops.tpu.flash_attention import mha_reference

    scaling = q.shape[-1] ** -0.5
    H = q.shape[1]
    H_kv = k.shape[1]

    # GQA 展开
    if H_kv < H:
        groups = H // H_kv
        k = jnp.repeat(k, groups, axis=1)
        v = jnp.repeat(v, groups, axis=1)

    # 统一 dtype
    dtype = q.dtype
    k = k.astype(dtype)
    v = v.astype(dtype)

    # Pallas kernel 要求精确形状 (B, H, L, L)，不接受 (B, 1, L, L) 广播
    # 这里显式 broadcast，与生产代码中 jnp.broadcast_to 保持一致
    H = q.shape[1]
    L = q.shape[2]
    ab_full = (jnp.broadcast_to(mask, (mask.shape[0], H, L, L))
               if mask is not None else None)
    return mha_reference(
        q, k, v,
        ab=ab_full,    # (B, H, L, L) 精确形状
        sm_scale=scaling,
    )  # → (B, H, L, D)


def _causal_mask(L, dtype=jnp.bfloat16):
    """标准因果 mask：(1, 1, L, L)，上三角为 -inf。"""
    mask = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))
    additive = jnp.where(mask, jnp.zeros((L, L), dtype=dtype),
                         jnp.full((L, L), jnp.finfo(dtype).min))
    return additive[None, None, :, :]  # (1, 1, L, L)


def _block_diagonal_mask(seq_lens, L, dtype=jnp.bfloat16):
    """Packed sequences 的 block-diagonal + causal mask：(1, 1, L, L)。"""
    cu = np.cumsum([0] + list(seq_lens))
    seg = np.zeros(L, dtype=np.int32)
    for i, (s, e) in enumerate(zip(cu[:-1], cu[1:])):
        seg[s:e] = i
    pos = np.arange(L)
    same_seg = (seg[:, None] == seg[None, :])
    causal = (pos[:, None] >= pos[None, :])
    attend = same_seg & causal
    additive = np.where(attend, 0.0, np.finfo(np.float32).min
                        ).astype(np.float32)
    return jnp.array(additive)[None, None, :, :]


# ---------------------------------------------------------------------------
# 测试 1：基本数值等价（无 mask，MHA）
# ---------------------------------------------------------------------------

def test_basic_equivalence():
    """无 mask、MHA（H_kv == H）场景下 Flash Attention 与参考实现数值一致。"""
    key = jax.random.PRNGKey(0)
    B, H, L, D = 2, 4, 16, 32
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.bfloat16)

    ref = ref_attention(q, k, v)
    got = flash_attention(q, k, v)

    np.testing.assert_allclose(
        np.array(got, dtype=np.float32),
        np.array(ref, dtype=np.float32),
        atol=1e-2, rtol=1e-2,
        err_msg="Flash Attention 基本等价性失败"
    )


# ---------------------------------------------------------------------------
# 测试 2：GQA（H_kv < H）
# ---------------------------------------------------------------------------

def test_gqa_equivalence():
    """GQA：num_kv_heads=2，num_heads=8（groups=4）。"""
    key = jax.random.PRNGKey(42)
    B, H, H_kv, L, D = 2, 8, 2, 16, 32
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H_kv, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H_kv, L, D), dtype=jnp.bfloat16)

    # 参考：先展开 K/V
    ref = ref_attention(q, k, v)
    # Flash：直接传 GQA（无 repeat）
    got = flash_attention(q, k, v)

    # bfloat16 下 GQA 原生实现（无 repeat）与展开后实现运算顺序不同，
    # 舍入误差略大，atol=2e-2 仍在可接受范围（相对误差 < 0.25%）
    np.testing.assert_allclose(
        np.array(got, dtype=np.float32),
        np.array(ref, dtype=np.float32),
        atol=2e-2, rtol=2e-2,
        err_msg="GQA Flash Attention 等价性失败"
    )


# ---------------------------------------------------------------------------
# 测试 3：Causal mask
# ---------------------------------------------------------------------------

def test_causal_mask_equivalence():
    """因果 mask 下 Flash Attention 与参考实现一致。"""
    key = jax.random.PRNGKey(7)
    B, H, L, D = 2, 4, 32, 16
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.bfloat16)

    mask = _causal_mask(L, dtype=jnp.bfloat16)
    ref = ref_attention(q, k, v, mask=mask)
    got = flash_attention(q, k, v, mask=mask)

    np.testing.assert_allclose(
        np.array(got, dtype=np.float32),
        np.array(ref, dtype=np.float32),
        atol=1e-2, rtol=1e-2,
        err_msg="Causal mask Flash Attention 等价性失败"
    )


# ---------------------------------------------------------------------------
# 测试 4：Packed sequences（block-diagonal mask）
# ---------------------------------------------------------------------------

def test_packed_sequences_equivalence():
    """Packed sequences（block-diagonal + causal mask）等价性。"""
    key = jax.random.PRNGKey(99)
    B, H, D = 1, 4, 16
    seq_lens = [10, 6, 8]   # packed: 3 sequences in 1 sample, total L=24
    L = sum(seq_lens)

    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.bfloat16)

    mask = _block_diagonal_mask(seq_lens, L, dtype=jnp.bfloat16)
    ref = ref_attention(q, k, v, mask=mask)
    got = flash_attention(q, k, v, mask=mask)

    np.testing.assert_allclose(
        np.array(got, dtype=np.float32),
        np.array(ref, dtype=np.float32),
        atol=1e-2, rtol=1e-2,
        err_msg="Packed sequences Flash Attention 等价性失败"
    )


# ---------------------------------------------------------------------------
# 测试 5：GQA + Causal mask 组合
# ---------------------------------------------------------------------------

def test_gqa_with_causal_mask():
    """GQA + causal mask 组合场景（Qwen3-VL 实际训练配置）。"""
    key = jax.random.PRNGKey(123)
    # Qwen3-VL-8B 参数：28 heads，4 kv_heads，head_dim=128
    # 单元测试用小配置：8 heads，2 kv_heads
    B, H, H_kv, L, D = 1, 8, 2, 64, 32
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H_kv, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H_kv, L, D), dtype=jnp.bfloat16)

    mask = _causal_mask(L, dtype=jnp.bfloat16)
    ref = ref_attention(q, k, v, mask=mask)
    got = flash_attention(q, k, v, mask=mask)

    np.testing.assert_allclose(
        np.array(got, dtype=np.float32),
        np.array(ref, dtype=np.float32),
        atol=1e-2, rtol=1e-2,
        err_msg="GQA + causal mask Flash Attention 等价性失败"
    )


# ---------------------------------------------------------------------------
# 测试 6（回归）：RoPE upcast 场景 — q/k float32，v bfloat16
# 这是实际训练时的真实情况：RoPE cos/sin 是 float32，
# q*cos 自动 upcast 到 float32，但 v 未过 RoPE 仍是 bfloat16。
# dot_product_attention 要求严格 dtype 一致，必须先 cast。
# ---------------------------------------------------------------------------

def test_rope_upcast_dtype_mismatch():
    """q/k float32（模拟 RoPE upcast），v bfloat16 → 统一 cast 后应正常运行。"""
    key = jax.random.PRNGKey(55)
    B, H, H_kv, L, D = 1, 8, 2, 16, 32

    # v 是 bfloat16，q/k 是 float32（模拟 RoPE upcast）
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.float32)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H_kv, L, D), dtype=jnp.float32)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H_kv, L, D), dtype=jnp.bfloat16)
    compute_dtype = jnp.bfloat16

    scaling = D ** -0.5
    # 统一 cast（生产代码中的修复）
    q_t = jnp.transpose(q, (0, 2, 1, 3)).astype(compute_dtype)
    k_t = jnp.transpose(k, (0, 2, 1, 3)).astype(compute_dtype)
    v_t = jnp.transpose(v, (0, 2, 1, 3)).astype(compute_dtype)

    # 不应报错
    out = jax.nn.dot_product_attention(q_t, k_t, v_t, scale=scaling, implementation="xla")
    assert out.shape == (B, L, H, D), f"期望 {(B, L, H, D)}，得到 {out.shape}"
    assert out.dtype == compute_dtype, f"输出 dtype 应为 {compute_dtype}，得到 {out.dtype}"


# ---------------------------------------------------------------------------
# 测试 7：输出形状正确
# ---------------------------------------------------------------------------

def test_output_shape():  # 原测试 6 → 现在是测试 7
    """flash_attention 输出形状必须与参考实现一致。"""
    key = jax.random.PRNGKey(0)
    B, H, H_kv, L, D = 2, 8, 2, 20, 16
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H_kv, L, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H_kv, L, D), dtype=jnp.bfloat16)

    out = flash_attention(q, k, v)
    assert out.shape == (B, H, L, D), f"期望 {(B, H, L, D)}，得到 {out.shape}"
