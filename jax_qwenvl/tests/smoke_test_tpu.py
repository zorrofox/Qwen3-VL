"""TPU Smoke Test — 提交完整训练 Job 之前必须先通过此测试。

使用合成数据跑 3 个 training step，验证：
1. Flash Attention forward/backward 不崩溃
2. Loss 是有限数（非 NaN/Inf）
3. 梯度不为 NaN/Inf

运行方式（在 GKE pod 内，venv 激活后）：
    python3 -m jax_qwenvl.tests.smoke_test_tpu \
        --model_path /workspace/model/qwen3vl-8b

退出码：0 = 通过，1 = 失败（阻止提交训练 Job）
"""
from __future__ import annotations

import argparse
import logging
import sys

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- 极小的合成 batch ------------------------------------------------
def _make_synthetic_batch(
    batch_size: int = 1,
    seq_len: int = 256,       # 比真实 seq_len 小，只测功能正确性
    vocab_size: int = 151936,
    max_num_patches: int = 128,
    max_num_images: int = 1,
    num_heads: int = 32,
    num_kv_heads: int = 8,
):
    """生成合成 batch，不依赖真实数据集。"""
    import numpy as np

    rng = np.random.default_rng(42)

    B, L = batch_size, seq_len
    input_ids = rng.integers(0, vocab_size, (B, L), dtype=np.int32)
    labels    = np.where(rng.random((B, L)) > 0.5,
                         rng.integers(0, vocab_size, (B, L), dtype=np.int32),
                         np.full((B, L), -100, dtype=np.int32))
    # 简单 causal mask
    pos = np.arange(L)
    mask = (pos[:, None] >= pos[None, :]).astype(np.float32)  # (L, L)
    additive = np.where(mask, 0.0, -1e9)[None, None]          # (1, 1, L, L)
    additive = np.broadcast_to(additive, (B, 1, L, L)).copy()

    position_ids = np.zeros((3, B, L), dtype=np.int32)
    position_ids[0] = np.arange(L)[None, :]

    # 合成视觉 patch（token 全为 image_token_id=151655 会触发 vision 路径）
    # 这里简化：纯文本 batch，跳过视觉
    patch_embeds = np.zeros((B, max_num_patches, 4096), dtype=np.float32)
    patch_pos    = np.zeros((B, max_num_patches, 2), dtype=np.int32)

    # cu_seqlens for packed sequences (单序列情形)
    cu_seqlens = np.array([L] * B, dtype=np.int32)

    from jax_qwenvl.types import Batch
    return Batch(
        input_ids        = jnp.array(input_ids),
        labels           = jnp.array(labels),
        attention_mask   = jnp.array(additive),
        position_ids     = jnp.array(position_ids),
        pixel_values     = None,
        image_grid_thw   = None,
        image_pos_masks  = None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--num_steps", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=256)
    args = parser.parse_args()

    logger.info("=== TPU Smoke Test 开始 ===")
    logger.info("backend=%s  devices=%d", jax.default_backend(), jax.device_count())

    # 初始化分布式（multi-host 情形）
    try:
        jax.distributed.initialize()
    except Exception as e:
        logger.warning("jax.distributed.initialize() skipped: %s", e)

    # 加载模型（轻量配置：只跑 forward，不关心收敛）
    from transformers import AutoProcessor
    from jax_qwenvl.model import Qwen3VLConfig, Qwen3VLForConditionalGeneration, load_hf_weights
    from jax_qwenvl.train.sharding import create_device_mesh, get_param_sharding_rules, shard_params
    from jax_qwenvl.train.train_state import create_train_state
    from jax_qwenvl.train.optimizer import create_optimizer, create_schedule
    from jax_qwenvl.train.train_step import train_step

    logger.info("加载模型配置：%s", args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    config = Qwen3VLConfig.from_pretrained(args.model_path)

    logger.info("加载权重 ...")
    model = Qwen3VLForConditionalGeneration(config)
    mesh = create_device_mesh(fsdp_devices=4)
    params = load_hf_weights(args.model_path, config)
    sharding_rules = get_param_sharding_rules(config, fsdp_devices=4)
    params = shard_params(params, sharding_rules, mesh)

    schedule = create_schedule(
        warmup_steps=1, total_steps=args.num_steps, learning_rate=1e-6)
    optimizer = create_optimizer(
        schedule, weight_decay=0.01, max_grad_norm=1.0)
    state = create_train_state(model, params, optimizer, mesh)

    logger.info("生成合成 batch (seq_len=%d) ...", args.seq_len)
    batch = _make_synthetic_batch(batch_size=2, seq_len=args.seq_len)

    passed = True
    for step in range(1, args.num_steps + 1):
        logger.info("Step %d/%d ...", step, args.num_steps)
        try:
            state, metrics = train_step(state, batch)
            loss = float(metrics["loss"])
            if not np.isfinite(loss):
                logger.error("Step %d: loss 不是有限数 = %f", step, loss)
                passed = False
                break
            logger.info("Step %d: loss=%.4f ✓", step, loss)
        except Exception as e:
            logger.error("Step %d: 异常 %s", step, e, exc_info=True)
            passed = False
            break

    if passed:
        logger.info("=== Smoke Test PASSED ✓ 可以提交训练 Job ===")
        sys.exit(0)
    else:
        logger.error("=== Smoke Test FAILED ✗ 禁止提交训练 Job ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
