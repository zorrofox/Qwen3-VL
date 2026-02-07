"""TPU validation script for JAX Qwen3-VL training pipeline.

Runs 8 tests on real TPU hardware to verify the full training stack:
1. TPU device detection
2. Model weight loading from HuggingFace
3. Forward inference (bfloat16)
4. SPMD sharding (DP mode)
5. Single training step (bfloat16)
6. Checkpoint save/restore round-trip
7. Gradient accumulation
8. FSDP sharding mode

All tests use bfloat16 parameters to match production TPU training.

Usage:
    python3 -m jax_qwenvl.scripts.tpu_validate
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_HF_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_PATH = None  # Set after snapshot_download
SEQ_LEN = 32
BATCH_SIZE = 1  # Single-device tests; multi-device tests use num_devices
PARAM_DTYPE = jnp.bfloat16  # Use bfloat16 for all tests

results = []


def download_model():
    """Download model from HuggingFace Hub and return local path."""
    global MODEL_PATH
    from huggingface_hub import snapshot_download

    logger.info("Downloading model %s from HuggingFace Hub...", MODEL_HF_ID)
    MODEL_PATH = snapshot_download(
        MODEL_HF_ID,
        ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.onnx"],
    )
    logger.info("Model downloaded to %s", MODEL_PATH)
    return MODEL_PATH


def record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    logger.info("[%s] %s %s", status, name, f"-- {detail}" if detail else "")


def _cast_to_bf16(params):
    """Cast all float parameters to bfloat16."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        params,
    )


# ---------------------------------------------------------------------------
# Test 1: TPU device detection
# ---------------------------------------------------------------------------
def test_tpu_devices():
    devices = jax.devices()
    backend = jax.default_backend()
    num_devices = len(devices)
    detail = f"backend={backend}, num_devices={num_devices}, devices={devices}"
    record("TPU device detection", backend == "tpu" and num_devices >= 1, detail)
    return num_devices


# ---------------------------------------------------------------------------
# Test 2: Model weight loading
# ---------------------------------------------------------------------------
def test_weight_loading():
    from jax_qwenvl.model import Qwen3VLConfig, load_hf_weights

    config = Qwen3VLConfig.from_pretrained(MODEL_PATH)
    loaded = load_hf_weights(MODEL_PATH, config, lora_rank=0)
    params = loaded["params"]

    # Cast to bfloat16
    params = _cast_to_bf16(params)

    # Count parameters and verify dtype
    leaves = jax.tree_util.tree_leaves(params)
    num_params = sum(x.size for x in leaves)
    first_dtype = leaves[0].dtype
    mem_gb = sum(x.size * x.dtype.itemsize for x in leaves) / 1e9
    detail = (
        f"num_params={num_params:,}, dtype={first_dtype}, "
        f"memory={mem_gb:.2f}GB, "
        f"config.text.hidden_size={config.text_config.hidden_size}"
    )
    record("Model weight loading (bf16)", num_params > 0 and first_dtype == jnp.bfloat16, detail)
    return config, params


# ---------------------------------------------------------------------------
# Test 3: Forward inference
# ---------------------------------------------------------------------------
def test_forward(config, params):
    from jax_qwenvl.model import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration(config=config)

    # Dummy inputs
    input_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    position_ids = jnp.zeros((3, BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)

    # Run forward
    t0 = time.time()
    logits, loss = model.apply({"params": params}, input_ids, position_ids)
    logits.block_until_ready()
    elapsed = time.time() - t0

    has_nan = bool(jnp.any(jnp.isnan(logits)))
    has_inf = bool(jnp.any(jnp.isinf(logits)))
    detail = (
        f"logits.shape={logits.shape}, dtype={logits.dtype}, "
        f"has_nan={has_nan}, has_inf={has_inf}, loss={loss}, "
        f"elapsed={elapsed:.2f}s"
    )
    record(
        "Forward inference (bf16)",
        logits.shape == (BATCH_SIZE, SEQ_LEN, config.text_config.vocab_size)
        and not has_nan
        and not has_inf,
        detail,
    )
    return model


# ---------------------------------------------------------------------------
# Test 4: SPMD sharding (DP mode)
# ---------------------------------------------------------------------------
def test_spmd_sharding(params, num_devices):
    from jax_qwenvl.train.sharding import (
        create_device_mesh,
        get_param_sharding_rules,
        shard_params,
        shard_batch,
    )
    from jax_qwenvl.types import Batch

    mesh = create_device_mesh(dp=num_devices, fsdp=1)
    rules = get_param_sharding_rules("dp")

    with mesh:
        sharded_params = shard_params(params, mesh, rules)

    # Verify params are on the mesh
    first_leaf = jax.tree_util.tree_leaves(sharded_params)[0]
    sharding_info = str(first_leaf.sharding) if hasattr(first_leaf, "sharding") else "unknown"

    # Shard a dummy batch (batch_size must be divisible by num_devices for DP)
    dp_batch = num_devices
    dummy_batch = Batch(
        input_ids=np.ones((dp_batch, SEQ_LEN), dtype=np.int32),
        labels=np.ones((dp_batch, SEQ_LEN), dtype=np.int32) * -100,
        attention_mask=np.ones((dp_batch, SEQ_LEN), dtype=np.int32),
        position_ids=np.zeros((3, dp_batch, SEQ_LEN), dtype=np.int32),
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    )
    with mesh:
        sharded_batch = shard_batch(dummy_batch, mesh)

    detail = f"mesh={mesh}, param_sharding={sharding_info}, dtype={first_leaf.dtype}"
    record("SPMD sharding (DP)", True, detail)
    return mesh, sharded_params


# ---------------------------------------------------------------------------
# Test 5: Single training step
# ---------------------------------------------------------------------------
def test_train_step(model, params, num_devices):
    from jax_qwenvl.train.sharding import (
        create_device_mesh,
        get_param_sharding_rules,
        shard_params,
        shard_batch,
    )
    from jax_qwenvl.train.optimizer import create_optimizer
    from jax_qwenvl.train.train_state import create_train_state
    from jax_qwenvl.train.train_step import train_step
    from jax_qwenvl.types import Batch

    mesh = create_device_mesh(dp=num_devices, fsdp=1)
    rules = get_param_sharding_rules("dp")

    with mesh:
        sharded_params = shard_params(params, mesh, rules)

    optimizer, _ = create_optimizer(
        params=sharded_params,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=10,
        total_steps=100,
        tune_llm=True,
    )
    state = create_train_state(model, sharded_params, optimizer)

    # Check optimizer state dtype
    opt_leaves = jax.tree_util.tree_leaves(state.opt_state)
    opt_float_leaves = [x for x in opt_leaves if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating)]
    opt_dtype = opt_float_leaves[0].dtype if opt_float_leaves else "none"
    opt_mem_gb = sum(x.size * x.dtype.itemsize for x in opt_float_leaves) / 1e9

    # Create a batch with labels (batch_size = num_devices for DP)
    dp_batch = num_devices
    dummy_batch = Batch(
        input_ids=np.ones((dp_batch, SEQ_LEN), dtype=np.int32),
        labels=np.concatenate([
            np.full((dp_batch, 1), -100, dtype=np.int32),
            np.ones((dp_batch, SEQ_LEN - 1), dtype=np.int32),
        ], axis=1),
        attention_mask=np.ones((dp_batch, SEQ_LEN), dtype=np.int32),
        position_ids=np.zeros((3, dp_batch, SEQ_LEN), dtype=np.int32),
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    )

    with mesh:
        sharded_batch = shard_batch(dummy_batch, mesh)

        t0 = time.time()
        new_state, metrics = train_step(state, sharded_batch)
        loss_val = float(metrics["loss"])
        jax.block_until_ready(new_state.params)
        elapsed = time.time() - t0

    is_finite = np.isfinite(loss_val)
    step_incremented = int(new_state.step) == 1
    detail = (
        f"loss={loss_val:.4f}, step={int(new_state.step)}, "
        f"param_dtype={first_dtype(new_state.params)}, "
        f"opt_dtype={opt_dtype}, opt_mem={opt_mem_gb:.2f}GB, "
        f"elapsed={elapsed:.2f}s"
    )
    record("Single training step (bf16)", is_finite and step_incremented, detail)
    # Return new_state (not state) because train_step donates the old state buffers
    return new_state, mesh


def first_dtype(params):
    """Get dtype of the first float leaf in params."""
    for leaf in jax.tree_util.tree_leaves(params):
        if hasattr(leaf, 'dtype') and jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf.dtype
    return "unknown"


# ---------------------------------------------------------------------------
# Test 6: Checkpoint round-trip
# ---------------------------------------------------------------------------
def test_checkpoint(state, mesh):
    import orbax.checkpoint as ocp

    tmpdir = tempfile.mkdtemp(prefix="tpu_ckpt_test_")
    try:
        # Use synchronous checkpointing to avoid async GC race
        options = ocp.CheckpointManagerOptions(
            max_to_keep=2,
            save_interval_steps=1,
            enable_async_checkpointing=False,
        )
        mgr = ocp.CheckpointManager(tmpdir, options=options)
        with mesh:
            mgr.save(1, args=ocp.args.StandardSave(state))
            mgr.wait_until_finished()
            restored = mgr.restore(
                1, args=ocp.args.StandardRestore(state)
            )

        if restored is None:
            record("Checkpoint round-trip", False, "restore returned None")
            return

        # Compare a few param leaves
        orig_leaves = jax.tree_util.tree_leaves(state.params)
        rest_leaves = jax.tree_util.tree_leaves(restored.params)
        max_diff = max(
            float(jnp.max(jnp.abs(o.astype(jnp.float32) - r.astype(jnp.float32))))
            for o, r in zip(orig_leaves[:5], rest_leaves[:5])
        )
        restored_dtype = first_dtype(restored.params)
        detail = f"max_diff={max_diff}, step={int(restored.step)}, dtype={restored_dtype}"
        record("Checkpoint round-trip", max_diff == 0.0, detail)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 7: Gradient accumulation
# ---------------------------------------------------------------------------
def test_gradient_accumulation(state, mesh, num_devices):
    from jax_qwenvl.train.train_step import train_step_with_accumulation
    from jax_qwenvl.types import Batch

    num_accum = 2
    dp_batch = num_devices
    # Create stacked micro-batches: shape (num_accum, B, L) for most fields
    stacked_batch = Batch(
        input_ids=np.ones((num_accum, dp_batch, SEQ_LEN), dtype=np.int32),
        labels=np.concatenate([
            np.full((num_accum, dp_batch, 1), -100, dtype=np.int32),
            np.ones((num_accum, dp_batch, SEQ_LEN - 1), dtype=np.int32),
        ], axis=2),
        attention_mask=np.ones((num_accum, dp_batch, SEQ_LEN), dtype=np.int32),
        position_ids=np.zeros((num_accum, 3, dp_batch, SEQ_LEN), dtype=np.int32),
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    )

    with mesh:
        # Stacked batches have an extra leading num_accum dim.
        # Shard on the batch dim (axis 1 for most, axis 2 for position_ids).
        from jax.sharding import NamedSharding, PartitionSpec as P
        stacked_dp = NamedSharding(mesh, P(None, 'dp'))
        stacked_pos = NamedSharding(mesh, P(None, None, 'dp', None))
        replicated = NamedSharding(mesh, P())

        def _shard_stacked(name, x):
            if x is None:
                return None
            if name == 'position_ids':
                return jax.device_put(x, stacked_pos)
            if x.ndim >= 2:
                return jax.device_put(x, stacked_dp)
            return jax.device_put(x, replicated)

        sharded_batch = Batch(*[
            _shard_stacked(n, v) for n, v in zip(stacked_batch._fields, stacked_batch)
        ])

        t0 = time.time()
        new_state, metrics = train_step_with_accumulation(
            state, sharded_batch, num_accum
        )
        loss_val = float(metrics["loss"])
        jax.block_until_ready(new_state.params)
        elapsed = time.time() - t0

    is_finite = np.isfinite(loss_val)
    detail = f"loss={loss_val:.4f}, accum_steps={num_accum}, elapsed={elapsed:.2f}s"
    record("Gradient accumulation", is_finite, detail)


# ---------------------------------------------------------------------------
# Test 8: FSDP mode
# ---------------------------------------------------------------------------
def test_fsdp(model, params, num_devices):
    from jax_qwenvl.train.sharding import (
        create_device_mesh,
        get_param_sharding_rules,
        shard_params,
        shard_batch,
    )
    from jax_qwenvl.train.optimizer import create_optimizer
    from jax_qwenvl.train.train_state import create_train_state
    from jax_qwenvl.train.train_step import train_step
    from jax_qwenvl.types import Batch

    mesh = create_device_mesh(dp=1, fsdp=num_devices)
    rules = get_param_sharding_rules("fsdp")

    with mesh:
        sharded_params = shard_params(params, mesh, rules)

    # Check FSDP sharding
    first_kernel = None
    for leaf in jax.tree_util.tree_leaves(sharded_params):
        if leaf.ndim == 2:
            first_kernel = leaf
            break

    kernel_sharding = str(first_kernel.sharding) if first_kernel is not None else "none"

    optimizer, _ = create_optimizer(
        params=sharded_params,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=10,
        total_steps=100,
        tune_llm=True,
    )
    state = create_train_state(model, sharded_params, optimizer)

    dummy_batch = Batch(
        input_ids=np.ones((BATCH_SIZE, SEQ_LEN), dtype=np.int32),
        labels=np.concatenate([
            np.full((BATCH_SIZE, 1), -100, dtype=np.int32),
            np.ones((BATCH_SIZE, SEQ_LEN - 1), dtype=np.int32),
        ], axis=1),
        attention_mask=np.ones((BATCH_SIZE, SEQ_LEN), dtype=np.int32),
        position_ids=np.zeros((3, BATCH_SIZE, SEQ_LEN), dtype=np.int32),
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    )

    with mesh:
        sharded_batch = shard_batch(dummy_batch, mesh)

        t0 = time.time()
        new_state, metrics = train_step(state, sharded_batch)
        loss_val = float(metrics["loss"])
        jax.block_until_ready(new_state.params)
        elapsed = time.time() - t0

    is_finite = np.isfinite(loss_val)
    detail = f"loss={loss_val:.4f}, kernel_sharding={kernel_sharding}, elapsed={elapsed:.2f}s"
    record("FSDP mode training (bf16)", is_finite, detail)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("JAX Qwen3-VL TPU Validation (bfloat16)")
    logger.info("=" * 60)

    # Test 1: TPU devices
    try:
        num_devices = test_tpu_devices()
    except Exception:
        record("TPU device detection", False, traceback.format_exc())
        num_devices = 1

    # Download model from HuggingFace Hub
    try:
        download_model()
    except Exception:
        logger.error("Failed to download model: %s", traceback.format_exc())
        record("Model weight loading (bf16)", False, "Failed to download model")
        _print_summary()
        return

    # Test 2: Weight loading (with bf16 cast)
    config, params = None, None
    try:
        config, params = test_weight_loading()
    except Exception:
        record("Model weight loading (bf16)", False, traceback.format_exc())

    if config is None or params is None:
        logger.error("Cannot proceed without model weights. Aborting.")
        _print_summary()
        return

    # Test 3: Forward inference (bf16)
    model = None
    try:
        model = test_forward(config, params)
    except Exception:
        record("Forward inference (bf16)", False, traceback.format_exc())

    if model is None:
        from jax_qwenvl.model import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration(config=config)

    # Test 4: SPMD sharding
    try:
        test_spmd_sharding(params, num_devices)
    except Exception:
        record("SPMD sharding (DP)", False, traceback.format_exc())

    # Test 5: Single training step (bf16)
    state, mesh = None, None
    try:
        state, mesh = test_train_step(model, params, num_devices)
    except Exception:
        record("Single training step (bf16)", False, traceback.format_exc())

    # Test 6: Checkpoint
    if state is not None and mesh is not None:
        try:
            test_checkpoint(state, mesh)
        except Exception:
            record("Checkpoint round-trip", False, traceback.format_exc())
    else:
        record("Checkpoint round-trip", False, "Skipped: no state from training step")

    # Test 7: Gradient accumulation (reuse state/mesh from test 5 to save memory)
    if state is not None and mesh is not None:
        try:
            test_gradient_accumulation(state, mesh, num_devices)
        except Exception:
            record("Gradient accumulation", False, traceback.format_exc())
    else:
        record("Gradient accumulation", False, "Skipped: no state from training step")

    # Test 8: FSDP mode
    if num_devices >= 2:
        try:
            test_fsdp(model, params, num_devices)
        except Exception:
            record("FSDP mode training (bf16)", False, traceback.format_exc())
    else:
        record("FSDP mode training (bf16)", False, "Skipped: need >= 2 devices for FSDP")

    _print_summary()


def _print_summary():
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    for name, p, detail in results:
        status = "PASS" if p else "FAIL"
        logger.info("  [%s] %s", status, name)
    logger.info("")
    logger.info("Result: %d/%d tests passed", passed, total)
    if passed < total:
        logger.info("SOME TESTS FAILED")
        sys.exit(1)
    else:
        logger.info("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
