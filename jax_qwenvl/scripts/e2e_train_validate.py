"""End-to-end training validation with real image data.

Runs 5 training steps using the demo_images dataset (real .png images)
on TPU to verify the full pipeline: data loading -> image processing ->
collation -> vision encoder -> LLM -> loss -> gradient update.

Tests:
  1. Dataset loads real images -> pixel_values is non-None with valid shape
  2. Batch collation produces pixel_values and image_grid_thw
  3. All training steps produce finite loss (no NaN/Inf)
  4. Loss does not explode (final loss < 100)
  5. Full 5-step E2E training completes without error

Usage:
    python3 -m jax_qwenvl.scripts.e2e_train_validate
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_HF_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_PATH = None
NUM_TRAIN_STEPS = 5
MAX_PIXELS = 28 * 28 * 64  # Small images to save memory
MIN_PIXELS = 28 * 28 * 4
MODEL_MAX_LENGTH = 2048
PARAM_DTYPE = jnp.bfloat16

results = []


def record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    logger.info("[%s] %s %s", status, name, f"-- {detail}" if detail else "")


def download_model():
    global MODEL_PATH
    from huggingface_hub import snapshot_download

    logger.info("Downloading model %s ...", MODEL_HF_ID)
    MODEL_PATH = snapshot_download(
        MODEL_HF_ID,
        ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.onnx"],
    )
    logger.info("Model downloaded to %s", MODEL_PATH)
    return MODEL_PATH


def _cast_to_bf16(params):
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        params,
    )


# ---------------------------------------------------------------------------
# Test 1: Dataset loads real images
# ---------------------------------------------------------------------------
def test_dataset_loading():
    from transformers import AutoProcessor
    from jax_qwenvl.data.data_processor import LazySupervisedDataset

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    @dataclass
    class DataArgs:
        dataset_use: str = "demo_images"
        max_pixels: int = MAX_PIXELS
        min_pixels: int = MIN_PIXELS
        video_max_frames: int = 8
        video_min_frames: int = 4
        video_max_pixels: int = 1024 * 28 * 28
        video_min_pixels: int = 256 * 28 * 28
        video_fps: float = 2.0
        data_flatten: bool = False
        data_packing: bool = False
        model_type: str = "qwen3vl"
        model_max_length: int = MODEL_MAX_LENGTH

    data_args = DataArgs()
    processor.tokenizer.model_max_length = MODEL_MAX_LENGTH

    dataset = LazySupervisedDataset(processor, data_args=data_args)
    sample = dataset[0]

    has_pixels = "pixel_values" in sample and sample["pixel_values"] is not None
    pixel_shape = sample["pixel_values"].shape if has_pixels else None
    has_grid = "image_grid_thw" in sample and sample["image_grid_thw"] is not None

    detail = (
        f"dataset_size={len(dataset)}, has_pixel_values={has_pixels}, "
        f"pixel_shape={pixel_shape}, has_grid_thw={has_grid}"
    )
    record(
        "Dataset loads real images",
        has_pixels and pixel_shape is not None and len(pixel_shape) >= 2 and has_grid,
        detail,
    )
    return dataset, processor, data_args


# ---------------------------------------------------------------------------
# Test 2: Batch collation
# ---------------------------------------------------------------------------
def test_batch_collation(dataset, processor):
    from jax_qwenvl.data.data_processor import DataCollatorForSupervisedDataset

    merge_size = getattr(processor.image_processor, "merge_size", 2)
    collator = DataCollatorForSupervisedDataset(
        tokenizer=processor.tokenizer,
        spatial_merge_size=merge_size,
    )

    num_devices = len(jax.devices())
    # Collect samples -- repeat if dataset is smaller than num_devices
    samples = []
    for i in range(num_devices):
        samples.append(dataset[i % len(dataset)])

    batch = collator(samples)

    has_pixels = batch.pixel_values is not None
    has_grid = batch.image_grid_thw is not None
    pv_shape = batch.pixel_values.shape if has_pixels else None
    gt_shape = batch.image_grid_thw.shape if has_grid else None
    ids_shape = batch.input_ids.shape
    pos_shape = batch.position_ids.shape

    detail = (
        f"input_ids={ids_shape}, position_ids={pos_shape}, "
        f"pixel_values={pv_shape}, image_grid_thw={gt_shape}"
    )
    record(
        "Batch collation with images",
        has_pixels and has_grid and pv_shape is not None and len(pv_shape) >= 2,
        detail,
    )
    return collator


# ---------------------------------------------------------------------------
# Tests 3-5: E2E training steps
# ---------------------------------------------------------------------------
def test_e2e_training(dataset, processor, data_args, collator):
    from jax_qwenvl.model import Qwen3VLConfig, Qwen3VLForConditionalGeneration, load_hf_weights
    from jax_qwenvl.train.sharding import (
        create_device_mesh, get_param_sharding_rules, shard_params, shard_batch,
    )
    from jax_qwenvl.train.optimizer import create_optimizer
    from jax_qwenvl.train.train_state import create_train_state
    from jax_qwenvl.train.train_step import train_step

    num_devices = len(jax.devices())

    # Load model
    config = Qwen3VLConfig.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration(config=config)

    logger.info("Loading weights ...")
    loaded = load_hf_weights(MODEL_PATH, config, lora_rank=0)
    params = _cast_to_bf16(loaded["params"])

    # Shard params
    mesh = create_device_mesh(dp=num_devices, fsdp=1)
    rules = get_param_sharding_rules("dp")
    with mesh:
        params = shard_params(params, mesh, rules)

    # Optimizer + state
    optimizer, _ = create_optimizer(
        params=params,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=2,
        total_steps=NUM_TRAIN_STEPS,
        tune_llm=True,
    )
    state = create_train_state(model, params, optimizer)

    # Training loop
    losses = []
    all_finite = True

    with mesh:
        for step_idx in range(NUM_TRAIN_STEPS):
            # Collect batch
            samples = []
            for i in range(num_devices):
                idx = (step_idx * num_devices + i) % len(dataset)
                samples.append(dataset[idx])

            batch = collator(samples)

            # Verify pixel_values present
            if batch.pixel_values is None:
                logger.warning("Step %d: pixel_values is None!", step_idx)

            batch = shard_batch(batch, mesh)

            step_t0 = time.time()
            state, metrics = train_step(state, batch)
            jax.block_until_ready(metrics["loss"])
            step_elapsed = time.time() - step_t0

            loss_val = float(metrics["loss"])
            losses.append(loss_val)
            is_finite = np.isfinite(loss_val)
            if not is_finite:
                all_finite = False

            logger.info(
                "  Step %d/%d | loss=%.4f | finite=%s | time=%.2fs",
                step_idx + 1, NUM_TRAIN_STEPS, loss_val, is_finite, step_elapsed,
            )

    # Test 3: All losses finite
    detail_3 = f"losses={[f'{l:.4f}' for l in losses]}"
    record("All training steps finite", all_finite, detail_3)

    # Test 4: Loss not exploded
    final_loss = losses[-1] if losses else float("inf")
    detail_4 = f"final_loss={final_loss:.4f}"
    record("Loss not exploded (<100)", final_loss < 100, detail_4)

    # Test 5: E2E completed
    completed = len(losses) == NUM_TRAIN_STEPS
    detail_5 = f"completed_steps={len(losses)}/{NUM_TRAIN_STEPS}"
    record("E2E vision training complete", completed, detail_5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("JAX Qwen3-VL E2E Training Validation (real images, bfloat16)")
    logger.info("=" * 60)
    logger.info("JAX devices: %s (backend=%s)", jax.devices(), jax.default_backend())

    # Download model
    try:
        download_model()
    except Exception:
        logger.error("Failed to download model:\n%s", traceback.format_exc())
        record("Model download", False, "Failed to download")
        _print_summary()
        return

    # Test 1: Dataset loading
    dataset, processor, data_args = None, None, None
    try:
        dataset, processor, data_args = test_dataset_loading()
    except Exception:
        record("Dataset loads real images", False, traceback.format_exc())

    if dataset is None:
        logger.error("Cannot proceed without dataset.")
        _print_summary()
        return

    # Test 2: Batch collation
    collator = None
    try:
        collator = test_batch_collation(dataset, processor)
    except Exception:
        record("Batch collation with images", False, traceback.format_exc())

    if collator is None:
        logger.error("Cannot proceed without collator.")
        _print_summary()
        return

    # Tests 3-5: E2E training
    try:
        test_e2e_training(dataset, processor, data_args, collator)
    except Exception:
        record("E2E vision training", False, traceback.format_exc())

    _print_summary()


def _print_summary():
    logger.info("")
    logger.info("=" * 60)
    logger.info("E2E VALIDATION SUMMARY")
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
