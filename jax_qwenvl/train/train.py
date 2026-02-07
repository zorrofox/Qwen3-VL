"""Training entry point for JAX Qwen3-VL fine-tuning.

Usage:
    python -m jax_qwenvl.train.train --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \\
        --dataset_use cambrian_737k --output_dir ./output --tune_mm_llm True
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field, fields
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoProcessor

from jax_qwenvl.model import Qwen3VLConfig, Qwen3VLForConditionalGeneration, load_hf_weights
from jax_qwenvl.data import data_list
from jax_qwenvl.data.data_processor import LazySupervisedDataset, DataCollatorForSupervisedDataset
from jax_qwenvl.train.optimizer import create_optimizer
from jax_qwenvl.train.train_state import create_train_state, TrainState
from jax_qwenvl.train.train_step import train_step, train_step_with_accumulation
from jax_qwenvl.train.sharding import create_device_mesh, get_param_sharding_rules, shard_params, shard_batch
from jax_qwenvl.train.checkpoint import CheckpointManager
from jax_qwenvl.train.metrics_logger import MetricsLogger
from jax_qwenvl.train.optimizer import create_schedule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    tune_mm_llm: bool = False
    tune_mm_mlp: bool = False
    tune_mm_vision: bool = False


@dataclass
class DataArguments:
    dataset_use: str = ""
    max_pixels: int = 28 * 28 * 576
    min_pixels: int = 28 * 28 * 16
    video_max_frames: int = 8
    video_min_frames: int = 4
    video_max_pixels: int = 1024 * 28 * 28
    video_min_pixels: int = 256 * 28 * 28
    video_fps: float = 2.0
    data_flatten: bool = False
    data_packing: bool = False
    model_type: str = "qwen3vl"


@dataclass
class TrainingArguments:
    output_dir: str = "./output"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    model_max_length: int = 8192
    lora_enable: bool = False
    lora_rank: int = 64
    lora_alpha: int = 128
    max_grad_norm: float = 1.0
    vision_lr: Optional[float] = None
    projector_lr: Optional[float] = None
    logging_steps: int = 10
    save_steps: int = 500
    bf16: bool = True
    seed: int = 42
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    fsdp: bool = False
    max_checkpoints: int = 3
    resume_from_checkpoint: Optional[str] = None
    report_to: str = "none"
    run_name: str = ""
    warmup_ratio: float = 0.0


# ---------------------------------------------------------------------------
# Param merging for LoRA
# ---------------------------------------------------------------------------

def _merge_params(loaded_params: dict, init_params: dict) -> dict:
    """Merge loaded HF weights with initialized LoRA params.

    Uses ``loaded_params`` as base, but fills in any keys present in
    ``init_params`` but missing in ``loaded_params`` (i.e., the LoRA A/B
    matrices).

    Args:
        loaded_params: parameter dict loaded from HuggingFace safetensors.
        init_params: parameter dict from ``model.init()``, containing LoRA
            params initialised with Kaiming / zeros.

    Returns:
        Merged parameter dict.
    """
    def merge(loaded, init):
        if isinstance(loaded, dict) and isinstance(init, dict):
            result = dict(loaded)
            for k, v in init.items():
                if k not in result:
                    result[k] = v
                else:
                    result[k] = merge(result[k], v)
            return result
        return loaded

    return merge(loaded_params, init_params)


# ---------------------------------------------------------------------------
# Argument parser helper
# ---------------------------------------------------------------------------

def _add_dataclass_args(parser: argparse.ArgumentParser, dc_class) -> None:
    """Add all fields of a dataclass as argparse arguments."""
    for f in fields(dc_class):
        arg_name = f"--{f.name}"
        if f.type is bool or f.type == "bool":
            parser.add_argument(
                arg_name,
                type=lambda v: v.lower() in ("true", "1", "yes"),
                default=f.default,
                help=f"{f.name} (bool)",
            )
        elif f.type is Optional[float] or str(f.type) == "typing.Optional[float]":
            parser.add_argument(
                arg_name,
                type=lambda v: None if v.lower() == "none" else float(v),
                default=f.default,
                help=f"{f.name} (optional float)",
            )
        elif f.type is Optional[str] or str(f.type) == "typing.Optional[str]":
            parser.add_argument(
                arg_name,
                type=lambda v: None if v.lower() == "none" else str(v),
                default=f.default,
                help=f"{f.name} (optional str)",
            )
        elif f.type is float or f.type == "float":
            parser.add_argument(arg_name, type=float, default=f.default)
        elif f.type is int or f.type == "int":
            parser.add_argument(arg_name, type=int, default=f.default)
        else:
            parser.add_argument(arg_name, type=str, default=f.default)


def _make_args(parser: argparse.ArgumentParser):
    """Parse args and build typed dataclass instances."""
    args = parser.parse_args()
    args_dict = vars(args)

    def _build(dc_cls):
        kw = {}
        for f in fields(dc_cls):
            if f.name in args_dict:
                kw[f.name] = args_dict[f.name]
        return dc_cls(**kw)

    return _build(ModelArguments), _build(DataArguments), _build(TrainingArguments)


# ---------------------------------------------------------------------------
# Simple batch iterator
# ---------------------------------------------------------------------------

def _batch_indices(dataset_size: int, batch_size: int, seed: int, epoch: int):
    """Yield lists of indices for each batch in one epoch."""
    rng = np.random.RandomState(seed + epoch)
    indices = rng.permutation(dataset_size)
    for start in range(0, dataset_size, batch_size):
        end = min(start + batch_size, dataset_size)
        yield indices[start:end].tolist()


# ---------------------------------------------------------------------------
# Micro-batch stacking helper
# ---------------------------------------------------------------------------

def _stack_micro_batches(micro_batches):
    """Stack a list of Batch NamedTuples into a single Batch with leading accum dim.

    Each field of shape (B, ...) becomes (num_accum, B, ...).
    Fields that are None remain None.
    Special case: position_ids (3, B, L) -> (num_accum, 3, B, L).
    """
    from jax_qwenvl.types import Batch

    def _stack_field(name, values):
        # Filter None values -- if any is None, return None
        if values[0] is None:
            return None
        return np.stack(values, axis=0)

    field_names = micro_batches[0]._fields
    stacked_values = []
    for name in field_names:
        field_vals = [getattr(mb, name) for mb in micro_batches]
        stacked_values.append(_stack_field(name, field_vals))

    return Batch(*stacked_values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="JAX Qwen3-VL fine-tuning")
    _add_dataclass_args(parser, ModelArguments)
    _add_dataclass_args(parser, DataArguments)
    _add_dataclass_args(parser, TrainingArguments)
    model_args, data_args, training_args = _make_args(parser)

    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info("JAX devices: %s", jax.devices())
    logger.info("Model: %s", model_args.model_name_or_path)

    # 2. Load config
    config = Qwen3VLConfig.from_pretrained(model_args.model_name_or_path)

    # 3. Load processor/tokenizer
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    # 4. Create device mesh
    if training_args.fsdp:
        mesh = create_device_mesh(dp=1, fsdp=-1)
        sharding_mode = 'fsdp'
    else:
        mesh = create_device_mesh(dp=-1, fsdp=1)
        sharding_mode = 'dp'
    logger.info("Device mesh: %s (mode=%s)", mesh, sharding_mode)

    # 5. Initialize model
    lora_rank = training_args.lora_rank if training_args.lora_enable else 0
    lora_alpha = float(training_args.lora_alpha) if training_args.lora_enable else 1.0
    model = Qwen3VLForConditionalGeneration(
        config=config,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        gradient_checkpointing=training_args.gradient_checkpointing,
    )

    # 6. Load weights from HuggingFace safetensors
    logger.info("Loading weights from %s ...", model_args.model_name_or_path)
    loaded = load_hf_weights(
        model_args.model_name_or_path, config, lora_rank=lora_rank
    )
    params = loaded["params"]

    # 7. Merge with initialised LoRA params if needed
    if lora_rank > 0:
        logger.info("LoRA enabled (rank=%d, alpha=%.1f). Initialising LoRA params ...",
                     lora_rank, lora_alpha)
        rng = jax.random.PRNGKey(training_args.seed)
        dummy_input_ids = jnp.ones((1, 4), dtype=jnp.int32)
        dummy_pos_ids = jnp.zeros((3, 1, 4), dtype=jnp.int32)
        init_variables = model.init(rng, dummy_input_ids, dummy_pos_ids)
        init_params = init_variables["params"]
        params = _merge_params(params, init_params)

    # 8. Shard params onto device mesh
    with mesh:
        rules = get_param_sharding_rules(sharding_mode)
        params = shard_params(params, mesh, rules)
    logger.info("Parameters sharded with mode=%s", sharding_mode)

    # 9. Create dataset and compute total steps
    dataset_configs = data_list(data_args.dataset_use.split(","))
    logger.info("Datasets: %s", dataset_configs)

    dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        from jax_qwenvl.data.data_processor import FlattenedDataCollatorForSupervisedDataset
        collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)
        logger.info("Using FlattenedDataCollator (data_flatten=%s, data_packing=%s)",
                     data_args.data_flatten, data_args.data_packing)
    else:
        collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)

    dataset_size = len(dataset)
    batch_size = training_args.per_device_train_batch_size
    steps_per_epoch = max(dataset_size // batch_size, 1)
    total_steps = steps_per_epoch * training_args.num_train_epochs
    logger.info(
        "Dataset size=%d, batch_size=%d, steps_per_epoch=%d, total_steps=%d",
        dataset_size, batch_size, steps_per_epoch, total_steps,
    )

    # Compute warmup steps from ratio if needed
    warmup_steps = training_args.warmup_steps
    if training_args.warmup_ratio > 0 and training_args.warmup_steps == 0:
        warmup_steps = int(total_steps * training_args.warmup_ratio)
        logger.info("Using warmup_ratio=%.3f -> warmup_steps=%d", training_args.warmup_ratio, warmup_steps)

    # Create metrics logger
    metrics_logger = MetricsLogger(
        output_dir=training_args.output_dir,
        report_to=training_args.report_to,
        run_name=training_args.run_name,
        config={
            "learning_rate": training_args.learning_rate,
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "model_name_or_path": model_args.model_name_or_path,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        },
    )

    # 10. Create optimizer
    optimizer, label_tree = create_optimizer(
        params=params,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_grad_norm=training_args.max_grad_norm,
        vision_lr=training_args.vision_lr,
        projector_lr=training_args.projector_lr,
        tune_vision=model_args.tune_mm_vision,
        tune_mlp=model_args.tune_mm_mlp,
        tune_llm=model_args.tune_mm_llm,
        lora_enabled=training_args.lora_enable,
    )

    # 11. Create train state
    state = create_train_state(model, params, optimizer)
    logger.info("Train state created.")

    # 12. Checkpoint manager
    ckpt_manager = CheckpointManager(
        output_dir=training_args.output_dir,
        max_to_keep=training_args.max_checkpoints,
        save_interval_steps=training_args.save_steps,
    )

    # Resume from checkpoint if requested
    if training_args.resume_from_checkpoint:
        restored = ckpt_manager.restore(state_template=state)
        if restored is not None:
            state = restored
            logger.info("Resumed from step %s", ckpt_manager.latest_step())

    logger.info("Starting training ...")

    # 13. Training loop
    global_step = 0
    accum_steps = training_args.gradient_accumulation_steps

    with mesh:
        for epoch in range(training_args.num_train_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            t0 = time.time()

            batch_iter = _batch_indices(
                dataset_size, batch_size, training_args.seed, epoch
            )

            if accum_steps > 1:
                # Gradient accumulation: collect accum_steps micro-batches
                micro_batch_buffer = []
                for batch_idx_list in batch_iter:
                    samples = [dataset[i] for i in batch_idx_list]
                    micro_batch = collator(samples)
                    micro_batch_buffer.append(micro_batch)

                    if len(micro_batch_buffer) == accum_steps:
                        # Stack micro-batches: each field gets leading dim of accum_steps
                        stacked = _stack_micro_batches(micro_batch_buffer)
                        stacked = shard_batch(stacked, mesh)
                        state, metrics = train_step_with_accumulation(
                            state, stacked, accum_steps
                        )
                        micro_batch_buffer = []

                        loss_val = float(metrics["loss"])
                        epoch_loss += loss_val
                        epoch_steps += 1
                        global_step += 1

                        if global_step % training_args.logging_steps == 0:
                            avg_loss = epoch_loss / max(epoch_steps, 1)
                            elapsed = time.time() - t0
                            logger.info(
                                "Epoch %d | Step %d (global %d) | loss=%.4f | avg_loss=%.4f | %.1fs",
                                epoch, epoch_steps, global_step, loss_val, avg_loss, elapsed,
                            )
                            metrics_logger.log({
                                "train/loss": loss_val,
                                "train/avg_loss": avg_loss,
                                "train/epoch": epoch,
                                "train/global_step": global_step,
                                "train/learning_rate": float(
                                    create_schedule(
                                        training_args.learning_rate, warmup_steps, total_steps
                                    )(global_step)
                                ),
                            }, step=global_step)

                        if ckpt_manager.should_save(global_step):
                            ckpt_manager.save(global_step, state)
            else:
                for batch_idx_list in batch_iter:
                    samples = [dataset[i] for i in batch_idx_list]
                    batch = collator(samples)
                    batch = shard_batch(batch, mesh)

                    state, metrics = train_step(state, batch)
                    loss_val = float(metrics["loss"])
                    epoch_loss += loss_val
                    epoch_steps += 1
                    global_step += 1

                    if global_step % training_args.logging_steps == 0:
                        avg_loss = epoch_loss / max(epoch_steps, 1)
                        elapsed = time.time() - t0
                        logger.info(
                            "Epoch %d | Step %d (global %d) | loss=%.4f | avg_loss=%.4f | %.1fs",
                            epoch, epoch_steps, global_step, loss_val, avg_loss, elapsed,
                        )
                        metrics_logger.log({
                            "train/loss": loss_val,
                            "train/avg_loss": avg_loss,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                            "train/learning_rate": float(
                                create_schedule(
                                    training_args.learning_rate, warmup_steps, total_steps
                                )(global_step)
                            ),
                        }, step=global_step)

                    if ckpt_manager.should_save(global_step):
                        ckpt_manager.save(global_step, state)

            avg_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(
                "Epoch %d complete | avg_loss=%.4f | steps=%d",
                epoch, avg_loss, epoch_steps,
            )

        # Final checkpoint save
        ckpt_manager.save(global_step, state, force=True)

        # Export weights to HuggingFace format
        logger.info("Exporting weights to HuggingFace safetensors format ...")
        from jax_qwenvl.model.weight_exporter import export_hf_weights
        export_hf_weights(
            params=state.params,
            output_dir=training_args.output_dir,
            config=config,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # Save processor/tokenizer
        processor.save_pretrained(training_args.output_dir)
        logger.info("Processor saved to %s", training_args.output_dir)

    metrics_logger.finish()
    logger.info("Training complete. Output dir: %s", training_args.output_dir)


if __name__ == "__main__":
    main()
