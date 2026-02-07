"""JIT-compiled training step for JAX Qwen3-VL fine-tuning.

Provides:
- ``cross_entropy_loss``: shifted cross-entropy with label masking.
- ``train_step``: a single gradient-descent step wrapped in ``jax.jit``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    ignore_index: int = -100,
) -> jnp.ndarray:
    """Cross-entropy loss with label masking.

    Args:
        logits: ``(batch, seq_len, vocab_size)`` float.
        labels: ``(batch, seq_len)`` int32.  Positions equal to
            ``ignore_index`` are excluded from the loss.

    Returns:
        Scalar loss (mean over valid tokens).
    """
    valid_mask = (labels != ignore_index).astype(jnp.float32)
    safe_labels = jnp.where(labels != ignore_index, labels, 0)
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    nll = -jnp.take_along_axis(
        log_probs, safe_labels[..., None], axis=-1
    ).squeeze(-1)
    nll = nll * valid_mask
    return nll.sum() / jnp.maximum(valid_mask.sum(), 1.0)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

@partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    """Single JIT-compiled training step.

    Computes the forward pass, loss, gradients, and applies the optimizer
    update.

    Args:
        state: ``TrainState`` containing ``params``, ``apply_fn``,
            ``opt_state``, and ``step``.
        batch: A ``Batch`` NamedTuple from the data pipeline.  Fields:
            - ``input_ids``:  ``(B, L)`` int32
            - ``labels``:     ``(B, L)`` int32
            - ``attention_mask``: ``(B, L)`` bool
            - ``position_ids``:   ``(3, B, L)`` int32
            - ``pixel_values``:  optional ``(N, C, H, W)`` float
            - ``image_grid_thw``: optional ``(num_images, 3)`` int32
            - ``pixel_values_videos``: optional ``(N, C, H, W)`` float
            - ``video_grid_thw``: optional ``(num_videos, 3)`` int32

    Returns:
        ``(new_state, metrics)`` where ``metrics`` is a dict with
        ``"loss"`` and ``"step"`` keys.
    """

    def loss_fn(params):
        # The model's __call__ returns (logits, loss)
        _logits, loss = state.apply_fn(
            {"params": params},
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            attention_mask=batch.attention_mask,
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
            pixel_values_videos=batch.pixel_values_videos,
            video_grid_thw=batch.video_grid_thw,
            labels=batch.labels,
        )
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss, "step": state.step}
    return new_state, metrics


# ---------------------------------------------------------------------------
# Training step with gradient accumulation
# ---------------------------------------------------------------------------

@partial(jax.jit, donate_argnums=(0,))
def train_step_with_accumulation(
    state,
    micro_batches,
    num_accumulation_steps: int,
):
    """Training step with gradient accumulation via ``jax.lax.scan``.

    Args:
        state: ``TrainState``.
        micro_batches: a pytree with each leaf having an extra leading
            dimension of size ``num_accumulation_steps``.  For example,
            ``input_ids`` has shape ``(num_accum, B, L)`` instead of ``(B, L)``.
        num_accumulation_steps: number of micro-batches to accumulate over.

    Returns:
        ``(new_state, metrics)`` where metrics contains ``"loss"`` and ``"step"``.
    """

    def micro_step(carry, micro_batch):
        accumulated_grads, accumulated_loss = carry

        def loss_fn(params):
            _logits, loss = state.apply_fn(
                {"params": params},
                input_ids=micro_batch.input_ids,
                position_ids=micro_batch.position_ids,
                attention_mask=micro_batch.attention_mask,
                pixel_values=micro_batch.pixel_values,
                image_grid_thw=micro_batch.image_grid_thw,
                pixel_values_videos=micro_batch.pixel_values_videos,
                video_grid_thw=micro_batch.video_grid_thw,
                labels=micro_batch.labels,
            )
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        accumulated_grads = jax.tree_util.tree_map(
            lambda a, g: a + g, accumulated_grads, grads
        )
        accumulated_loss = accumulated_loss + loss
        return (accumulated_grads, accumulated_loss), None

    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    init_carry = (zero_grads, jnp.float32(0.0))

    (total_grads, total_loss), _ = jax.lax.scan(
        micro_step, init_carry, micro_batches
    )

    avg_grads = jax.tree_util.tree_map(
        lambda g: g / num_accumulation_steps, total_grads
    )
    avg_loss = total_loss / num_accumulation_steps

    new_state = state.apply_gradients(grads=avg_grads)
    metrics = {"loss": avg_loss, "step": state.step}
    return new_state, metrics
