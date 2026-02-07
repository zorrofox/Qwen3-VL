"""Top-level Qwen3-VL model for conditional generation in Flax.

Combines: VisionModel + TextModel + lm_head.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .config import Qwen3VLConfig
from .vit import VisionModel
from .llm import TextModel


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    ignore_index: int = -100,
) -> jnp.ndarray:
    """Cross-entropy loss with label masking.

    Args:
        logits: ``(batch, seq_len, vocab_size)``
        labels: ``(batch, seq_len)`` int32, with ``ignore_index`` for masked positions.

    Returns:
        Scalar loss.
    """
    valid_mask = (labels != ignore_index).astype(jnp.float32)
    safe_labels = jnp.where(labels != ignore_index, labels, 0)
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    nll = -jnp.take_along_axis(
        log_probs, safe_labels[..., None], axis=-1
    ).squeeze(-1)
    return (nll * valid_mask).sum() / jnp.maximum(valid_mask.sum(), 1.0)


# ---------------------------------------------------------------------------
# Embedding scatter
# ---------------------------------------------------------------------------

def _scatter_embeddings(
    hidden_states: jnp.ndarray,
    mask: jnp.ndarray,
    embeddings: jnp.ndarray,
) -> jnp.ndarray:
    """Replace placeholder token embeddings with vision embeddings.

    Args:
        hidden_states: ``(batch, seq_len, hidden_size)``
        mask: ``(batch, seq_len)`` bool -- True at positions to replace.
        embeddings: ``(total_vision_tokens, hidden_size)``

    Returns:
        Updated hidden_states.
    """
    B, L, D = hidden_states.shape
    mask_flat = mask.reshape(-1)  # (B*L,)
    hs_flat = hidden_states.reshape(-1, D)  # (B*L, D)

    # Compute indices into embeddings via cumulative sum of mask
    cumsum = jnp.cumsum(mask_flat.astype(jnp.int32)) - 1
    safe_idx = jnp.clip(cumsum, 0, embeddings.shape[0] - 1)
    gathered = embeddings[safe_idx]  # (B*L, D)

    # Replace where mask is True
    hs_flat = jnp.where(mask_flat[:, None], gathered, hs_flat)
    return hs_flat.reshape(B, L, D)


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

def _make_causal_mask(
    seq_len: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jnp.ndarray:
    """Create a causal attention mask.

    Returns:
        ``(1, 1, seq_len, seq_len)`` with 0 for attend and large negative
        for masked positions.
    """
    mask = jnp.triu(
        jnp.full((seq_len, seq_len), jnp.finfo(dtype).min, dtype=dtype),
        k=1,
    )
    return mask[None, None, :, :]


def _make_packed_causal_mask(
    cu_seqlens: jnp.ndarray,
    seq_len: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jnp.ndarray:
    """Build causal + block-diagonal mask from cumulative sequence lengths.

    Each packed subsequence has its own causal mask.
    Cross-subsequence attention is blocked.

    Args:
        cu_seqlens: 1-D int32 array of cumulative sequence lengths
            (e.g. ``[0, 3, 5]``).
        seq_len: total sequence length.
        dtype: output dtype.

    Returns:
        ``(1, 1, seq_len, seq_len)`` attention mask.
    """
    pos = jnp.arange(seq_len)
    # Segment assignment: for each position, count how many cu_seqlens it is >=
    belongs = (pos[:, None] >= cu_seqlens[None, :])
    segment_ids = jnp.sum(belongs.astype(jnp.int32), axis=-1) - 1
    # Same-segment + causal
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    causal = pos[:, None] >= pos[None, :]
    mask = same_segment & causal
    attn_mask = jnp.where(mask, 0.0, jnp.finfo(dtype).min)
    return attn_mask[None, None, :, :].astype(dtype)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Qwen3VLForConditionalGeneration(nn.Module):
    """Qwen3-VL: Vision-Language model for conditional generation.

    Architecture:
        1. ``embed_tokens(input_ids)``
        2. If ``pixel_values``: run VisionModel, scatter image embeddings
        3. If ``pixel_values_videos``: run VisionModel, scatter video embeddings
        4. TextModel with DeepStack features
        5. ``lm_head`` (or tied embeddings)
        6. If ``labels``: compute cross-entropy loss
    """

    config: Qwen3VLConfig
    lora_rank: int = 0
    lora_alpha: float = 1.0
    gradient_checkpointing: bool = False

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        position_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        pixel_values: Optional[jnp.ndarray] = None,
        image_grid_thw: Optional[jnp.ndarray] = None,
        pixel_values_videos: Optional[jnp.ndarray] = None,
        video_grid_thw: Optional[jnp.ndarray] = None,
        labels: Optional[jnp.ndarray] = None,
    ):
        """Forward pass.

        Args:
            input_ids: ``(batch, seq_len)`` int32
            position_ids: ``(3, batch, seq_len)`` int32 (from MRoPE computation)
            attention_mask: ``(batch, seq_len)`` int32/bool or
                ``(batch, 1, seq_len, seq_len)`` float causal mask
            pixel_values: ``(N, C*T, H, W)`` float -- image patches
            image_grid_thw: ``(num_images, 3)`` int32
            pixel_values_videos: ``(N, C*T, H, W)`` float -- video patches
            video_grid_thw: ``(num_videos, 3)`` int32
            labels: ``(batch, seq_len)`` int32 with -100 for ignored positions

        Returns:
            ``(logits, loss)`` where loss is None if labels not provided.
        """
        cfg = self.config
        tcfg = cfg.text_config
        B, L = input_ids.shape

        # Token embedding
        embed_table = self.param(
            "embed_tokens",
            nn.initializers.normal(stddev=tcfg.initializer_range),
            (tcfg.vocab_size, tcfg.hidden_size),
        )
        inputs_embeds = embed_table[input_ids]  # (B, L, D)

        # Process images through vision encoder
        image_mask = None
        video_mask = None
        all_deepstack_features: List[jnp.ndarray] = []
        visual_pos_masks = None

        if pixel_values is not None and image_grid_thw is not None:
            # Cast pixel_values to match param dtype (e.g. bfloat16)
            pixel_values = pixel_values.astype(inputs_embeds.dtype)
            image_embeds, ds_feats = VisionModel(
                config=cfg.vision_config,
                gradient_checkpointing=self.gradient_checkpointing,
                name="visual",
            )(pixel_values, image_grid_thw)
            # image_embeds: (num_merged_tokens, out_hidden_size)

            # Mask: where input_ids == image_token_id
            image_mask = (input_ids == cfg.image_token_id)  # (B, L) bool
            inputs_embeds = _scatter_embeddings(
                inputs_embeds, image_mask, image_embeds
            )
            all_deepstack_features = ds_feats

        if pixel_values_videos is not None and video_grid_thw is not None:
            # Cast video pixel_values to match param dtype
            pixel_values_videos = pixel_values_videos.astype(inputs_embeds.dtype)
            video_embeds, ds_feats_v = VisionModel(
                config=cfg.vision_config,
                gradient_checkpointing=self.gradient_checkpointing,
                name="visual",
            )(pixel_values_videos, video_grid_thw)

            video_mask = (input_ids == cfg.video_token_id)  # (B, L) bool
            inputs_embeds = _scatter_embeddings(
                inputs_embeds, video_mask, video_embeds
            )

            # Merge deepstack features from video with image features
            if len(all_deepstack_features) == 0:
                all_deepstack_features = ds_feats_v
            else:
                # Concatenate matching layers
                all_deepstack_features = [
                    jnp.concatenate([img_f, vid_f], axis=0)
                    for img_f, vid_f in zip(all_deepstack_features, ds_feats_v)
                ]

        # Compute visual_pos_masks: OR of image and video masks
        if image_mask is not None or video_mask is not None:
            if image_mask is not None and video_mask is not None:
                visual_pos_masks = image_mask | video_mask
            elif image_mask is not None:
                visual_pos_masks = image_mask
            else:
                visual_pos_masks = video_mask

        # Build causal attention mask if needed
        if attention_mask is not None and attention_mask.ndim == 2:
            # Convert (B, L) attention mask to (B, 1, L, L) causal mask
            causal = _make_causal_mask(L, dtype=inputs_embeds.dtype)
            # Combine with padding mask: positions where attention_mask == 0 should
            # have large negative values.
            pad_mask = (
                attention_mask[:, None, None, :]
                .astype(inputs_embeds.dtype)
            )
            # 0 in pad_mask -> should be masked -> large negative
            pad_mask = (1.0 - pad_mask) * jnp.finfo(inputs_embeds.dtype).min
            attn_mask = causal + pad_mask
        elif attention_mask is not None and attention_mask.ndim == 4:
            attn_mask = attention_mask
        elif attention_mask is not None and attention_mask.ndim == 1:
            # Packed sequences: attention_mask is cu_seqlens (1D)
            attn_mask = _make_packed_causal_mask(
                attention_mask, L, dtype=inputs_embeds.dtype
            )
        else:
            attn_mask = _make_causal_mask(L, dtype=inputs_embeds.dtype)

        # Text decoder
        deepstack_embeds = (
            all_deepstack_features if len(all_deepstack_features) > 0 else None
        )
        hidden_states = TextModel(
            config=tcfg,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            gradient_checkpointing=self.gradient_checkpointing,
            name="model",
        )(
            inputs_embeds,
            position_ids,
            attention_mask=attn_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_embeds,
        )

        # Language model head
        if cfg.tie_word_embeddings:
            logits = hidden_states @ embed_table.T
        else:
            logits = nn.Dense(
                tcfg.vocab_size, use_bias=False, name="lm_head"
            )(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = cross_entropy_loss(shift_logits, shift_labels)

        return logits, loss
