"""Rotary Position Embeddings for Qwen3-VL (vision + text MRoPE)."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Rotate the second half of the last dimension: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


# ---------------------------------------------------------------------------
# Vision RoPE
# ---------------------------------------------------------------------------

def vision_rotary_freq_table(
    max_pos: int,
    dim: int,
    theta: float = 10000.0,
) -> jnp.ndarray:
    """Compute a frequency look-up table for the vision rotary embedding.

    Args:
        max_pos: maximum spatial position index.
        dim: the rotary dim passed to VisionRotaryEmbedding, which is
            ``head_dim // 2``.
        theta: base frequency.

    Returns:
        ``(max_pos, dim // 2)`` frequency table.
    """
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    )  # (dim // 2,)
    seq = jnp.arange(max_pos, dtype=jnp.float32)
    freqs = jnp.outer(seq, inv_freq)  # (max_pos, dim // 2)
    return freqs


def compute_vision_rotary_cos_sin(
    pos_ids_2d: jnp.ndarray,
    head_dim: int,
    theta: float = 10000.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute cos/sin for the vision encoder from 2-D position indices.

    Implements the ``rot_pos_emb`` method from HF VisionModel, which:
    1. Builds a freq table of shape ``(max_pos, head_dim // 4)``.
    2. Looks up ``(total_tokens, 2)`` position indices to get
       ``(total_tokens, 2, head_dim // 4)``.
    3. Flattens to ``(total_tokens, head_dim // 2)``.
    4. Doubles via concat to ``(total_tokens, head_dim)``.

    Args:
        pos_ids_2d: ``(total_tokens, 2)`` int32 with (row, col) indices.
        head_dim: per-head dimension.
        theta: base frequency.

    Returns:
        ``(cos, sin)`` each of shape ``(total_tokens, head_dim)``.
    """
    rotary_dim = head_dim // 2  # dim passed to VisionRotaryEmbedding

    # Build frequency table: (max_pos, rotary_dim // 2)
    max_pos = int(pos_ids_2d.max()) + 1
    freq_table = vision_rotary_freq_table(max_pos, rotary_dim, theta)
    # freq_table: (max_pos, rotary_dim // 2) = (max_pos, head_dim // 4)

    # Look up: pos_ids_2d has shape (total_tokens, 2)
    # freq_table[pos_ids_2d] -> (total_tokens, 2, head_dim // 4)
    embeddings = freq_table[pos_ids_2d]  # (total_tokens, 2, head_dim // 4)
    # Flatten: (total_tokens, head_dim // 2)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    # Double: (total_tokens, head_dim)
    emb = jnp.concatenate([embeddings, embeddings], axis=-1)
    cos = jnp.cos(emb)
    sin = jnp.sin(emb)
    return cos, sin


def apply_rotary_pos_emb_vision(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply RoPE to vision Q/K tensors.

    Shapes:
        q, k: ``(seq_len, num_heads, head_dim)``
        cos, sin: ``(seq_len, head_dim)`` -- will be broadcast over heads.
    """
    orig_dtype = q.dtype
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    cos_ = cos[:, None, :].astype(jnp.float32)  # (seq, 1, head_dim)
    sin_ = sin[:, None, :].astype(jnp.float32)
    q_embed = q_f32 * cos_ + rotate_half(q_f32) * sin_
    k_embed = k_f32 * cos_ + rotate_half(k_f32) * sin_
    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


# ---------------------------------------------------------------------------
# Text MRoPE
# ---------------------------------------------------------------------------

def compute_mrope_cos_sin(
    position_ids: jnp.ndarray,
    head_dim: int,
    mrope_section: Tuple[int, ...],
    rope_theta: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute interleaved Multi-Resolution RoPE cos/sin for the text decoder.

    Args:
        position_ids: ``(3, batch, seq_len)`` int32 -- temporal, height, width.
        head_dim: per-head dimension (e.g. 128).
        mrope_section: frequency counts per dimension, e.g. ``(24, 20, 20)``.
            Sum must equal ``head_dim // 2``.
        rope_theta: base frequency for the inverse-frequency computation.

    Returns:
        ``(cos, sin)`` each of shape ``(batch, seq_len, head_dim)``.
    """
    half_dim = head_dim // 2
    inv_freq = 1.0 / (
        rope_theta
        ** (jnp.arange(0, half_dim * 2, 2, dtype=jnp.float32) / (half_dim * 2))
    )  # (half_dim,)

    # position_ids: (3, B, L)
    # inv_freq: (half_dim,) -> expand for broadcasting
    # freqs[d] = inv_freq * position_ids[d]
    # inv_freq_exp: (1, 1, half_dim, 1), pos_exp: (3, B, 1, L)
    inv_freq_exp = inv_freq[None, None, :, None]  # (1, 1, half_dim, 1)
    pos_exp = position_ids[:, :, None, :].astype(jnp.float32)  # (3, B, 1, L)
    freqs = (inv_freq_exp * pos_exp).transpose(0, 1, 3, 2)  # (3, B, L, half_dim)

    # Interleave: start with temporal, then scatter height/width at
    # interleaved positions.
    #
    # HF implementation:
    #   freqs_t = freqs[0].clone()
    #   for dim_idx, offset in enumerate((1, 2), start=1):
    #       length = mrope_section[dim_idx] * 3
    #       idx = slice(offset, length, 3)
    #       freqs_t[..., idx] = freqs[dim_idx, ..., idx]
    freqs_out = freqs[0]  # (B, L, half_dim) -- start with temporal

    for dim_idx, offset in ((1, 1), (2, 2)):
        length = mrope_section[dim_idx] * 3
        indices = jnp.arange(offset, length, 3)
        freqs_out = freqs_out.at[..., indices].set(
            freqs[dim_idx][..., indices]
        )

    # Double for full head_dim: (B, L, half_dim) -> (B, L, head_dim)
    emb = jnp.concatenate([freqs_out, freqs_out], axis=-1)
    cos = jnp.cos(emb)
    sin = jnp.sin(emb)
    return cos, sin


def apply_rotary_pos_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply RoPE to text Q/K tensors.

    Shapes:
        q: ``(batch, num_heads, seq_len, head_dim)``
        k: ``(batch, num_kv_heads, seq_len, head_dim)``
        cos, sin: ``(batch, seq_len, head_dim)`` -- unsqueezed on dim=1 for heads.
    """
    cos = cos[:, None, :, :]  # (B, 1, L, D)
    sin = sin[:, None, :, :]
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed
