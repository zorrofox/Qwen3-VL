"""Qwen3-VL Vision Encoder (ViT) in Flax.

Key components:
- PatchEmbed3D: 3D convolution for temporal-spatial patches.
- VisionAttention: Multi-head attention with combined QKV.
- VisionBlock: Pre-norm transformer block.
- PatchMerger: Spatial merge of 2x2 patches.
- VisionModel: Full ViT with DeepStack feature extraction.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from .config import Qwen3VLVisionConfig
from .layers import VisionMLP
from .rope import compute_vision_rotary_cos_sin, apply_rotary_pos_emb_vision


# ---------------------------------------------------------------------------
# Host-side precomputation (numpy, called BEFORE JIT)
# ---------------------------------------------------------------------------

def precompute_vision_position_ids(
    grid_thw: np.ndarray,
    spatial_merge_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute vision position IDs on the host (numpy, outside JIT).

    Args:
        grid_thw: ``(num_entries, 3)`` int array of (T, H, W) per image/video.
        spatial_merge_size: merge factor (typically 2).

    Returns:
        ``(pos_ids_2d, pos_ids_1d)`` where:
        - ``pos_ids_2d``: ``(total_tokens, 2)`` int32 -- (row, col) for rotary.
        - ``pos_ids_1d``: ``(total_tokens,)`` int32 -- for learned position embed.
    """
    merge = spatial_merge_size
    all_2d = []
    all_1d = []

    for idx in range(grid_thw.shape[0]):
        t = int(grid_thw[idx, 0])
        h = int(grid_thw[idx, 1])
        w = int(grid_thw[idx, 2])

        merged_h = h // merge
        merged_w = w // merge

        block_rows = np.arange(merged_h, dtype=np.int32)
        block_cols = np.arange(merged_w, dtype=np.int32)
        intra_row = np.arange(merge, dtype=np.int32)
        intra_col = np.arange(merge, dtype=np.int32)

        row_idx = (
            block_rows[:, None, None, None] * merge
            + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * merge
            + intra_col[None, None, None, :]
        )

        row_idx = np.broadcast_to(
            row_idx, (merged_h, merged_w, merge, merge)
        ).reshape(-1)
        col_idx = np.broadcast_to(
            col_idx, (merged_h, merged_w, merge, merge)
        ).reshape(-1)

        coords = np.stack([row_idx, col_idx], axis=-1)  # (h*w, 2)
        if t > 1:
            coords = np.tile(coords, (t, 1))
        all_2d.append(coords)

        pos_1d = row_idx * w + col_idx
        if t > 1:
            pos_1d = np.tile(pos_1d, t)
        all_1d.append(pos_1d)

    if len(all_2d) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.int32)
    return (
        np.concatenate(all_2d, axis=0).astype(np.int32),
        np.concatenate(all_1d, axis=0).astype(np.int32),
    )


def precompute_vision_cu_seqlens(grid_thw: np.ndarray) -> np.ndarray:
    """Compute cumulative sequence lengths on the host (numpy, outside JIT).

    Args:
        grid_thw: ``(num_entries, 3)`` int array of (T, H, W) per image/video.

    Returns:
        ``(num_segments + 1,)`` int32 with ``cu_seqlens[0] == 0``.
    """
    all_lens = []
    for idx in range(grid_thw.shape[0]):
        t = int(grid_thw[idx, 0])
        h = int(grid_thw[idx, 1])
        w = int(grid_thw[idx, 2])
        frame_len = h * w
        for _ in range(t):
            all_lens.append(frame_len)

    if len(all_lens) == 0:
        return np.array([0], dtype=np.int32)

    lens = np.array(all_lens, dtype=np.int32)
    cu = np.concatenate([np.array([0], dtype=np.int32), np.cumsum(lens)])
    return cu.astype(np.int32)


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """3-D convolution patch embedding.

    HF input convention:
        ``(batch*num_patches, in_channels * temporal_patch_size, patch_size, patch_size)``

    We reshape to 5-D ``(N, T, H, W, C)`` (channels-last for JAX) and apply
    a Conv that has kernel ``(temporal_patch_size, patch_size, patch_size)`` with
    matching strides, then flatten back to ``(total_patches, embed_dim)``.
    """

    config: Qwen3VLVisionConfig

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        cfg = self.config
        # hidden_states: (N, in_channels*temporal_patch_size, patch_size, patch_size)
        N = hidden_states.shape[0]
        # PyTorch layout: (N, C_in * T, H, W) -> reshape to (N, C, T, H, W)
        x = hidden_states.reshape(
            N,
            cfg.in_channels,
            cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.patch_size,
        )
        # (N, C, T, H, W) -> (N, T, H, W, C) for channels-last
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        # Flax nn.Conv for 3-D spatial dimensions with channels-last.
        x = nn.Conv(
            features=cfg.hidden_size,
            kernel_size=(cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size),
            strides=(cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size),
            padding="VALID",
            use_bias=True,
            name="proj",
        )(x)
        # x: (N, 1, 1, 1, hidden_size) -> flatten to (N, hidden_size)
        x = x.reshape(N, -1)
        return x


# ---------------------------------------------------------------------------
# Vision Attention
# ---------------------------------------------------------------------------

class VisionAttention(nn.Module):
    """Multi-head attention with combined QKV projection for the vision encoder."""

    config: Qwen3VLVisionConfig

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cu_seqlens: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
    ) -> jnp.ndarray:
        cfg = self.config
        head_dim = cfg.head_dim
        num_heads = cfg.num_heads
        seq_len = hidden_states.shape[0]

        # Combined QKV projection
        qkv = nn.Dense(
            3 * cfg.hidden_size, use_bias=True, name="qkv"
        )(hidden_states)  # (seq_len, 3*hidden)

        qkv = qkv.reshape(seq_len, 3, num_heads, head_dim)
        q = qkv[:, 0, :, :]  # (seq_len, num_heads, head_dim)
        k = qkv[:, 1, :, :]
        v = qkv[:, 2, :, :]

        # Apply vision rotary position embedding
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Build block-diagonal attention mask from cu_seqlens
        attn_mask = _build_block_diagonal_mask(cu_seqlens, seq_len, sharding_guide=hidden_states)

        # Scaled dot-product attention
        scaling = head_dim ** -0.5
        # (seq, heads, dim) -> (heads, seq, dim)
        q = jnp.transpose(q, (1, 0, 2))  # (H, S, D)
        k = jnp.transpose(k, (1, 0, 2))
        v = jnp.transpose(v, (1, 0, 2))

        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scaling
        # attn_mask: (S, S) -- True = attend
        attn_weights = jnp.where(
            attn_mask[None, :, :], attn_weights, jnp.finfo(attn_weights.dtype).min
        )
        attn_weights = jax.nn.softmax(
            attn_weights.astype(jnp.float32), axis=-1
        ).astype(hidden_states.dtype)
        attn_output = jnp.matmul(attn_weights, v)  # (H, S, D)

        # (H, S, D) -> (S, H, D) -> (S, hidden)
        attn_output = jnp.transpose(attn_output, (1, 0, 2))
        attn_output = attn_output.reshape(seq_len, -1)

        # Output projection
        attn_output = nn.Dense(
            cfg.hidden_size, use_bias=True, name="proj"
        )(attn_output)
        return attn_output


def _build_block_diagonal_mask(
    cu_seqlens: jnp.ndarray,
    total_len: int,
    sharding_guide: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Build a boolean block-diagonal mask from cumulative sequence lengths.

    Args:
        cu_seqlens: 1-D int32 of shape ``(num_segments + 1,)`` with
            ``cu_seqlens[0] == 0``.
        total_len: total number of tokens.

    Returns:
        ``(total_len, total_len)`` boolean mask where ``True`` means the
        pair belongs to the same segment (attend).
    """
    pos = jnp.arange(total_len, dtype=jnp.int32)
    if sharding_guide is not None and hasattr(sharding_guide, 'sharding'):
        # Propagate sharding from hidden_states (usually sharded on axis 0)
        pos = jax.lax.with_sharding_constraint(pos, sharding_guide.sharding)

    # segment_id[i] = how many cu_seqlens entries are <= i, minus 1
    belongs = (pos[:, None] >= cu_seqlens[None, :])  # (total_len, num_seg+1)
    segment_ids = jnp.sum(belongs.astype(jnp.int32), axis=-1) - 1  # (total_len,)
    mask = segment_ids[:, None] == segment_ids[None, :]
    return mask


# ---------------------------------------------------------------------------
# Vision Block
# ---------------------------------------------------------------------------

class VisionBlock(nn.Module):
    """Pre-norm transformer block for the vision encoder.

    Uses LayerNorm (not RMSNorm), matching HF Qwen3VLVisionBlock.
    """

    config: Qwen3VLVisionConfig

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cu_seqlens: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
    ) -> jnp.ndarray:
        cfg = self.config
        # Pre-norm attention
        residual = hidden_states
        hidden_states = nn.LayerNorm(
            epsilon=1e-6, name="norm1"
        )(hidden_states)
        hidden_states = VisionAttention(
            config=cfg, name="attn"
        )(hidden_states, cu_seqlens, cos, sin)
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = nn.LayerNorm(
            epsilon=1e-6, name="norm2"
        )(hidden_states)
        hidden_states = VisionMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            use_bias=True,
            name="mlp",
        )(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Patch Merger
# ---------------------------------------------------------------------------

class PatchMerger(nn.Module):
    """Spatially merge adjacent patches (default 2x2) using LayerNorm + GELU MLP.

    HF: ``Qwen3VLVisionPatchMerger``.

    When ``use_postshuffle_norm=False`` (final merger):
        - LayerNorm on each patch individually (hidden_size dim)
        - Then reshape to merged_dim and pass through MLP
    When ``use_postshuffle_norm=True`` (deepstack mergers):
        - Reshape to merged_dim first, then LayerNorm on merged_dim
        - Then pass through MLP
    """

    config: Qwen3VLVisionConfig
    use_postshuffle_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        cfg = self.config
        merged_dim = cfg.hidden_size * (cfg.spatial_merge_size ** 2)

        if self.use_postshuffle_norm:
            # Reshape first, then norm
            x_merged = x.reshape(-1, merged_dim)
            normed = nn.LayerNorm(epsilon=1e-6, name="norm")(x_merged)
        else:
            # Norm first (on hidden_size dim), then reshape
            normed = nn.LayerNorm(epsilon=1e-6, name="norm")(x)
            normed = normed.reshape(-1, merged_dim)

        out = nn.Dense(merged_dim, use_bias=True, name="linear_fc1")(normed)
        out = jax.nn.gelu(out, approximate=True)
        out = nn.Dense(cfg.out_hidden_size, use_bias=True, name="linear_fc2")(out)
        return out


# ---------------------------------------------------------------------------
# Vision Model
# ---------------------------------------------------------------------------

class VisionModel(nn.Module):
    """Complete Qwen3-VL Vision Transformer with DeepStack feature extraction.

    Returns:
        ``(merged_output, deepstack_features_list)``
        where ``merged_output`` has shape ``(num_merged_tokens, out_hidden_size)``
        and each entry in ``deepstack_features_list`` has the same shape.
    """

    config: Qwen3VLVisionConfig
    gradient_checkpointing: bool = False

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        grid_thw: jnp.ndarray,
        pos_ids_2d: Optional[jnp.ndarray] = None,
        pos_ids_1d: Optional[jnp.ndarray] = None,
        cu_seqlens: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        cfg = self.config

        # 1. Patch embedding
        hidden_states = PatchEmbed3D(config=cfg, name="patch_embed")(hidden_states)
        # hidden_states: (total_patches, hidden_size)

        # 2. Use precomputed position IDs (must be computed on host before JIT)
        assert pos_ids_2d is not None and pos_ids_1d is not None, (
            "pos_ids_2d and pos_ids_1d must be precomputed on host and passed in. "
            "Use precompute_vision_position_ids(grid_thw, spatial_merge_size)."
        )

        # 3. Learned position embedding
        pos_embed = nn.Embed(
            num_embeddings=cfg.num_position_embeddings,
            features=cfg.hidden_size,
            name="pos_embed",
        )(pos_ids_1d)  # (total_patches, hidden_size)
        hidden_states = hidden_states + pos_embed

        # 4. Rotary position embeddings using 2-D (row, col) indices
        cos, sin = compute_vision_rotary_cos_sin(
            pos_ids_2d, cfg.head_dim, theta=cfg.rope_theta
        )

        # 5. Use precomputed cu_seqlens
        assert cu_seqlens is not None, (
            "cu_seqlens must be precomputed on host and passed in. "
            "Use precompute_vision_cu_seqlens(grid_thw)."
        )

        # 6. Transformer blocks with DeepStack feature extraction
        deepstack_features: List[jnp.ndarray] = []
        deepstack_idx = 0
        # Select block class based on gradient checkpointing
        BlockClass = VisionBlock
        if self.gradient_checkpointing:
            BlockClass = nn.remat(VisionBlock)

        for layer_num in range(cfg.depth):
            hidden_states = BlockClass(
                config=cfg, name=f"blocks_{layer_num}"
            )(hidden_states, cu_seqlens, cos, sin)

            if layer_num in cfg.deepstack_visual_indexes:
                ds_merged = PatchMerger(
                    config=cfg,
                    use_postshuffle_norm=True,
                    name=f"deepstack_merger_list_{deepstack_idx}",
                )(hidden_states)
                deepstack_features.append(ds_merged)
                deepstack_idx += 1

        # 7. Final patch merger (Qwen3-VL has no ln_post before merger)
        merged_output = PatchMerger(
            config=cfg,
            use_postshuffle_norm=False,
            name="merger",
        )(hidden_states)

        return merged_output, deepstack_features

    @staticmethod
    def _compute_position_ids(
        grid_thw: jnp.ndarray,
        cfg: Qwen3VLVisionConfig,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute position IDs for each token from grid_thw.

        Implements the HF ``rot_pos_emb`` position indexing logic:
        For each (T, H, W) entry, compute 2-D (row, col) indices accounting
        for the spatial merge pattern.

        Returns:
            pos_ids_2d: ``(total_tokens, 2)`` int32 -- (row, col) for rotary
            pos_ids_1d: ``(total_tokens,)`` int32 -- flattened index for
                learned position embedding
        """
        merge = cfg.spatial_merge_size
        all_2d = []
        all_1d = []

        num_entries = grid_thw.shape[0]
        for idx in range(num_entries):
            t = int(grid_thw[idx, 0])
            h = int(grid_thw[idx, 1])
            w = int(grid_thw[idx, 2])

            merged_h = h // merge
            merged_w = w // merge

            # Build (row, col) index pairs matching the HF rot_pos_emb method.
            # block_rows/cols give merged block indices, intra_row/col give
            # offsets within each merge block.
            block_rows = jnp.arange(merged_h, dtype=jnp.int32)
            block_cols = jnp.arange(merged_w, dtype=jnp.int32)
            intra_row = jnp.arange(merge, dtype=jnp.int32)
            intra_col = jnp.arange(merge, dtype=jnp.int32)

            # row_idx: block_row * merge + intra_row
            # Shape: (merged_h, merged_w, merge, merge) -> flattened
            row_idx = (
                block_rows[:, None, None, None] * merge
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge
                + intra_col[None, None, None, :]
            )

            # Broadcast to (merged_h, merged_w, merge, merge) and flatten
            row_idx = jnp.broadcast_to(
                row_idx, (merged_h, merged_w, merge, merge)
            ).reshape(-1)
            col_idx = jnp.broadcast_to(
                col_idx, (merged_h, merged_w, merge, merge)
            ).reshape(-1)

            coords = jnp.stack([row_idx, col_idx], axis=-1)  # (h*w, 2)

            if t > 1:
                coords = jnp.tile(coords, (t, 1))

            all_2d.append(coords)

            # 1-D position for learned embedding: row * w + col
            pos_1d = row_idx * w + col_idx
            if t > 1:
                pos_1d = jnp.tile(pos_1d, t)
            all_1d.append(pos_1d)

        if len(all_2d) == 0:
            return (
                jnp.zeros((0, 2), dtype=jnp.int32),
                jnp.zeros(0, dtype=jnp.int32),
            )
        return (
            jnp.concatenate(all_2d, axis=0).astype(jnp.int32),
            jnp.concatenate(all_1d, axis=0).astype(jnp.int32),
        )

    @staticmethod
    def _compute_cu_seqlens(grid_thw: jnp.ndarray) -> jnp.ndarray:
        """Compute cumulative sequence lengths from grid_thw.

        Each entry (T, H, W) produces T segments of H*W tokens each.
        """
        all_lens = []
        num_entries = grid_thw.shape[0]
        for idx in range(num_entries):
            t = int(grid_thw[idx, 0])
            h = int(grid_thw[idx, 1])
            w = int(grid_thw[idx, 2])
            frame_len = h * w
            for _ in range(t):
                all_lens.append(frame_len)

        if len(all_lens) == 0:
            return jnp.array([0], dtype=jnp.int32)

        lens = jnp.array(all_lens, dtype=jnp.int32)
        cu = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(lens)])
        return cu
