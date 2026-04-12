"""Qwen3-VL Text Decoder (LLM) in Flax.

Key components:
- TextAttention: GQA with q_norm and k_norm (Qwen3-specific) and MRoPE.
- DecoderLayer: Pre-norm decoder block.
- TextModel: Stack of decoder layers with DeepStack visual injection.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .config import Qwen3VLTextConfig
from .layers import RMSNorm, SwiGLUMLP, LoRADense
from .rope import compute_mrope_cos_sin, apply_rotary_pos_emb


# ---------------------------------------------------------------------------
# Text Attention (GQA with QK-Norm)
# ---------------------------------------------------------------------------

class TextAttention(nn.Module):
    """Grouped-Query Attention with per-head RMSNorm on Q and K (Qwen3-style)."""

    config: Qwen3VLTextConfig
    lora_rank: int = 0
    lora_alpha: float = 1.0

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        cfg = self.config
        B, L, _ = hidden_states.shape
        head_dim = cfg.head_dim
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        num_kv_groups = cfg.num_key_value_groups

        # Projections
        if self.lora_rank > 0:
            q = LoRADense(
                num_heads * head_dim,
                use_bias=cfg.attention_bias,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                name="q_proj",
            )(hidden_states)
            k = LoRADense(
                num_kv_heads * head_dim,
                use_bias=cfg.attention_bias,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                name="k_proj",
            )(hidden_states)
            v = LoRADense(
                num_kv_heads * head_dim,
                use_bias=cfg.attention_bias,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                name="v_proj",
            )(hidden_states)
        else:
            q = nn.Dense(
                num_heads * head_dim,
                use_bias=cfg.attention_bias,
                name="q_proj",
            )(hidden_states)
            k = nn.Dense(
                num_kv_heads * head_dim,
                use_bias=cfg.attention_bias,
                name="k_proj",
            )(hidden_states)
            v = nn.Dense(
                num_kv_heads * head_dim,
                use_bias=cfg.attention_bias,
                name="v_proj",
            )(hidden_states)

        # Reshape: (B, L, H*D) -> (B, L, H, D)
        q = q.reshape(B, L, num_heads, head_dim)
        k = k.reshape(B, L, num_kv_heads, head_dim)
        v = v.reshape(B, L, num_kv_heads, head_dim)

        # Per-head RMSNorm on Q and K (Qwen3-specific)
        q = RMSNorm(eps=cfg.rms_norm_eps, name="q_norm")(q)
        k = RMSNorm(eps=cfg.rms_norm_eps, name="k_norm")(k)

        # Transpose to (B, H, L, D) for attention
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Apply MRoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Pallas Flash Attention（TPU）/ jax.nn.dot_product_attention（CPU/GPU 回退）
        # 注意：RoPE (cos/sin float32) 让 q/k upcast 为 float32，v 仍是 bf16
        # 统一 cast 后再计算
        compute_dtype = hidden_states.dtype
        scaling = head_dim ** -0.5

        # GQA 展开：Pallas 不支持 GQA，dot_product_attention 支持但需要 cast
        # 两条路都需要展开 K/V，保持一致
        if num_kv_groups > 1:
            k = jnp.repeat(k, num_kv_groups, axis=1)  # (B, H, L, D)
            v = jnp.repeat(v, num_kv_groups, axis=1)

        q = q.astype(compute_dtype)
        k = k.astype(compute_dtype)
        v = v.astype(compute_dtype)

        if jax.default_backend() == "tpu":
            # Pallas Flash Attention：真正的 O(L) 内存
            # Pallas kernel 要求 ab 精确形状 (B, H, L, L)，不接受 (B, 1, L, L) 广播
            from jax.experimental.pallas.ops.tpu import flash_attention as tpu_fa  # noqa
            num_heads = q.shape[1]
            ab = (jnp.broadcast_to(attention_mask, (B, num_heads, L, L))
                  if attention_mask is not None else None)
            attn_output = tpu_fa.flash_attention(
                q, k, v,
                ab=ab,          # (B, H, L, L) 精确形状
                sm_scale=scaling,
            )  # → (B, H, L, D)
        else:
            # CPU/GPU 回退：jax.nn.dot_product_attention
            # 期望 (B, L, H, D)，需要 transpose
            q_t = jnp.transpose(q, (0, 2, 1, 3))
            k_t = jnp.transpose(k, (0, 2, 1, 3))
            v_t = jnp.transpose(v, (0, 2, 1, 3))
            attn_output = jax.nn.dot_product_attention(
                q_t, k_t, v_t, bias=attention_mask, scale=scaling,
            )  # → (B, L, H, D)
            attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # → (B, H, L, D)

        # Reshape: (B, H, L, D) -> (B, L, H*D)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(B, L, -1)

        # Output projection
        if self.lora_rank > 0:
            attn_output = LoRADense(
                cfg.hidden_size,
                use_bias=cfg.attention_bias,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                name="o_proj",
            )(attn_output)
        else:
            attn_output = nn.Dense(
                cfg.hidden_size,
                use_bias=cfg.attention_bias,
                name="o_proj",
            )(attn_output)

        return attn_output


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Pre-norm decoder layer with SwiGLU MLP (text decoder uses bias=False)."""

    config: Qwen3VLTextConfig
    lora_rank: int = 0
    lora_alpha: float = 1.0

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        cfg = self.config

        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = RMSNorm(
            eps=cfg.rms_norm_eps, name="input_layernorm"
        )(hidden_states)
        hidden_states = TextAttention(
            config=cfg,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            name="self_attn",
        )(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = RMSNorm(
            eps=cfg.rms_norm_eps, name="post_attention_layernorm"
        )(hidden_states)
        hidden_states = SwiGLUMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            use_bias=False,
            name="mlp",
        )(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# DeepStack injection
# ---------------------------------------------------------------------------

def _deepstack_inject(
    hidden_states: jnp.ndarray,
    visual_pos_masks: jnp.ndarray,
    visual_embeds: jnp.ndarray,
) -> jnp.ndarray:
    """Add visual embeddings at vision token positions.

    Args:
        hidden_states: ``(batch, seq_len, hidden_size)``
        visual_pos_masks: ``(batch, seq_len)`` bool
        visual_embeds: ``(total_vision_tokens, hidden_size)`` -- same
            total count as True values in ``visual_pos_masks``.

    Returns:
        Updated hidden_states with visual embeddings added at masked positions.
    """
    # Expand visual_embeds to match hidden_states layout by scattering
    # into a zero tensor of the same shape.
    B, L, D = hidden_states.shape
    expanded = jnp.zeros_like(hidden_states)
    # visual_pos_masks: (B, L) bool -- True at vision positions
    # We need to scatter visual_embeds into expanded at those positions.
    # Use cumulative sum to map flat visual_embeds into the right positions.
    mask_flat = visual_pos_masks.reshape(-1)  # (B*L,)
    cumsum = jnp.cumsum(mask_flat.astype(jnp.int32)) - 1  # index into visual_embeds
    # Clamp to valid range
    safe_idx = jnp.clip(cumsum, 0, visual_embeds.shape[0] - 1)
    # Gather visual embeddings for all positions (invalid positions will be zeroed by mask)
    gathered = visual_embeds[safe_idx]  # (B*L, D)
    gathered = gathered * mask_flat[:, None]  # zero out non-vision positions
    expanded = gathered.reshape(B, L, D)

    return hidden_states + expanded


# ---------------------------------------------------------------------------
# Text Model
# ---------------------------------------------------------------------------

class TextModel(nn.Module):
    """Stack of decoder layers with MRoPE and DeepStack visual feature injection."""

    config: Qwen3VLTextConfig
    lora_rank: int = 0
    lora_alpha: float = 1.0
    gradient_checkpointing: bool = False

    @nn.compact
    def __call__(
        self,
        inputs_embeds: jnp.ndarray,
        position_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        visual_pos_masks: Optional[jnp.ndarray] = None,
        deepstack_visual_embeds: Optional[List[jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Forward pass through the text decoder.

        Args:
            inputs_embeds: ``(batch, seq_len, hidden_size)``
            position_ids: ``(3, batch, seq_len)`` int32
            attention_mask: causal mask ``(batch, 1, seq_len, seq_len)`` float
                with 0 for attend and large negative for mask. Or None.
            visual_pos_masks: ``(batch, seq_len)`` bool, True at vision tokens.
            deepstack_visual_embeds: list of ``(total_vis_tokens, hidden_size)``
                tensors to inject at early decoder layers.

        Returns:
            ``(batch, seq_len, hidden_size)`` hidden states after all layers + norm.
        """
        cfg = self.config

        # Compute MRoPE cos/sin from position_ids
        cos, sin = compute_mrope_cos_sin(
            position_ids, cfg.head_dim, cfg.mrope_section, cfg.rope_theta
        )

        hidden_states = inputs_embeds

        # Process through decoder layers
        num_ds = (
            len(deepstack_visual_embeds)
            if deepstack_visual_embeds is not None
            else 0
        )

        # Select layer class based on gradient checkpointing
        LayerClass = DecoderLayer
        if self.gradient_checkpointing:
            LayerClass = nn.remat(
                DecoderLayer,
                policy=jax.checkpoint_policies.nothing_saveable,
            )

        for layer_idx in range(cfg.num_hidden_layers):
            hidden_states = LayerClass(
                config=cfg,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                name=f"layers_{layer_idx}",
            )(hidden_states, cos, sin, attention_mask)

            # DeepStack injection: inject at layers 0, 1, 2, ...
            # (matching the number of deepstack features extracted from the
            # vision encoder at deepstack_visual_indexes).
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < num_ds
            ):
                hidden_states = _deepstack_inject(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        # Final RMSNorm
        hidden_states = RMSNorm(eps=cfg.rms_norm_eps, name="norm")(hidden_states)
        return hidden_states
