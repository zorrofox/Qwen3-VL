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

        # Attention 计算
        # RoPE (cos/sin float32) 让 q/k upcast 为 float32，v 仍是 bf16，统一 cast
        compute_dtype = hidden_states.dtype
        scaling = head_dim ** -0.5
        q = q.astype(compute_dtype)
        k = k.astype(compute_dtype)
        v = v.astype(compute_dtype)

        from jax_qwenvl.train.sharding import get_global_mesh  # noqa
        global_mesh, sharding_mode = get_global_mesh()

        if global_mesh is not None and jax.default_backend() == "tpu":
            # ── Splash Attention via shard_map(vmap(kernel)) — O(L) 内存 ──────
            # 参考 MaxText 实现：mask 内嵌到 kernel，vmap 在 batch 维度上遍历
            from jax.experimental.pallas.ops.tpu.splash_attention import (  # noqa
                splash_attention_kernel as _sak,
                splash_attention_mask as _sam,
            )
            from jax import shard_map  # noqa
            from jax.sharding import PartitionSpec as P  # noqa

            # 批次分区 spec（与 shard_batch 中的 data_axis 一致）
            if sharding_mode == 'hybrid':
                batch_spec = P(('dp', 'fsdp'), None, None, None)
            elif sharding_mode == 'fsdp':
                batch_spec = P('fsdp', None, None, None)
            else:
                batch_spec = P('dp', None, None, None)

            # 构造 CausalMask 并内嵌到 kernel（不需要传 ab/float mask）
            # 对于非 packed 序列：CausalMask 即正确的 mask
            # 对于 packed 序列：后续通过 segment_ids 支持（当前 benchmark 不启用 packing）
            _L = L
            _H_q = num_heads
            block_q = min(512, _L)
            block_kv = min(512, _L)

            causal_mask = _sam.CausalMask(shape=(_L, _L))
            multi_head_mask = _sam.MultiHeadMask(masks=(causal_mask,) * _H_q)
            splash_kernel = _sak.make_splash_mha(
                mask=multi_head_mask,
                block_sizes=_sak.BlockSizes(
                    block_q=block_q,
                    block_kv=block_kv,
                    block_kv_compute=block_kv,
                ),
                head_shards=1,    # 本实现中 heads 不跨 mesh 轴分片
                q_seq_shards=1,   # 本实现中序列不跨 mesh 轴分片
            )

            def _splash_fn(q_l, k_l, v_l):
                # 在 shard_map 内：q_l (B_local, H_q, L, D), k_l (B_local, H_kv, L, D)
                # GQA 展开（splash_attention 不原生支持 GQA）
                if num_kv_groups > 1:
                    k_l = jnp.repeat(k_l, num_kv_groups, axis=1)
                    v_l = jnp.repeat(v_l, num_kv_groups, axis=1)
                # vmap 在 batch 维度上遍历（MaxText 正式模式）
                # 每次 kernel 输入：(H_q, L, D) → 输出 (H_q, L, D)
                return jax.vmap(splash_kernel)(q_l, k_l, v_l)  # → (B_local, H_q, L, D)

            attn_output = shard_map(
                _splash_fn,
                mesh=global_mesh,
                in_specs=(batch_spec, batch_spec, batch_spec),
                out_specs=batch_spec,
                check_vma=False,
            )(q, k, v)  # → (B, H_q, L, D)

            # (B, H_q, L, D) → (B, L, H_q*D)
            attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
            attn_output = attn_output.reshape(B, L, -1)

        else:
            # ── CPU/GPU 回退 / mesh 未注册 ─────────────────────────────────────
            if num_kv_groups > 1:
                k = jnp.repeat(k, num_kv_groups, axis=1)
                v = jnp.repeat(v, num_kv_groups, axis=1)
            q_t = jnp.transpose(q, (0, 2, 1, 3))
            k_t = jnp.transpose(k, (0, 2, 1, 3))
            v_t = jnp.transpose(v, (0, 2, 1, 3))
            attn_output = jax.nn.dot_product_attention(
                q_t, k_t, v_t,
                bias=attention_mask,
                scale=scaling,
            )  # → (B, L, H, D)
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
