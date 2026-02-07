"""Shared layers for Qwen3-VL in Flax: RMSNorm, SwiGLU MLP, LoRADense."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (used in the text decoder)."""

    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param(
            "weight", nn.initializers.ones, (x.shape[-1],)
        )
        # Compute in float32 for numerical stability, then cast back
        orig_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
        normed = x_f32 * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight.astype(jnp.float32)).astype(orig_dtype)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP with gate_proj, up_proj, down_proj.

    Vision encoder uses bias=True; text decoder uses bias=False.
    """

    hidden_size: int
    intermediate_size: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.Dense(
            self.intermediate_size, use_bias=self.use_bias, name="gate_proj"
        )(x)
        up = nn.Dense(
            self.intermediate_size, use_bias=self.use_bias, name="up_proj"
        )(x)
        return nn.Dense(
            self.hidden_size, use_bias=self.use_bias, name="down_proj"
        )(nn.silu(gate) * up)


class VisionMLP(nn.Module):
    """Vision encoder MLP using fc1 / fc2 naming (matches HF Qwen3VLVisionMLP).

    This is a simple GELU MLP, NOT a gated MLP (the vision encoder does not use SwiGLU).
    """

    hidden_size: int
    intermediate_size: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.intermediate_size, use_bias=self.use_bias, name="linear_fc1"
        )(x)
        # gelu_pytorch_tanh approximation
        x = jax.nn.gelu(x, approximate=True)
        x = nn.Dense(
            self.hidden_size, use_bias=self.use_bias, name="linear_fc2"
        )(x)
        return x


class LoRADense(nn.Module):
    """Dense layer with optional LoRA branch.

    When ``lora_rank == 0`` this behaves as a normal ``nn.Dense``.
    """

    features: int
    use_bias: bool = False
    lora_rank: int = 0
    lora_alpha: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.Dense(
            self.features, use_bias=self.use_bias, name="base"
        )(x)
        if self.lora_rank > 0:
            scaling = self.lora_alpha / self.lora_rank
            a = nn.Dense(
                self.lora_rank,
                use_bias=False,
                name="lora_A",
                kernel_init=nn.initializers.he_uniform(),
            )(x)
            b = nn.Dense(
                self.features,
                use_bias=False,
                name="lora_B",
                kernel_init=nn.initializers.zeros,
            )(a)
            y = y + b * scaling
        return y
