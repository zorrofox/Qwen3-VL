# Model Layer Conversion: PyTorch nn.Module → Flax nn.Module

## Table of Contents

1. [Module Structure](#module-structure)
2. [Normalization (RMSNorm)](#normalization)
3. [Dense / Linear](#dense-linear)
4. [LoRA](#lora)
5. [MLP (SwiGLU / GELU)](#mlp)
6. [Attention (GQA)](#attention)
7. [Conv3D (Patch Embedding)](#conv3d)
8. [Gradient Checkpointing (remat)](#gradient-checkpointing)
9. [Gotchas](#gotchas)

## Module Structure

### PyTorch

```python
class MyLayer(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.eps = eps

    def forward(self, x):
        return self.linear(x * self.weight)
```

### Flax

```python
class MyLayer(nn.Module):
    hidden_size: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        weight = self.param("weight", nn.initializers.ones, (self.hidden_size,))
        return nn.Dense(self.hidden_size)(x * weight)
```

### Key differences

- Config via class attributes (dataclass-style), not `__init__` args
- Parameters declared with `self.param("name", initializer, shape)` — lazy, not eager
- Sub-modules declared inline inside `__call__` with `@nn.compact`
- No `super().__init__()` call
- No `self.xxx = nn.Linear(...)` in init — sub-modules are inline

## Normalization

### PyTorch RMSNorm

```python
def forward(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + self.eps) * self.weight
```

### Flax RMSNorm

```python
@nn.compact
def __call__(self, x):
    weight = self.param("weight", nn.initializers.ones, (x.shape[-1],))
    # MUST compute in float32 for numerical stability
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(variance + self.eps)
    return (normed * weight.astype(jnp.float32)).astype(x.dtype)
```

### Weight naming

- PyTorch norm: `weight` parameter
- Flax norm: `weight` parameter (via `self.param("weight", ...)`)
- HF safetensors for vision norm: key ends in `.weight` → Flax uses `scale`

## Dense / Linear

### PyTorch

```python
self.proj = nn.Linear(in_features, out_features, bias=True)
# Weight shape: (out_features, in_features)
```

### Flax

```python
proj = nn.Dense(out_features, use_bias=True, name="proj")
# Kernel shape: (in_features, out_features) — TRANSPOSED
```

### Weight transpose rule

```python
# Loading PyTorch weights into Flax:
if tensor.ndim == 2:
    flax_kernel = pytorch_weight.T  # (out,in) → (in,out)
```

## LoRA

### PyTorch (separate modules)

```python
self.base = nn.Linear(in_f, out_f)
self.lora_A = nn.Linear(in_f, rank, bias=False)
self.lora_B = nn.Linear(rank, out_f, bias=False)
nn.init.zeros_(self.lora_B.weight)

def forward(self, x):
    return self.base(x) + self.lora_B(self.lora_A(x)) * (alpha / rank)
```

### Flax (single class, optional LoRA)

```python
class LoRADense(nn.Module):
    features: int
    lora_rank: int = 0
    lora_alpha: float = 1.0

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.features, name="base")(x)
        if self.lora_rank > 0:
            a = nn.Dense(self.lora_rank, name="lora_A",
                        kernel_init=nn.initializers.he_uniform())(x)
            b = nn.Dense(self.features, name="lora_B",
                        kernel_init=nn.initializers.zeros)(a)
            y = y + b * (self.lora_alpha / self.lora_rank)
        return y
```

When `lora_rank=0`, degrades to plain Dense. LoRA-B initialized to zeros ensures zero initial contribution.

### LoRA merge (for export)

```python
merged_kernel = base_kernel + (lora_a @ lora_b) * (alpha / rank)
```

## MLP

### SwiGLU (text decoder)

```python
# PyTorch
gate = F.silu(self.gate_proj(x))
up = self.up_proj(x)
return self.down_proj(gate * up)

# Flax
gate = nn.Dense(intermediate, name="gate_proj")(x)
up = nn.Dense(intermediate, name="up_proj")(x)
return nn.Dense(hidden, name="down_proj")(nn.silu(gate) * up)
```

### GELU MLP (vision encoder)

```python
# PyTorch
x = self.fc1(x)
x = F.gelu(x, approximate='tanh')  # pytorch_tanh
return self.fc2(x)

# Flax
x = nn.Dense(intermediate, name="linear_fc1")(x)
x = jax.nn.gelu(x, approximate=True)  # matches pytorch_tanh
return nn.Dense(hidden, name="linear_fc2")(x)
```

## Attention

### PyTorch GQA

```python
q = self.q_proj(x).reshape(B, L, num_heads, head_dim).transpose(1, 2)
k = self.k_proj(x).reshape(B, L, num_kv_heads, head_dim).transpose(1, 2)
# Expand KV heads
k = k.repeat_interleave(num_groups, dim=1)
v = v.repeat_interleave(num_groups, dim=1)
attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

### Flax GQA

```python
q = nn.Dense(num_heads * head_dim, name="q_proj")(x)
q = q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
k = nn.Dense(num_kv_heads * head_dim, name="k_proj")(x)
k = k.reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

# Qwen3-specific: per-head RMSNorm before RoPE
q = RMSNorm(name="q_norm")(q)
k = RMSNorm(name="k_norm")(k)

# Apply RoPE
q, k = apply_rotary_pos_emb(q, k, cos, sin)

# Expand KV heads via repeat
if num_kv_groups > 1:
    k = jnp.repeat(k, num_kv_groups, axis=1)
    v = jnp.repeat(v, num_kv_groups, axis=1)

# Manual scaled dot-product (no F.scaled_dot_product_attention in JAX)
attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
attn = jnp.where(mask, attn, jnp.finfo(attn.dtype).min)
attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
output = jnp.matmul(attn, v)
```

### Block-diagonal mask (replaces Flash Attention varlen)

```python
def _build_block_diagonal_mask(cu_seqlens, total_len):
    pos = jnp.arange(total_len)
    belongs = (pos[:, None] >= cu_seqlens[None, :])
    segment_ids = jnp.sum(belongs.astype(jnp.int32), axis=-1) - 1
    return segment_ids[:, None] == segment_ids[None, :]
```

## Conv3D

### PyTorch (channels-first)

```python
# Input: (N, in_ch, T, H, W)
self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=(t,h,w), stride=(t,h,w))
# Weight shape: (out_ch, in_ch, T, H, W)
```

### Flax (channels-last)

```python
# Input must be (N, T, H, W, in_ch)
x = jnp.transpose(x, (0, 2, 3, 4, 1))  # channels-first → channels-last

conv = nn.Conv(
    features=out_ch,
    kernel_size=(t, h, w),
    strides=(t, h, w),
    padding="VALID",
)
# Kernel shape: (T, H, W, in_ch, out_ch) — TRANSPOSED from PyTorch
```

### Weight transpose for Conv3D

```python
# PyTorch (out, in, T, H, W) → JAX (T, H, W, in, out)
if tensor.ndim == 5:
    flax_kernel = np.transpose(pytorch_weight, (2, 3, 4, 1, 0))
```

## Gradient Checkpointing

### PyTorch

```python
model.gradient_checkpointing_enable()
# Or per-layer:
from torch.utils.checkpoint import checkpoint
output = checkpoint(layer, input)
```

### Flax (nn.remat)

```python
# Wrap layer class before instantiation
if gradient_checkpointing:
    DecoderLayerClass = nn.remat(
        DecoderLayer,
        policy=jax.checkpoint_policies.nothing_saveable
    )
else:
    DecoderLayerClass = DecoderLayer

# Use wrapped class in model
for i in range(num_layers):
    hidden = DecoderLayerClass(config)(hidden)
```

**Policies:**
- `nothing_saveable`: Minimum memory, recompute everything (recommended for TPU)
- `dots_with_no_batch_dims`: Save some small activations

## Gotchas

1. **No `model.train()` / `model.eval()`**: JAX models are stateless. Dropout must use explicit `deterministic` flag.

2. **Weight naming mismatch**: HF safetensors may use `model.language_model.` or `model.` prefix — support both.

3. **`tie_word_embeddings`**: Some models share `embed_tokens` and `lm_head`. Check config and skip `lm_head` loading if tied.

4. **Transpose only Dense/Conv weights**: Embedding, bias, and norm parameters are NOT transposed.

5. **GELU approximate**: PyTorch `gelu(approximate='tanh')` = JAX `gelu(approximate=True)`. Without this, outputs differ.

6. **Softmax always in float32**: Even in bfloat16 training, softmax must compute in float32 to avoid overflow/underflow.
