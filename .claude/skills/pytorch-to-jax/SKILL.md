---
name: pytorch-to-jax
description: |
  Convert PyTorch/GPU training code to JAX/Flax for TPU. Use when the user needs to:
  (1) Migrate nn.Module models to Flax nn.Module (@nn.compact)
  (2) Convert training loops (.backward → jax.value_and_grad)
  (3) Replace torch operations with JAX/NumPy equivalents
  (4) Convert weight formats (HF safetensors ↔ Flax params)
  (5) Replace DeepSpeed/FSDP with JAX SPMD sharding
  (6) Replace torch DataLoader with NumPy-based data pipelines
  (7) Handle XLA compilation constraints (fixed shapes, JIT boundaries)
  Triggers: "convert to JAX", "migrate to Flax", "PyTorch to JAX", "torch to jax",
  "port to TPU", "rewrite in JAX", "JAX equivalent", "Flax version"
---

# PyTorch to JAX/Flax Migration

Convert PyTorch training code to JAX/Flax for TPU execution. See reference files for detailed patterns:

- **Model layers**: See [model-layers.md](references/model-layers.md) for nn.Module → Flax conversion
- **Training loop**: See [training.md](references/training.md) for loss, gradients, optimizer, accumulation
- **Data pipeline**: See [data-pipeline.md](references/data-pipeline.md) for torch tensor → numpy, XLA fixed shapes
- **Distribution**: See [distribution.md](references/distribution.md) for DeepSpeed → SPMD sharding
- **Weight conversion**: See [weights.md](references/weights.md) for HF safetensors ↔ Flax params

## Quick Reference Table

| PyTorch | JAX/Flax |
|---------|----------|
| `nn.Module` + `__init__`/`forward` | `nn.Module` + `@nn.compact`/`__call__` |
| `nn.Parameter(...)` | `self.param("name", init_fn, shape)` |
| `nn.Linear(in, out)` | `nn.Dense(out)` |
| `nn.Conv3d(in, out, k, s)` | `nn.Conv(out, k, s)` (channels-last) |
| `nn.LayerNorm` / `RMSNorm` | Custom `RMSNorm` with float32 variance |
| `loss.backward()` | `jax.value_and_grad(loss_fn)(params)` |
| `optimizer.step()` | `state.apply_gradients(grads=grads)` |
| `torch.Tensor` | `jnp.ndarray` (JIT) / `np.ndarray` (CPU) |
| `F.softmax(x)` | `jax.nn.softmax(x.astype(f32))` |
| `F.silu(x)` | `nn.silu(x)` |
| `F.gelu(x)` | `jax.nn.gelu(x, approximate=True)` |
| `x.transpose(-2, -1)` | `jnp.swapaxes(x, -2, -1)` |
| `torch.matmul(a, b)` | `jnp.matmul(a, b)` |
| `torch.where(cond, a, b)` | `jnp.where(cond, a, b)` |
| `x.reshape(...)` | `x.reshape(...)` |
| `torch.cat([a, b])` | `jnp.concatenate([a, b])` |
| `torch.stack([a, b])` | `jnp.stack([a, b])` |
| `x.repeat(n, 1)` | `jnp.repeat(x, n, axis=0)` |
| `torch.arange(n)` | `jnp.arange(n)` |
| `torch.cumsum(x, dim=0)` | `jnp.cumsum(x, axis=0)` |
| `F.cross_entropy(logits, labels)` | Custom: `log_softmax` + `take_along_axis` |
| `model.train()` / `model.eval()` | No-op (stateless model) |
| DeepSpeed ZeRO-2/3 | `Mesh` + `PartitionSpec` + `jax.device_put` |
| `torch.save` / `torch.load` | Orbax `CheckpointManager` |
| `gradient_checkpointing_enable()` | `nn.remat(Layer, policy=...)` |
| `torch.nn.utils.rnn.pad_sequence` | Custom `np.zeros` + slice assignment |

## Migration Workflow

```
1. Data Layer (CPU-side, pure NumPy)
   ├── Replace torchvision → PIL
   ├── Replace torch.Tensor → np.ndarray
   ├── Replace pad_sequence → custom numpy padding
   └── Add fixed-shape padding for XLA

2. Model Layer (Flax nn.Module)
   ├── Convert nn.Module → @nn.compact
   ├── Transpose weight shapes (out,in → in,out)
   ├── Add float32 upcasting for norms/softmax
   └── Replace Flash Attention → explicit mask + matmul

3. Training Layer (functional + JIT)
   ├── Replace .backward() → jax.value_and_grad
   ├── Replace optimizer.step() → state.apply_gradients
   ├── Replace param_groups → optax.multi_transform
   └── Replace gradient accumulation loop → jax.lax.scan

4. Distribution Layer (SPMD)
   ├── Replace DeepSpeed → Mesh + PartitionSpec
   ├── Replace DDP → data-parallel sharding
   └── Add multi-host support (host_local_array_to_global_array)

5. Weight Conversion
   ├── HF safetensors → Flax params (key mapping + transpose)
   └── Flax params → HF safetensors (reverse mapping + LoRA merge)
```

## Critical Rules

1. **Channels-last for Conv**: PyTorch `(N,C,H,W)` → JAX `(N,H,W,C)`. Always transpose.
2. **Dense kernel transpose**: PyTorch `(out,in)` → JAX `(in,out)`. Always transpose 2D weights.
3. **Float32 for numerics**: Softmax, RMSNorm variance, RoPE cos/sin, cross-entropy must use float32.
4. **Fixed shapes for XLA**: All tensors entering `@jax.jit` must have static shapes. Pad in DataCollator.
5. **No mutation in JIT**: JAX functions are pure. Return new state, don't modify in-place.
6. **Loss inside model**: `jax.value_and_grad` traces a loss function; model must return loss.
7. **CPU vs JIT boundary**: Data preprocessing (RoPE position IDs, vision grid) runs on CPU as NumPy. Only model forward + backward runs inside JIT.
