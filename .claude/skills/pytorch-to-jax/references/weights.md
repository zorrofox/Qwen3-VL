# Weight Conversion: HF Safetensors ↔ Flax Params

## Table of Contents

1. [HF → Flax (Loading)](#hf-to-flax)
2. [Flax → HF (Exporting)](#flax-to-hf)
3. [LoRA Merge](#lora-merge)
4. [Key Mapping Tables](#key-mapping)
5. [Gotchas](#gotchas)

## HF to Flax

### Overall flow

```
1. Load safetensors files (may be sharded: model-00001-of-00004.safetensors)
2. For each key-value pair:
   a. Map PyTorch key → Flax path tuple
   b. Transpose tensor if needed
   c. Convert dtype (e.g., float32 → bfloat16)
3. Build nested param dict from path tuples
```

### Key mapping function

```python
def _pytorch_key_to_flax_path(key):
    # Strip model prefix (two variants)
    if key.startswith("model.language_model."):
        key = key[len("model.language_model."):]
    elif key.startswith("model."):
        key = key[len("model."):]

    # Handle special cases
    if key == "embed_tokens.weight":
        return ("embed_tokens",)  # Direct leaf, not nn.Embed
    if key == "lm_head.weight":
        return ("lm_head", "kernel")

    # Replace dots with tuple elements
    # "layers.0.self_attn.q_proj.weight" → ("blocks_0", "attention", "q_proj", "kernel")
    parts = key.split(".")
    # ... mapping logic
    return tuple(parts)
```

### Transpose rules

```python
def _maybe_transpose(key, tensor):
    if tensor.ndim == 2 and "kernel" in key:
        # Dense: (out, in) → (in, out)
        return tensor.T
    if tensor.ndim == 5:
        # Conv3D: (out, in, T, H, W) → (T, H, W, in, out)
        return np.transpose(tensor, (2, 3, 4, 1, 0))
    # Bias, norm weight, embedding: no transpose
    return tensor
```

### Building nested dict

```python
def _set_nested(d, path, value):
    """Set value in nested dict using path tuple."""
    for key in path[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[path[-1]] = value

# Usage:
params = {}
for pt_key, tensor in safetensors_data.items():
    flax_path = _pytorch_key_to_flax_path(pt_key)
    tensor = _maybe_transpose(flax_path[-1], tensor)
    tensor = tensor.astype(target_dtype)
    _set_nested(params, flax_path, tensor)
```

### Sharded model loading

```python
# HF models > 10GB are split into multiple safetensors files
# model.safetensors.index.json maps keys → file names
import json
with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
    index = json.load(f)
    weight_map = index["weight_map"]  # {"layer.0.weight": "model-00001-of-00004.safetensors"}

# Load each shard and process its keys
for shard_file in set(weight_map.values()):
    with safe_open(os.path.join(model_dir, shard_file), framework="numpy") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # ... process
```

## Flax to HF

### Overall flow

```
1. Optionally merge LoRA weights
2. Flatten nested Flax param dict to flat key-value pairs
3. Map Flax path → PyTorch key
4. Reverse transpose tensors
5. Save as safetensors (with optional sharding)
```

### Reverse key mapping

```python
def _flax_path_to_pytorch_key(path):
    # ("blocks_0", "attention", "q_proj", "kernel")
    # → "model.layers.0.self_attn.q_proj.weight"
    parts = list(path)

    # Reverse special mappings
    if parts[-1] == "kernel":
        parts[-1] = "weight"
    if parts[0] == "embed_tokens":
        return "model.embed_tokens.weight"

    # "blocks_N" → "layers.N"
    for i, p in enumerate(parts):
        if p.startswith("blocks_"):
            parts[i] = "layers." + p[7:]  # "blocks_0" → "layers.0"

    return "model." + ".".join(parts)
```

### Reverse transpose

```python
def _reverse_transpose(key, tensor):
    if tensor.ndim == 2 and "weight" in key:
        # Flax (in, out) → PyTorch (out, in)
        return tensor.T
    if tensor.ndim == 5:
        # Flax (T, H, W, in, out) → PyTorch (out, in, T, H, W)
        return np.transpose(tensor, (4, 3, 0, 1, 2))
    return tensor
```

### Sharded saving

```python
def save_sharded(tensors, output_dir, max_shard_size=5 * 1024**3):
    """Save tensors across multiple safetensors files if total > max_shard_size."""
    total_size = sum(t.nbytes for t in tensors.values())

    if total_size <= max_shard_size:
        save_file(tensors, os.path.join(output_dir, "model.safetensors"))
        return

    # Split into shards
    shards = []
    current_shard = {}
    current_size = 0
    for key, tensor in tensors.items():
        if current_size + tensor.nbytes > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor.nbytes
    if current_shard:
        shards.append(current_shard)

    # Save each shard + index
    weight_map = {}
    for i, shard in enumerate(shards):
        filename = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(output_dir, filename))
        for key in shard:
            weight_map[key] = filename

    # Write index
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
```

## LoRA Merge

Before exporting, merge LoRA weights back into base weights:

```python
def merge_lora(params, lora_alpha, lora_rank):
    """Recursively merge LoRA weights in param tree."""
    def _merge_node(node):
        if isinstance(node, dict):
            # Check for LoRA structure: {base: {kernel}, lora_A: {kernel}, lora_B: {kernel}}
            if "base" in node and "lora_A" in node and "lora_B" in node:
                base_kernel = node["base"]["kernel"]
                lora_a = node["lora_A"]["kernel"]
                lora_b = node["lora_B"]["kernel"]
                scale = lora_alpha / lora_rank
                merged = base_kernel + (lora_a @ lora_b) * scale
                # Replace LoRA structure with merged kernel
                result = {k: v for k, v in node.items()
                         if k not in ("base", "lora_A", "lora_B")}
                result["kernel"] = merged
                return result
            return {k: _merge_node(v) for k, v in node.items()}
        return node

    return _merge_node(params)
```

## Key Mapping

### Vision encoder

| PyTorch key | Flax path |
|-------------|-----------|
| `model.visual.blocks.{N}.attn.qkv.weight` | `visual/blocks_{N}/attention/qkv/kernel` |
| `model.visual.blocks.{N}.norm1.weight` | `visual/blocks_{N}/norm1/scale` |
| `model.visual.blocks.{N}.mlp.fc1.weight` | `visual/blocks_{N}/mlp/linear_fc1/kernel` |
| `model.visual.patch_embed.proj.weight` | `visual/patch_embed/proj/kernel` |
| `model.visual.merger.mlp.0.weight` | `visual/merger/dense_0/kernel` |

### Text decoder

| PyTorch key | Flax path |
|-------------|-----------|
| `model.layers.{N}.self_attn.q_proj.weight` | `blocks_{N}/attention/q_proj/kernel` |
| `model.layers.{N}.self_attn.q_norm.weight` | `blocks_{N}/attention/q_norm/weight` |
| `model.layers.{N}.mlp.gate_proj.weight` | `blocks_{N}/mlp/gate_proj/kernel` |
| `model.layers.{N}.input_layernorm.weight` | `blocks_{N}/input_layernorm/weight` |
| `model.embed_tokens.weight` | `embed_tokens` (direct leaf) |
| `model.norm.weight` | `final_norm/weight` |
| `lm_head.weight` | `lm_head/kernel` |

### Dual prefix support

Qwen3-VL uses `model.language_model.` prefix; Qwen2.5-VL uses `model.`. Support both:

```python
if key.startswith("model.language_model."):
    key = key[len("model.language_model."):]
elif key.startswith("model."):
    key = key[len("model."):]
```

## Gotchas

1. **embed_tokens is a direct leaf**: In Flax, `embed_tokens` is `self.param("embed_tokens", ...)`, not `nn.Embed`. The weight maps to a single array, not `{"embedding": array}`.

2. **tie_word_embeddings**: Qwen3-VL-2B has `tie_word_embeddings=True` — no separate `lm_head`. Skip `lm_head.weight` during loading.

3. **Vision norm uses `scale`**: Vision encoder norms map `weight` → `scale` in Flax (convention from `nn.LayerNorm`). Text decoder norms keep `weight` → `weight` (custom `RMSNorm`).

4. **Conv3D 5-axis transpose**: Easy to get wrong. PyTorch `(out,in,T,H,W)` ↔ Flax `(T,H,W,in,out)`. Always test with a known input.

5. **bfloat16 conversion**: Convert during loading, not after building param tree. Avoids doubling memory temporarily.

6. **Safetensors framework**: Use `framework="numpy"` (not `"pt"` or `"flax"`) for loading, since we want pure numpy arrays.
