# Data Pipeline Conversion

## Table of Contents

1. [Tensor Format](#tensor-format)
2. [Vision Preprocessing](#vision-preprocessing)
3. [DataCollator Pattern](#datacollator-pattern)
4. [Fixed-Shape Padding for XLA](#fixed-shape-padding)
5. [CPU vs JIT Boundary](#cpu-vs-jit-boundary)
6. [Gotchas](#gotchas)

## Tensor Format

### PyTorch pipeline

```python
# HF processor returns torch.Tensor
processed = processor(images=image, text=prompt, return_tensors="pt")
pixel_values = processed["pixel_values"]  # torch.Tensor
input_ids = processed["input_ids"]        # torch.Tensor

# DataLoader yields batches of torch.Tensor
for batch in DataLoader(dataset, collate_fn=collator):
    batch = {k: v.to(device) for k, v in batch.items()}
```

### JAX pipeline

```python
# Convert HF processor output to numpy immediately
def _ensure_numpy(val):
    """Duck-typing conversion — no import torch needed."""
    if hasattr(val, 'numpy'):  # torch.Tensor
        return val.numpy()
    if isinstance(val, list):
        return np.array(val)
    return val

# Force TPU-friendly dtypes
if val.dtype.kind == 'f':
    val = val.astype(np.float32)
elif val.dtype.kind in ('i', 'u'):
    val = val.astype(np.int32)

# DataLoader yields Batch namedtuple of np.ndarray
# Transfer to device via jax.device_put(batch) or shard_batch(batch, mesh)
```

Key principle: **zero `import torch` in data pipeline**. Use duck typing for tensor conversion.

## Vision Preprocessing

### PyTorch (torchvision)

```python
from torchvision.transforms import Resize, Normalize, ToTensor

transform = Compose([
    Resize((height, width)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_tensor = transform(pil_image)
```

### JAX (PIL only)

```python
from PIL import Image
import numpy as np

image = Image.open(path).convert("RGB")
image = image.resize((width, height), Image.BICUBIC)
image_array = np.array(image, dtype=np.float32) / 255.0
# Normalize
image_array = (image_array - mean) / std
# (H, W, C) → (C, H, W) if needed by model
image_array = np.transpose(image_array, (2, 0, 1))
```

### Video decoding

```python
# PyTorch: torchvision.io or decord
# JAX: decord only (no torchvision dependency)
import decord
decord.bridge.set_bridge("numpy")  # NOT "torch"
reader = decord.VideoReader(path)
frames = reader.get_batch(indices).asnumpy()  # np.ndarray directly
```

## DataCollator Pattern

### PyTorch

```python
class DataCollator:
    def __call__(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [inst["input_ids"] for inst in instances],
            batch_first=True, padding_value=pad_token_id
        )
        return {"input_ids": input_ids, "labels": labels, ...}
```

### JAX/NumPy

```python
class DataCollator:
    def __call__(self, instances):
        # Custom pad_sequence replacement
        input_ids_list = [inst["input_ids"] for inst in instances]
        max_len = max(x.shape[0] for x in input_ids_list)
        padded = np.full((len(instances), max_len), pad_token_id, dtype=np.int32)
        for i, ids in enumerate(input_ids_list):
            padded[i, :len(ids)] = ids
        return Batch(input_ids=padded, labels=labels, ...)
```

Return a `NamedTuple` (not dict) for `jax.device_put()` compatibility:

```python
class Batch(NamedTuple):
    input_ids: np.ndarray
    labels: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray
    pixel_values: Optional[np.ndarray]
    image_grid_thw: Optional[np.ndarray]
    # ... other fields
```

## Fixed-Shape Padding

### Problem

XLA compiles a separate graph for each unique tensor shape combination. If batch tensors have different shapes each step, XLA recompiles every step (~60-100s instead of ~0.3s).

Dynamic shapes come from:
- **Text**: sequences padded to batch-max length (varies per batch)
- **Vision**: different numbers of images/patches per batch

### Solution: pad everything to fixed maximums

#### Text tensors

```python
# DON'T use tokenizer.model_max_length (e.g., 262144 for Qwen3-VL)
# DO use training_args.model_max_length (e.g., 1024)
max_len = training_args.model_max_length

input_ids = np.full((B, max_len), pad_token_id, dtype=np.int32)
labels = np.full((B, max_len), -100, dtype=np.int32)
attention_mask = np.zeros((B, max_len), dtype=np.int32)
position_ids = np.zeros((3, B, max_len), dtype=np.int32)

for i, inst in enumerate(instances):
    seq_len = len(inst["input_ids"])
    input_ids[i, :seq_len] = inst["input_ids"]
    labels[i, :seq_len] = inst["labels"]
    attention_mask[i, :seq_len] = 1
    position_ids[:, i, :seq_len] = inst["position_ids"]
```

#### Vision tensors

Compute maximums from config (no data scanning needed):

```python
patch_area = patch_size ** 2                                    # e.g., 16^2 = 256
max_patches_per_image = max_pixels // patch_area                # e.g., 50176/256 = 196
max_total_patches = batch_size * max_patches_per_image          # e.g., 4*196 = 784
max_num_images = batch_size                                     # assume 1 image per sample

# Pad vision tensors
pixel_values = np.zeros((max_total_patches, C, pH, pW), dtype=np.float32)
image_grid_thw = np.zeros((max_num_images, 3), dtype=np.int32)
image_pos_ids_2d = np.zeros((max_total_patches, 2), dtype=np.int32)
image_cu_seqlens = np.zeros((max_num_images + 1,), dtype=np.int32)
```

#### Why padding is safe

| Component | Safety guarantee |
|-----------|-----------------|
| PatchEmbed3D | Zero padding → ~zero embeddings |
| Block-diagonal mask | Padding tokens form isolated segments |
| PatchMerger | Padding patches merge into padding tokens |
| `_scatter_embeddings` | Only selects non-padding via cumsum+clip |
| Cross-entropy loss | Padding labels = -100, masked out |

### Impact

```
Before (dynamic shapes):  Step 3+ = 60-106s/step, ~50 tokens/s
After  (fixed shapes):    Step 3+ = 0.30s/step,   ~14,000 tokens/s
Speedup: ~250x
```

## CPU vs JIT Boundary

### What runs on CPU (pure NumPy)

- Image/video loading and resizing
- Tokenization
- RoPE position ID computation (`get_rope_index` functions)
- Vision position ID computation (`precompute_vision_position_ids`)
- Data collation and padding
- `cu_seqlens` computation

### What runs inside JIT (JAX)

- Model forward pass
- Loss computation
- Gradient computation
- Optimizer step
- RoPE cos/sin computation from position IDs
- Attention mask construction from `cu_seqlens`

### Why this split matters

- CPU code can use dynamic shapes, Python loops, conditionals freely
- JIT code must have static shapes for efficient XLA compilation
- Precomputing position IDs on CPU avoids dynamic indexing inside JIT

## Gotchas

1. **tokenizer.model_max_length trap**: Qwen3-VL tokenizer default is 262144 (256K). Using this for padding hangs XLA compilation. Always use `training_args.model_max_length`.

2. **model_max_length=8192 OOM**: Attention matrix `(B, num_heads, L, L)` in float32. At L=8192, B=4, heads=16: `4*16*8192*8192*4 bytes = 16 GB`. Use 1024-2048 for fine-tuning.

3. **Vision attention scales with global batch**: Vision model is NOT DP-sharded. All patches from the entire global batch create one attention matrix per chip. `max_pixels` must be controlled.

4. **Batch NamedTuple vs dict**: `jax.device_put` works with both, but NamedTuple is cleaner. Ensure all fields are `np.ndarray` or `None`.

5. **decord bridge**: Must call `decord.bridge.set_bridge("numpy")`, not `"torch"`. Otherwise returns torch tensors.

6. **HF processor output**: May return torch tensors even when you didn't ask. Always run `_ensure_numpy()` on every field.
