"""Export Flax parameters back to HuggingFace safetensors format.

This is the REVERSE of ``weight_loader.py``.  It also handles merging
LoRA parameters (A @ B * alpha/rank) before export.

Key mapping: Flax path tuple -> PyTorch state-dict key.
Transpose:   Flax (in, out) -> PyTorch (out, in), etc.

No ``torch`` or ``jax`` imports -- pure numpy + standard library.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Qwen3VLConfig


# ---------------------------------------------------------------------------
# LoRA merge
# ---------------------------------------------------------------------------

def merge_lora_params(
    params: dict,
    lora_alpha: float = 1.0,
    lora_rank: int = 1,
) -> dict:
    """Merge LoRA A/B into base weights in-place (conceptually).

    Looks for the pattern ``{...}/base/kernel``, ``{...}/lora_A/kernel``,
    ``{...}/lora_B/kernel`` and produces ``{...}/kernel`` with value
    ``base + (A @ B) * (alpha / rank)``.

    Args:
        params: nested Flax parameter dict (mutated copy returned).
        lora_alpha: LoRA scaling alpha.
        lora_rank: LoRA rank (r).

    Returns:
        New parameter dict with LoRA params merged and removed.
    """
    scale = lora_alpha / max(lora_rank, 1)
    return _merge_node(params, scale)


def _merge_node(node: Any, scale: float) -> Any:
    """Recursively walk the param tree and merge LoRA sub-dicts."""
    if not isinstance(node, dict):
        return node

    # Detect a dict that has {base, lora_A, lora_B} children
    if "base" in node and "lora_A" in node and "lora_B" in node:
        base_kernel = node["base"]["kernel"]
        lora_a = node["lora_A"]["kernel"]  # (in, rank)
        lora_b = node["lora_B"]["kernel"]  # (rank, out)
        merged = base_kernel + (lora_a @ lora_b) * scale
        # Rebuild node without base/lora_A/lora_B, putting merged kernel
        merged_node = {}
        for k, v in node.items():
            if k in ("base", "lora_A", "lora_B"):
                continue
            merged_node[k] = _merge_node(v, scale)
        merged_node["kernel"] = merged
        return merged_node

    # Otherwise recurse into children
    return {k: _merge_node(v, scale) for k, v in node.items()}


# ---------------------------------------------------------------------------
# Flatten nested dict
# ---------------------------------------------------------------------------

def _flatten_params(
    d: dict,
    prefix: Tuple[str, ...] = (),
) -> List[Tuple[Tuple[str, ...], np.ndarray]]:
    """Flatten a nested dict to a list of ``(path_tuple, array)``."""
    items: List[Tuple[Tuple[str, ...], np.ndarray]] = []
    for k, v in d.items():
        path = prefix + (k,)
        if isinstance(v, dict):
            items.extend(_flatten_params(v, path))
        else:
            items.append((path, np.asarray(v)))
    return items


# ---------------------------------------------------------------------------
# Flax path -> PyTorch key
# ---------------------------------------------------------------------------

# Vision norm parameter names: Flax -> PyTorch
_VISION_NORM_MAP = {"scale": "weight", "bias": "bias"}


def _flax_path_to_pytorch_key(path: Tuple[str, ...]) -> Optional[str]:
    """Convert a Flax parameter path tuple to a PyTorch state-dict key.

    Returns None if the path cannot be mapped (should not happen for a
    well-formed tree).
    """
    # --- lm_head ---
    if path[0] == "lm_head":
        # ("lm_head", "kernel") -> "lm_head.weight"
        return "lm_head.weight"

    # --- embed_tokens ---
    if path[0] == "embed_tokens":
        # ("embed_tokens", "embedding") -> "model.embed_tokens.weight"
        return "model.embed_tokens.weight"

    # --- Vision encoder ---
    if path[0] == "visual":
        return _visual_path_to_key(path[1:])

    # --- Text decoder (model.*) ---
    if path[0] == "model":
        return _text_path_to_key(path)

    return None


def _visual_path_to_key(path: Tuple[str, ...]) -> Optional[str]:
    """Map vision sub-path (after 'visual') to PyTorch key."""
    if not path:
        return None

    first = path[0]

    # patch_embed.proj.{weight|bias}
    if first == "patch_embed":
        # ("patch_embed", "proj", "kernel"|"bias")
        param = "weight" if path[2] == "kernel" else path[2]
        return f"model.visual.patch_embed.proj.{param}"

    # pos_embed.weight
    if first == "pos_embed":
        # ("pos_embed", "embedding") -> "model.visual.pos_embed.weight"
        return "model.visual.pos_embed.weight"

    # blocks_N.{sub}
    if first.startswith("blocks_"):
        idx = first.split("_", 1)[1]
        sub_key = _visual_block_sub_key(path[1:])
        if sub_key is not None:
            return f"model.visual.blocks.{idx}.{sub_key}"
        return None

    # merger.{sub}
    if first == "merger":
        sub_key = _merger_sub_key(path[1:])
        return f"model.visual.merger.{sub_key}"

    # deepstack_merger_list_N.{sub}
    if first.startswith("deepstack_merger_list_"):
        idx = first.split("deepstack_merger_list_", 1)[1]
        sub_key = _merger_sub_key(path[1:])
        return f"model.visual.deepstack_merger_list.{idx}.{sub_key}"

    # ln_post.{weight|bias}
    if first == "ln_post":
        param = path[1]
        pt_param = _VISION_NORM_MAP.get(param, param)
        return f"model.visual.ln_post.{pt_param}"

    return None


def _visual_block_sub_key(path: Tuple[str, ...]) -> Optional[str]:
    """Convert vision block sub-path to dotted PyTorch key fragment."""
    if not path:
        return None

    first = path[0]

    # norm1 / norm2
    if first in ("norm1", "norm2"):
        param = path[1]
        pt_param = _VISION_NORM_MAP.get(param, param)
        return f"{first}.{pt_param}"

    # attn.{qkv|proj}.{weight|bias}
    if first == "attn":
        layer = path[1]
        param = "weight" if path[2] == "kernel" else path[2]
        return f"attn.{layer}.{param}"

    # mlp.{linear_fc1|linear_fc2}.{weight|bias}
    if first == "mlp":
        layer = path[1]
        param = "weight" if path[2] == "kernel" else path[2]
        return f"mlp.{layer}.{param}"

    return None


def _merger_sub_key(path: Tuple[str, ...]) -> str:
    """Convert merger sub-path to dotted PyTorch key fragment."""
    if path[0] == "norm":
        param = path[1]
        pt_param = _VISION_NORM_MAP.get(param, param)
        return f"norm.{pt_param}"
    # linear_fc1, linear_fc2, etc.
    layer = path[0]
    param = "weight" if path[1] == "kernel" else path[1]
    return f"{layer}.{param}"


def _text_path_to_key(path: Tuple[str, ...]) -> Optional[str]:
    """Convert text decoder Flax path (starting with 'model') to PyTorch key.

    Examples:
        ("model", "norm", "weight") -> "model.norm.weight"
        ("model", "layers_0", "self_attn", "q_proj", "kernel")
            -> "model.layers.0.self_attn.q_proj.weight"
    """
    # ("model", "norm", "weight")
    if len(path) == 3 and path[1] == "norm":
        return "model.norm.weight"

    # ("model", "layers_N", ...)
    if len(path) >= 3 and path[1].startswith("layers_"):
        idx = path[1].split("_", 1)[1]
        sub = path[2:]
        sub_key = _text_layer_sub_key(sub)
        if sub_key is not None:
            return f"model.layers.{idx}.{sub_key}"

    return None


def _text_layer_sub_key(path: Tuple[str, ...]) -> Optional[str]:
    """Convert text layer sub-path to dotted PyTorch key fragment."""
    if not path:
        return None

    first = path[0]

    # input_layernorm / post_attention_layernorm
    if first in ("input_layernorm", "post_attention_layernorm"):
        # ("input_layernorm", "weight") -> "input_layernorm.weight"
        return f"{first}.{path[1]}"

    # self_attn.{q_proj,k_proj,v_proj,o_proj}.{weight}
    # self_attn.{q_norm,k_norm}.weight
    if first == "self_attn":
        proj = path[1]
        if proj in ("q_norm", "k_norm"):
            return f"self_attn.{proj}.weight"
        # Regular projection
        param = "weight" if path[2] == "kernel" else path[2]
        return f"self_attn.{proj}.{param}"

    # mlp.{gate_proj,up_proj,down_proj}.{weight}
    if first == "mlp":
        proj = path[1]
        param = "weight" if path[2] == "kernel" else path[2]
        return f"mlp.{proj}.{param}"

    return None


# ---------------------------------------------------------------------------
# Reverse transpose
# ---------------------------------------------------------------------------

def _reverse_transpose(key: str, tensor: np.ndarray) -> np.ndarray:
    """Transpose weight tensors from Flax to PyTorch convention.

    REVERSE of ``weight_loader._maybe_transpose``:
    - Dense 2D ``(in, out)`` -> ``(out, in)``
    - Conv3D 5D ``(T, H, W, in, out)`` -> ``(out, in, T, H, W)``
    - Embedding, norm, bias: no transpose
    """
    # 1D: bias, norm weight -- no transpose
    if tensor.ndim <= 1:
        return tensor

    # Embedding: no transpose
    if "embed_tokens" in key or "pos_embed" in key:
        return tensor

    # Conv3D: (T, H, W, in, out) -> (out, in, T, H, W)
    if tensor.ndim == 5:
        return np.transpose(tensor, (4, 3, 0, 1, 2))

    # Dense 2D: (in, out) -> (out, in)
    if tensor.ndim == 2:
        return tensor.T

    return tensor


# ---------------------------------------------------------------------------
# Sharded save
# ---------------------------------------------------------------------------

def _save_sharded(
    tensors: Dict[str, np.ndarray],
    output_dir: str,
    max_shard_size: int,
) -> None:
    """Save tensors to one or more safetensors shards, plus an index JSON.

    If the total size fits in a single shard, saves as
    ``model.safetensors``.  Otherwise saves as
    ``model-00001-of-NNNNN.safetensors`` etc.
    """
    from safetensors.numpy import save_file

    os.makedirs(output_dir, exist_ok=True)

    # Compute total size
    total_bytes = sum(t.nbytes for t in tensors.values())

    if total_bytes <= max_shard_size:
        # Single file
        path = os.path.join(output_dir, "model.safetensors")
        save_file(tensors, path)
        return

    # Shard
    shards: List[Dict[str, np.ndarray]] = []
    current_shard: Dict[str, np.ndarray] = {}
    current_size = 0

    for key in sorted(tensors.keys()):
        t = tensors[key]
        if current_size + t.nbytes > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = t
        current_size += t.nbytes

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map: Dict[str, str] = {}

    for i, shard in enumerate(shards, 1):
        filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        path = os.path.join(output_dir, filename)
        save_file(shard, path)
        for key in shard:
            weight_map[key] = filename

    # Write index
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_hf_weights(
    params: dict,
    output_dir: str,
    config,  # Qwen3VLConfig
    lora_rank: int = 0,
    lora_alpha: float = 1.0,
    max_shard_size: int = 5 * 1024**3,  # 5 GB per shard
) -> None:
    """Export Flax parameters to HuggingFace safetensors format.

    Steps:
        1. If LoRA is active, merge ``base + A @ B * alpha/rank``.
        2. Flatten the nested param dict.
        3. Convert each Flax path to a PyTorch state-dict key.
        4. Apply reverse transpose.
        5. Save as safetensors (sharded if needed).

    Args:
        params: nested Flax parameter dict (e.g. ``state.params``).
        output_dir: directory to write safetensors files.
        config: ``Qwen3VLConfig`` (unused currently but kept for
            forward-compatibility).
        lora_rank: LoRA rank; 0 means no LoRA.
        lora_alpha: LoRA scaling alpha.
        max_shard_size: maximum bytes per shard file.
    """
    # 1. Merge LoRA if needed
    if lora_rank > 0:
        params = merge_lora_params(params, lora_alpha=lora_alpha, lora_rank=lora_rank)

    # 2. Flatten
    flat = _flatten_params(params)

    # 3 & 4. Convert keys and transpose
    tensors: Dict[str, np.ndarray] = {}
    for flax_path, array in flat:
        pt_key = _flax_path_to_pytorch_key(flax_path)
        if pt_key is None:
            # Skip unknown paths
            continue
        array = np.asarray(array, dtype=np.float32) if array.dtype != np.float32 else array
        array = _reverse_transpose(pt_key, array)
        # Ensure contiguous C-order for safetensors
        array = np.ascontiguousarray(array)
        tensors[pt_key] = array

    # 5. Save
    _save_sharded(tensors, output_dir, max_shard_size)
