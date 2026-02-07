"""Load HuggingFace safetensors checkpoints and convert to Flax parameter dict.

Key mapping rules:
- PyTorch Dense weights are transposed: (out, in) -> Flax (in, out)
- Conv3D weights: PyTorch (out, in, T, H, W) -> JAX (T, H, W, in, out)
- Embedding weights: direct copy
- Norm weights: direct copy
- When LoRA is active, text attention projections get nested under 'base/'
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import Qwen3VLConfig


def load_hf_weights(
    model_path: str,
    config: Qwen3VLConfig,
    lora_rank: int = 0,
    dtype: Any = None,
) -> Dict[str, Any]:
    """Load HuggingFace safetensors and convert to a nested Flax param dict.

    Args:
        model_path: path to HuggingFace model directory containing
            ``*.safetensors`` files.
        config: model configuration.
        lora_rank: when > 0, text attention projections are nested under
            ``base/`` to make room for LoRA parameters.
        dtype: target numpy dtype for parameter values (e.g. ``np.float32``
            or ``jnp.bfloat16`` via ``ml_dtypes.bfloat16``). Defaults to
            ``np.float32``.

    Returns:
        Nested dict mirroring the Flax parameter tree, ready for
        ``model.init`` or ``model.bind``.
    """
    from safetensors import safe_open

    if dtype is None:
        dtype = np.float32

    # Discover safetensors files (handle sharded checkpoints)
    st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not st_files:
        raise FileNotFoundError(
            f"No safetensors files found in {model_path}"
        )

    # Load all tensors
    raw_tensors: Dict[str, np.ndarray] = {}
    for fpath in st_files:
        with safe_open(fpath, framework="numpy") as f:
            for key in f.keys():
                raw_tensors[key] = f.get_tensor(key)

    # Build Flax param dict
    params: Dict[str, Any] = {}
    for pt_key, tensor in raw_tensors.items():
        flax_path = _pytorch_key_to_flax_path(pt_key, lora_rank)
        if flax_path is None:
            continue
        tensor = _maybe_transpose(pt_key, tensor)
        tensor = np.array(tensor, dtype=dtype)
        _set_nested(params, flax_path, tensor)

    return {"params": params}


# ---------------------------------------------------------------------------
# Key mapping
# ---------------------------------------------------------------------------

# Text attention projection names that get LoRA nesting
_TEXT_ATTN_PROJS = {"q_proj", "k_proj", "v_proj", "o_proj"}


def _pytorch_key_to_flax_path(
    key: str,
    lora_rank: int = 0,
) -> Optional[Tuple[str, ...]]:
    """Convert a PyTorch state_dict key to a Flax parameter path tuple.

    Returns None if the key should be skipped.

    Supports two naming conventions:
    - Qwen2.5-VL style: ``model.embed_tokens.weight``, ``model.layers.*``
    - Qwen3-VL style: ``model.language_model.embed_tokens.weight``,
      ``model.language_model.layers.*``
    """
    # Handle lm_head (both styles)
    if key in ("lm_head.weight", "model.language_model.lm_head.weight"):
        return ("lm_head", "kernel")

    # Handle embed_tokens (both styles)
    # Model uses self.param("embed_tokens", ...) so this is a direct leaf.
    if key in ("model.embed_tokens.weight",
               "model.language_model.embed_tokens.weight"):
        return ("embed_tokens",)

    # ---------------------------------------------------------------
    # Vision encoder: model.visual.*
    # ---------------------------------------------------------------
    if key.startswith("model.visual.") or key.startswith("visual."):
        vkey = key.replace("model.visual.", "").replace("visual.", "")
        return _vision_key_to_flax(vkey)

    # ---------------------------------------------------------------
    # Text decoder: model.language_model.layers.*, model.layers.*, etc.
    # ---------------------------------------------------------------
    if key.startswith("model.language_model."):
        tkey = key[len("model.language_model."):]
        return _text_key_to_flax(tkey, lora_rank)

    if key.startswith("model."):
        tkey = key[len("model."):]
        return _text_key_to_flax(tkey, lora_rank)

    return None


def _vision_key_to_flax(vkey: str) -> Optional[Tuple[str, ...]]:
    """Map a vision encoder key (after stripping 'model.visual.') to Flax path."""
    parts = vkey.split(".")

    # patch_embed.proj.weight / bias
    if parts[0] == "patch_embed" and parts[1] == "proj":
        param_name = "kernel" if parts[2] == "weight" else "bias"
        return ("visual", "patch_embed", "proj", param_name)

    # pos_embed.weight -> embedding
    if parts[0] == "pos_embed" and parts[1] == "weight":
        return ("visual", "pos_embed", "embedding")

    # blocks.{i}.{submodule}
    if parts[0] == "blocks":
        idx = parts[1]
        block_name = f"blocks_{idx}"
        sub = parts[2:]
        flax_sub = _vision_block_sub(sub)
        if flax_sub is not None:
            return ("visual", block_name) + flax_sub
        return None

    # merger.{submodule}
    if parts[0] == "merger":
        return ("visual", "merger") + _merger_sub(parts[1:])

    # deepstack_merger_list.{i}.{submodule}
    if parts[0] == "deepstack_merger_list":
        idx = parts[1]
        merger_name = f"deepstack_merger_list_{idx}"
        return ("visual", merger_name) + _merger_sub(parts[2:])

    # ln_post.weight / bias
    if parts[0] == "ln_post":
        param_name = _norm_param(parts[1])
        return ("visual", "ln_post", param_name)

    return None


def _vision_block_sub(parts: list) -> Optional[Tuple[str, ...]]:
    """Map vision block sub-keys."""
    if not parts:
        return None

    # norm1.weight, norm2.weight, norm1.bias, norm2.bias
    if parts[0] in ("norm1", "norm2"):
        param_name = _norm_param(parts[1])
        return (parts[0], param_name)

    # attn.qkv.weight/bias, attn.proj.weight/bias
    if parts[0] == "attn":
        layer = parts[1]  # qkv or proj
        param_name = "kernel" if parts[2] == "weight" else "bias"
        return ("attn", layer, param_name)

    # mlp.linear_fc1.weight/bias, mlp.linear_fc2.weight/bias
    if parts[0] == "mlp":
        layer = parts[1]  # linear_fc1 or linear_fc2
        param_name = "kernel" if parts[2] == "weight" else "bias"
        return ("mlp", layer, param_name)

    return None


def _merger_sub(parts: list) -> Tuple[str, ...]:
    """Map merger sub-keys (norm, linear_fc1, linear_fc2)."""
    if parts[0] == "norm":
        param_name = _norm_param(parts[1])
        return ("norm", param_name)
    # linear_fc1, linear_fc2
    param_name = "kernel" if parts[1] == "weight" else "bias"
    return (parts[0], param_name)


def _text_key_to_flax(
    tkey: str,
    lora_rank: int = 0,
) -> Optional[Tuple[str, ...]]:
    """Map a text decoder key (after stripping 'model.') to Flax path."""
    parts = tkey.split(".")

    # model.norm.weight -> model/norm/weight
    if parts[0] == "norm":
        return ("model", "norm", "weight")

    # layers.{i}.{submodule}
    if parts[0] == "layers":
        idx = parts[1]
        layer_name = f"layers_{idx}"
        sub = parts[2:]
        flax_sub = _text_layer_sub(sub, lora_rank)
        if flax_sub is not None:
            return ("model", layer_name) + flax_sub
        return None

    return None


def _text_layer_sub(
    parts: list,
    lora_rank: int = 0,
) -> Optional[Tuple[str, ...]]:
    """Map text decoder layer sub-keys."""
    if not parts:
        return None

    # input_layernorm.weight, post_attention_layernorm.weight
    if parts[0] in ("input_layernorm", "post_attention_layernorm"):
        return (parts[0], "weight")

    # self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
    if parts[0] == "self_attn":
        proj = parts[1]
        param_name = "kernel" if parts[2] == "weight" else "bias"

        # q_norm, k_norm
        if proj in ("q_norm", "k_norm"):
            return ("self_attn", proj, "weight")

        if proj in _TEXT_ATTN_PROJS:
            if lora_rank > 0:
                return ("self_attn", proj, "base", param_name)
            else:
                return ("self_attn", proj, param_name)

        return None

    # mlp.gate_proj, mlp.up_proj, mlp.down_proj
    if parts[0] == "mlp":
        proj = parts[1]  # gate_proj, up_proj, down_proj
        param_name = "kernel" if parts[2] == "weight" else "bias"
        return ("mlp", proj, param_name)

    return None


# ---------------------------------------------------------------------------
# Transpose logic
# ---------------------------------------------------------------------------

def _maybe_transpose(key: str, tensor: np.ndarray) -> np.ndarray:
    """Transpose weight tensors from PyTorch to Flax convention.

    - Dense (2D): (out, in) -> (in, out)
    - Conv3D (5D): (out, in, T, H, W) -> (T, H, W, in, out)
    - Embedding, norm, bias: no transpose
    """
    # Skip 1-D tensors (bias, norm weight, etc.)
    if tensor.ndim <= 1:
        return tensor

    # Embedding: (vocab, dim) -- no transpose
    if "embed_tokens" in key or "pos_embed" in key:
        return tensor

    # Conv3D weight: (out_ch, in_ch, T, H, W) -> (T, H, W, in_ch, out_ch)
    if tensor.ndim == 5:
        return np.transpose(tensor, (2, 3, 4, 1, 0))

    # Dense weight: (out, in) -> (in, out)
    if tensor.ndim == 2:
        return tensor.T

    return tensor


# ---------------------------------------------------------------------------
# Nested dict helper
# ---------------------------------------------------------------------------

def _set_nested(d: Dict, path: Tuple[str, ...], value: Any) -> None:
    """Set a value in a nested dict given a path tuple."""
    for part in path[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
    d[path[-1]] = value


def _norm_param(name: str) -> str:
    """Map PyTorch norm parameter names to Flax names."""
    if name == "weight":
        return "scale"
    if name == "bias":
        return "bias"
    return name
