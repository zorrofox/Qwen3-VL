"""Optax multi-group optimizer for Qwen3-VL fine-tuning.

Supports component-specific learning rates (vision, projector, LLM) and
per-parameter weight decay rules, mirroring the PyTorch trainer logic from
``qwen-vl-finetune/qwenvl/train/trainer.py``.

Uses ``optax.multi_transform`` with a label tree to assign each parameter
to its optimizer group.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def create_schedule(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> optax.Schedule:
    """Linear warmup followed by cosine decay.

    Args:
        peak_lr: peak learning rate reached at the end of warmup.
        warmup_steps: number of linear warmup steps.
        total_steps: total number of training steps.

    Returns:
        An ``optax`` schedule function.
    """
    warmup = optax.linear_schedule(0.0, peak_lr, warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    decay = optax.cosine_decay_schedule(peak_lr, decay_steps)
    return optax.join_schedules([warmup, decay], [warmup_steps])


# ---------------------------------------------------------------------------
# Optimizer chain builder
# ---------------------------------------------------------------------------

def _make_optimizer(
    lr: float,
    wd: float,
    warmup_steps: int,
    total_steps: int,
    beta1: float,
    beta2: float,
    eps: float,
    max_grad_norm: float,
) -> optax.GradientTransformation:
    """Build an AdamW optimizer chain with gradient clipping and schedule."""
    schedule = create_schedule(lr, warmup_steps, total_steps)
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            schedule,
            b1=beta1,
            b2=beta2,
            eps=eps,
            weight_decay=wd,
        ),
    )


# ---------------------------------------------------------------------------
# Parameter classification helpers
# ---------------------------------------------------------------------------

def _is_decay_param(path_str: str) -> bool:
    """Return True if the parameter should receive weight decay.

    Weight decay is applied to Dense kernel (weight) parameters only.
    The following are excluded from weight decay:
    - bias parameters
    - norm weight / scale parameters (RMSNorm, LayerNorm)
    - embedding parameters (embed_tokens, pos_embed)
    """
    # Bias params
    if path_str.endswith("/bias"):
        return False
    # Norm weights (RMSNorm uses 'weight', LayerNorm uses 'scale')
    if "/norm/" in path_str or "/norm1/" in path_str or "/norm2/" in path_str:
        return False
    if "layernorm" in path_str.lower():
        return False
    if path_str.endswith("/scale") or path_str.endswith("/weight"):
        # Check if this is a norm weight
        parts = path_str.split("/")
        if len(parts) >= 2:
            parent = parts[-2]
            if parent in (
                "norm", "norm1", "norm2", "ln_post",
                "input_layernorm", "post_attention_layernorm",
                "q_norm", "k_norm",
            ):
                return False
    # Embedding params
    if "embed_tokens" in path_str or "pos_embed" in path_str:
        return False
    if path_str.endswith("/embedding"):
        return False
    # Everything else (kernel params) gets weight decay
    return True


def _is_vision_param(path_str: str) -> bool:
    """Return True if the parameter belongs to the vision encoder (but NOT merger)."""
    return "visual" in path_str and "merger" not in path_str


def _is_projector_param(path_str: str) -> bool:
    """Return True if the parameter belongs to the merger/projector."""
    return "merger" in path_str


def _is_lora_param(path_str: str) -> bool:
    """Return True if the parameter is a LoRA parameter (lora_A or lora_B)."""
    return "lora_A" in path_str or "lora_B" in path_str


def _is_trainable(
    path_str: str,
    tune_vision: bool,
    tune_mlp: bool,
    tune_llm: bool,
    lora_enabled: bool,
) -> bool:
    """Determine if a parameter should be trained.

    When LoRA is enabled, only lora_A and lora_B parameters are trainable.
    Otherwise, trainability is determined by the tune_* flags:
    - tune_vision: vision encoder params (excluding merger)
    - tune_mlp: merger/projector params
    - tune_llm: language model params (model.layers, model.norm, embed_tokens, lm_head)
    """
    if lora_enabled:
        return _is_lora_param(path_str)

    if _is_vision_param(path_str):
        return tune_vision
    if _is_projector_param(path_str):
        return tune_mlp
    # Everything else is LLM
    return tune_llm


def _classify_param(
    path_str: str,
    tune_vision: bool,
    tune_mlp: bool,
    tune_llm: bool,
    lora_enabled: bool,
    vision_lr: Optional[float],
    projector_lr: Optional[float],
) -> str:
    """Classify a parameter into its optimizer group label.

    Returns a string label used as a key in the ``optax.multi_transform``
    transforms dict.

    Label convention:
        ``"frozen"`` -- parameter is not trained (set_to_zero)
        ``"llm_decay"`` / ``"llm_nodecay"`` -- LLM params with/without weight decay
        ``"vision_decay"`` / ``"vision_nodecay"`` -- Vision params (separate lr)
        ``"proj_decay"`` / ``"proj_nodecay"`` -- Projector params (separate lr)
        ``"default_decay"`` / ``"default_nodecay"`` -- When no separate lr is set
    """
    trainable = _is_trainable(
        path_str, tune_vision, tune_mlp, tune_llm, lora_enabled
    )
    if not trainable:
        return "frozen"

    decay = _is_decay_param(path_str)
    is_vision = _is_vision_param(path_str)
    is_proj = _is_projector_param(path_str)

    # Case 1: both vision_lr AND projector_lr set -- 6+1 groups
    if projector_lr is not None and vision_lr is not None:
        if is_vision:
            return "vision_decay" if decay else "vision_nodecay"
        elif is_proj:
            return "proj_decay" if decay else "proj_nodecay"
        else:
            return "llm_decay" if decay else "llm_nodecay"

    # Case 2: only projector_lr set -- 4+1 groups
    # LLM + Vision grouped together, Projector separate
    if projector_lr is not None:
        if is_proj:
            return "proj_decay" if decay else "proj_nodecay"
        else:
            return "default_decay" if decay else "default_nodecay"

    # Case 3: neither set -- 2+1 groups
    # All trainable grouped by decay/no_decay
    return "default_decay" if decay else "default_nodecay"


# ---------------------------------------------------------------------------
# Path serialization
# ---------------------------------------------------------------------------

def _path_to_str(path) -> str:
    """Convert a JAX key path to a slash-separated string for classification."""
    parts = []
    for key in path:
        if hasattr(key, "key"):
            parts.append(str(key.key))
        elif isinstance(key, str):
            parts.append(key)
        else:
            parts.append(str(key))
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Main optimizer factory
# ---------------------------------------------------------------------------

def create_optimizer(
    params: dict,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
    max_grad_norm: float = 1.0,
    vision_lr: Optional[float] = None,
    projector_lr: Optional[float] = None,
    tune_vision: bool = False,
    tune_mlp: bool = False,
    tune_llm: bool = False,
    lora_enabled: bool = False,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[optax.GradientTransformation, dict]:
    """Create an optimizer with component-specific learning rates.

    This mirrors the 6-group optimizer logic from the PyTorch trainer:
    - When both ``vision_lr`` and ``projector_lr`` are set: 6 groups + frozen
    - When only ``projector_lr`` is set: 4 groups + frozen
    - When neither is set: 2 groups + frozen

    Args:
        params: nested parameter dict (the Flax param tree).
        learning_rate: base learning rate for LLM parameters.
        weight_decay: weight decay coefficient.
        warmup_steps: number of warmup steps for the LR schedule.
        total_steps: total number of training steps.
        max_grad_norm: maximum gradient norm for clipping.
        vision_lr: optional separate LR for vision encoder parameters.
        projector_lr: optional separate LR for merger/projector parameters.
        tune_vision: whether to train vision encoder parameters.
        tune_mlp: whether to train merger/projector parameters.
        tune_llm: whether to train LLM parameters.
        lora_enabled: whether LoRA is active (only lora_A/B are trainable).
        beta1: Adam beta1.
        beta2: Adam beta2.
        eps: Adam epsilon.

    Returns:
        ``(optimizer, label_tree)`` where ``label_tree`` maps each parameter
        to its group label string.
    """
    # Common optimizer kwargs
    opt_kwargs = dict(
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        max_grad_norm=max_grad_norm,
    )

    # Build the transforms dict based on which LR overrides are active
    transforms = {
        "frozen": optax.set_to_zero(),
    }

    if projector_lr is not None and vision_lr is not None:
        # Case 1: 6+1 groups
        transforms["llm_decay"] = _make_optimizer(
            lr=learning_rate, wd=weight_decay, **opt_kwargs
        )
        transforms["llm_nodecay"] = _make_optimizer(
            lr=learning_rate, wd=0.0, **opt_kwargs
        )
        transforms["vision_decay"] = _make_optimizer(
            lr=vision_lr, wd=weight_decay, **opt_kwargs
        )
        transforms["vision_nodecay"] = _make_optimizer(
            lr=vision_lr, wd=0.0, **opt_kwargs
        )
        transforms["proj_decay"] = _make_optimizer(
            lr=projector_lr, wd=weight_decay, **opt_kwargs
        )
        transforms["proj_nodecay"] = _make_optimizer(
            lr=projector_lr, wd=0.0, **opt_kwargs
        )
    elif projector_lr is not None:
        # Case 2: 4+1 groups (LLM+Vision share base lr)
        transforms["default_decay"] = _make_optimizer(
            lr=learning_rate, wd=weight_decay, **opt_kwargs
        )
        transforms["default_nodecay"] = _make_optimizer(
            lr=learning_rate, wd=0.0, **opt_kwargs
        )
        transforms["proj_decay"] = _make_optimizer(
            lr=projector_lr, wd=weight_decay, **opt_kwargs
        )
        transforms["proj_nodecay"] = _make_optimizer(
            lr=projector_lr, wd=0.0, **opt_kwargs
        )
    else:
        # Case 3: 2+1 groups
        transforms["default_decay"] = _make_optimizer(
            lr=learning_rate, wd=weight_decay, **opt_kwargs
        )
        transforms["default_nodecay"] = _make_optimizer(
            lr=learning_rate, wd=0.0, **opt_kwargs
        )

    # Build label tree using tree_map_with_path
    def _label_fn(path, _leaf):
        path_str = _path_to_str(path)
        return _classify_param(
            path_str,
            tune_vision=tune_vision,
            tune_mlp=tune_mlp,
            tune_llm=tune_llm,
            lora_enabled=lora_enabled,
            vision_lr=vision_lr,
            projector_lr=projector_lr,
        )

    label_tree = jax.tree_util.tree_map_with_path(_label_fn, params)

    optimizer = optax.multi_transform(transforms, label_tree)

    return optimizer, label_tree
