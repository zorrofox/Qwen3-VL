"""Training state for JAX Qwen3-VL fine-tuning.

Extends ``flax.training.train_state.TrainState`` to hold the model apply
function, parameters, and optimizer state in a single pytree-compatible
dataclass.
"""

from __future__ import annotations

import flax.linen as nn
from flax.training import train_state
import optax


class TrainState(train_state.TrainState):
    """Extended TrainState for Qwen3-VL.

    Inherits ``apply_fn``, ``params``, ``tx``, ``opt_state``, and ``step``
    from the Flax base class.  The ``apply_gradients`` method applies
    optimizer updates and increments ``step``.
    """
    pass


def create_train_state(
    model: nn.Module,
    params: dict,
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Create a TrainState from model, params, and optimizer.

    Args:
        model: Flax module (used for ``apply_fn``).
        params: nested parameter dict.
        optimizer: an ``optax.GradientTransformation`` (e.g. from
            ``create_optimizer``).

    Returns:
        A ``TrainState`` instance ready for training.
    """
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
