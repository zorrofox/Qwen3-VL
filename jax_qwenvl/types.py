"""Shared type definitions for the JAX Qwen-VL pipeline."""

from typing import NamedTuple, Optional

import numpy as np

IGNORE_INDEX = -100


class Batch(NamedTuple):
    """A collated batch ready for training.

    All arrays are numpy until jax.device_put() is called at the training boundary.
    """

    input_ids: np.ndarray  # (batch, seq_len), int32
    labels: np.ndarray  # (batch, seq_len), int32
    attention_mask: np.ndarray  # (batch, seq_len) bool or (cumsum_len,) int32 for packed
    position_ids: np.ndarray  # (3, batch, seq_len), int32
    pixel_values: Optional[np.ndarray] = None  # (N, C, H, W), float32 -- images
    image_grid_thw: Optional[np.ndarray] = None  # (num_images, 3), int32
    pixel_values_videos: Optional[np.ndarray] = None  # (N, C, H, W), float32 -- video frames
    video_grid_thw: Optional[np.ndarray] = None  # (num_videos, 3), int32


class CausalLMOutput(NamedTuple):
    """Output from the causal LM forward pass."""

    logits: np.ndarray  # (batch, seq_len, vocab_size)
    loss: Optional[np.ndarray] = None  # scalar
