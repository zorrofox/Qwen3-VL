"""Configuration dataclasses for Qwen3-VL, mirroring HuggingFace defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Qwen3VLVisionConfig:
    """Vision encoder configuration.

    Defaults match the HuggingFace Qwen3-VL config.
    """

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    rope_theta: float = 10000.0
    initializer_range: float = 0.02
    deepstack_visual_indexes: List[int] = field(
        default_factory=lambda: [8, 16, 24]
    )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass
class Qwen3VLTextConfig:
    """Text decoder configuration.

    Defaults match the HuggingFace Qwen3-VL config.
    """

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: Optional[int] = None
    rope_theta: float = 1000000.0
    mrope_section: Tuple[int, ...] = (24, 20, 20)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass
class Qwen3VLConfig:
    """Top-level configuration combining vision and text configs."""

    vision_config: Qwen3VLVisionConfig = field(
        default_factory=Qwen3VLVisionConfig
    )
    text_config: Qwen3VLTextConfig = field(default_factory=Qwen3VLTextConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen3VLConfig":
        """Load configuration from a HuggingFace model directory."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            raw = json.load(f)

        # Parse vision config
        v_raw = raw.get("vision_config", {})
        vision_cfg = Qwen3VLVisionConfig(
            depth=v_raw.get("depth", 27),
            hidden_size=v_raw.get("hidden_size", 1152),
            hidden_act=v_raw.get("hidden_act", "gelu_pytorch_tanh"),
            intermediate_size=v_raw.get("intermediate_size", 4304),
            num_heads=v_raw.get("num_heads", v_raw.get("num_attention_heads", 16)),
            in_channels=v_raw.get("in_channels", 3),
            patch_size=v_raw.get("patch_size", 16),
            spatial_merge_size=v_raw.get("spatial_merge_size", 2),
            temporal_patch_size=v_raw.get("temporal_patch_size", 2),
            out_hidden_size=v_raw.get("out_hidden_size", 3584),
            num_position_embeddings=v_raw.get("num_position_embeddings", 2304),
            rope_theta=v_raw.get("rope_theta", 10000.0),
            initializer_range=v_raw.get("initializer_range", 0.02),
            deepstack_visual_indexes=v_raw.get(
                "deepstack_visual_indexes", [8, 16, 24]
            ),
        )

        # Parse text config
        t_raw = raw.get("text_config", {})
        # mrope_section may live in rope_parameters or rope_scaling
        rope_params = t_raw.get("rope_parameters", t_raw.get("rope_scaling", {}))
        if rope_params is None:
            rope_params = {}
        mrope_section = tuple(
            rope_params.get("mrope_section", [24, 20, 20])
        )
        rope_theta = t_raw.get(
            "rope_theta",
            rope_params.get("rope_theta", 1000000.0),
        )
        text_cfg = Qwen3VLTextConfig(
            vocab_size=t_raw.get("vocab_size", 151936),
            hidden_size=t_raw.get("hidden_size", 4096),
            intermediate_size=t_raw.get("intermediate_size", 22016),
            num_hidden_layers=t_raw.get("num_hidden_layers", 32),
            num_attention_heads=t_raw.get("num_attention_heads", 32),
            num_key_value_heads=t_raw.get(
                "num_key_value_heads",
                t_raw.get("num_attention_heads", 32),
            ),
            head_dim=t_raw.get("head_dim", 128),
            hidden_act=t_raw.get("hidden_act", "silu"),
            max_position_embeddings=t_raw.get("max_position_embeddings", 128000),
            initializer_range=t_raw.get("initializer_range", 0.02),
            rms_norm_eps=t_raw.get("rms_norm_eps", 1e-6),
            use_cache=t_raw.get("use_cache", True),
            attention_bias=t_raw.get("attention_bias", False),
            attention_dropout=t_raw.get("attention_dropout", 0.0),
            pad_token_id=t_raw.get("pad_token_id", None),
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )

        return cls(
            vision_config=vision_cfg,
            text_config=text_cfg,
            image_token_id=raw.get("image_token_id", 151655),
            video_token_id=raw.get("video_token_id", 151656),
            vision_start_token_id=raw.get("vision_start_token_id", 151652),
            vision_end_token_id=raw.get("vision_end_token_id", 151653),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
        )
