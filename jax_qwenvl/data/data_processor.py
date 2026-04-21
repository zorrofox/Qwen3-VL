"""Data processing pipeline for JAX Qwen-VL fine-tuning (pure NumPy, no PyTorch)."""

import json
import random
import logging
import re
import time
import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import PIL.Image

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3
from ..types import Batch
from ..model.vit import precompute_vision_position_ids, precompute_vision_cu_seqlens

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"Updated image_processor max_pixels to {data_args.max_pixels}")

    # Note: do NOT override ip.size["longest_edge"] / ip.size["shortest_edge"] here.
    # Qwen3-VL image processor uses max_pixels/min_pixels as area constraints.
    # size["longest_edge"] is an edge-length constraint (pixels, not area) and
    # setting it to max_pixels (an area value) causes images to bypass resizing
    # entirely (COCO images have max edge ~640px << max_pixels=50176).

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
        rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"Updated video_processor fps to {data_args.video_fps}")

        # Note: do NOT override vp.size["longest_edge"] / vp.size["shortest_edge"].
        # Same reason as image processor: video_max_pixels is area, not edge length.

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path, max_pixels: int = None) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with pre-loaded PIL images
    # (passing path strings can fail in transformers 5.x's load_image)
    image_pool = []
    for img in images:
        img_path = _make_abs_paths(base_path, img)
        pil_img = PIL.Image.open(img_path).convert("RGB")
        # Pre-resize to max_pixels area limit before passing to processor.
        # apply_chat_template in transformers 5.x does not reliably respect
        # image_processor.max_pixels, so we enforce the constraint here.
        if max_pixels is not None:
            w, h = pil_img.size
            if w * h > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
                pil_img = pil_img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    PIL.Image.LANCZOS,
                )
        image_pool.append({"type": "image", "image": pil_img})
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def _ensure_numpy(val):
    """Convert torch tensors or lists to numpy arrays."""
    if hasattr(val, 'numpy'):  # torch tensor
        return val.numpy()
    if isinstance(val, list):
        return np.array(val)
    return val


def preprocess_qwen_visual(
    sources,
    processor,
    max_pixels: int = None,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path, max_pixels=max_pixels)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    # Convert torch tensor / list to numpy
    if hasattr(input_ids, 'numpy'):
        input_ids = input_ids.numpy()
    if isinstance(input_ids, list):
        input_ids = np.array(input_ids, dtype=np.int32).reshape(1, -1)
    elif input_ids.ndim == 1:
        input_ids = input_ids.reshape(1, -1)
    input_ids = np.asarray(input_ids, dtype=np.int32)

    labels = np.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids

    # Ensure pixel_values and grid tensors are numpy arrays (HF processor may
    # return torch tensors even with return_tensors="np")
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if key in full_result:
            val = _ensure_numpy(full_result[key])
            # Enforce TPU-friendly dtypes
            if val.dtype.kind == 'f':
                val = val.astype(np.float32)
            elif val.dtype.kind in ('i', 'u'):
                val = val.astype(np.int32)
            full_result[key] = val

    return full_result


class LazySupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, np.ndarray]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, np.ndarray]:
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            max_pixels=getattr(self.data_args, "max_pixels", None),
        )

        seq_len = data_dict["input_ids"].shape[1]

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=np.concatenate(grid_thw, axis=0) if grid_thw else None,
            video_grid_thw=(
                np.concatenate(video_grid_thw, axis=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, np.ndarray]:

        if isinstance(sources, dict):
            if isinstance(sources, dict):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list_items = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list_items.append(self._get_item(source))

            input_ids = np.concatenate([d["input_ids"] for d in data_list_items], axis=1)
            labels = np.concatenate([d["labels"] for d in data_list_items], axis=1)
            position_ids = np.concatenate([d["position_ids"] for d in data_list_items], axis=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list_items if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list_items):
                new_data_dict.update(
                    {
                        "pixel_values": np.concatenate(
                            [
                                d["pixel_values"]
                                for d in data_list_items
                                if "pixel_values" in d
                            ],
                            axis=0,
                        ),
                        "image_grid_thw": np.concatenate(
                            [
                                d["image_grid_thw"]
                                for d in data_list_items
                                if "image_grid_thw" in d
                            ],
                            axis=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list_items):
                new_data_dict.update(
                    {
                        "pixel_values_videos": np.concatenate(
                            [
                                d["pixel_values_videos"]
                                for d in data_list_items
                                if "pixel_values_videos" in d
                            ],
                            axis=0,
                        ),
                        "video_grid_thw": np.concatenate(
                            [
                                d["video_grid_thw"]
                                for d in data_list_items
                                if "video_grid_thw" in d
                            ],
                            axis=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list, max_length=None):
    """Pad 3D arrays along axis 2 to max length, then concatenate along axis 1."""
    if max_length is None:
        max_length = max(arr.shape[2] for arr in tensor_list)
    padded = []
    for arr in tensor_list:
        pad_width = max_length - arr.shape[2]
        if pad_width > 0:
            padded_arr = np.pad(arr, ((0, 0), (0, 0), (0, pad_width)),
                                mode='constant', constant_values=1)
        else:
            padded_arr = arr[:, :, :max_length]
        padded.append(padded_arr)
    return np.concatenate(padded, axis=1)


def _pad_sequence(arrays, padding_value, max_length=None):
    """Pad list of 1D arrays to same length and stack into 2D array."""
    if max_length is None:
        max_length = max(a.shape[0] for a in arrays)
    result = np.full((len(arrays), max_length), padding_value, dtype=arrays[0].dtype)
    for i, arr in enumerate(arrays):
        length = min(arr.shape[0], max_length)
        result[i, :length] = arr[:length]
    return result


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    spatial_merge_size: int = 2
    max_total_patches: int = 0    # 0 = no padding (backward compatible)
    max_num_images: int = 0       # 0 = no padding
    model_max_length: int = 0     # 0 = use batch max (no fixed padding)

    def _pad_vision_inputs(self, pixel_values, grid_thw, pos_ids_2d, pos_ids_1d, cu_seqlens):
        """Pad vision tensors to fixed shapes to avoid JIT recompilation."""
        if self.max_total_patches <= 0:
            return pixel_values, grid_thw, pos_ids_2d, pos_ids_1d, cu_seqlens

        actual_N = pixel_values.shape[0]
        pad_N = self.max_total_patches - actual_N
        if pad_N < 0:
            raise ValueError(f"Actual patches {actual_N} > max {self.max_total_patches}")

        # 1. Pad pixel_values: (actual_N, ...) -> (max_N, ...)
        if pad_N > 0:
            pixel_values = np.pad(pixel_values,
                [(0, pad_N)] + [(0, 0)] * (pixel_values.ndim - 1))

        # 2. Pad position IDs
        pos_ids_2d = np.pad(pos_ids_2d, [(0, pad_N), (0, 0)])
        pos_ids_1d = np.pad(pos_ids_1d, [(0, pad_N)])

        # 3. Pad cu_seqlens: add padding segment + pad to fixed length
        if cu_seqlens[-1] < self.max_total_patches:
            cu_seqlens = np.append(cu_seqlens, self.max_total_patches)
        max_cu_len = self.max_num_images + 2
        if len(cu_seqlens) < max_cu_len:
            cu_seqlens = np.pad(cu_seqlens,
                (0, max_cu_len - len(cu_seqlens)),
                constant_values=self.max_total_patches)

        # 4. Pad grid_thw: (actual_imgs, 3) -> (max_num_images, 3)
        if grid_thw.shape[0] < self.max_num_images:
            pad_grid = np.zeros((self.max_num_images - grid_thw.shape[0], 3), dtype=grid_thw.dtype)
            grid_thw = np.concatenate([grid_thw, pad_grid])

        return pixel_values, grid_thw, pos_ids_2d, pos_ids_1d, cu_seqlens

    def __call__(self, instances: Sequence[Dict]) -> Batch:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) if ids.ndim > 1 else ids for ids in input_ids]
        labels = [ids.squeeze(0) if ids.ndim > 1 else ids for ids in labels]
        max_len = self.model_max_length if self.model_max_length > 0 else None
        input_ids = _pad_sequence(
            input_ids, padding_value=self.tokenizer.pad_token_id, max_length=max_len
        )
        labels = _pad_sequence(
            labels, padding_value=IGNORE_INDEX, max_length=max_len
        )
        position_ids = pad_and_cat(position_ids, max_length=max_len)
        attention_mask = (input_ids != self.tokenizer.pad_token_id)

        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        image_pos_ids_2d = None
        image_pos_ids_1d = None
        image_cu_seqlens = None
        if len(images) != 0:
            concat_images = np.concatenate([image for image in images], axis=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = np.concatenate(grid_thw, axis=0)
            # Precompute vision position IDs on host (outside JIT)
            image_pos_ids_2d, image_pos_ids_1d = precompute_vision_position_ids(
                grid_thw, self.spatial_merge_size
            )
            image_cu_seqlens = precompute_vision_cu_seqlens(grid_thw)
            # Pad vision tensors to fixed shapes to avoid XLA recompilation
            concat_images, grid_thw, image_pos_ids_2d, image_pos_ids_1d, image_cu_seqlens = \
                self._pad_vision_inputs(concat_images, grid_thw, image_pos_ids_2d, image_pos_ids_1d, image_cu_seqlens)
        else:
            concat_images = None
            grid_thw = None

        video_pos_ids_2d = None
        video_pos_ids_1d = None
        video_cu_seqlens = None
        if len(videos) != 0:
            concat_videos = np.concatenate([video for video in videos], axis=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = np.concatenate(video_grid_thw, axis=0)
            video_pos_ids_2d, video_pos_ids_1d = precompute_vision_position_ids(
                video_grid_thw, self.spatial_merge_size
            )
            video_cu_seqlens = precompute_vision_cu_seqlens(video_grid_thw)
        else:
            concat_videos = None
            video_grid_thw = None

        return Batch(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=concat_images,
            image_grid_thw=grid_thw,
            pixel_values_videos=concat_videos,
            video_grid_thw=video_grid_thw,
            image_pos_ids_2d=image_pos_ids_2d,
            image_pos_ids_1d=image_pos_ids_1d,
            image_cu_seqlens=image_cu_seqlens,
            video_pos_ids_2d=video_pos_ids_2d,
            video_pos_ids_1d=video_pos_ids_1d,
            video_cu_seqlens=video_cu_seqlens,
        )


@dataclass
class FlattenedDataCollatorForSupervisedDataset(object):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer
    spatial_merge_size: int = 2
    max_total_patches: int = 0
    max_num_images: int = 0
    model_max_length: int = 0

    def _pad_vision_inputs(self, pixel_values, grid_thw, pos_ids_2d, pos_ids_1d, cu_seqlens):
        """Pad vision tensors to fixed shapes to avoid JIT recompilation."""
        if self.max_total_patches <= 0:
            return pixel_values, grid_thw, pos_ids_2d, pos_ids_1d, cu_seqlens

        actual_N = pixel_values.shape[0]
        pad_N = self.max_total_patches - actual_N
        if pad_N < 0:
            raise ValueError(f"Actual patches {actual_N} > max {self.max_total_patches}")

        # 1. Pad pixel_values: (actual_N, ...) -> (max_N, ...)
        if pad_N > 0:
            pixel_values = np.pad(pixel_values,
                [(0, pad_N)] + [(0, 0)] * (pixel_values.ndim - 1))

        # 2. Pad position IDs
        pos_ids_2d = np.pad(pos_ids_2d, [(0, pad_N), (0, 0)])
        pos_ids_1d = np.pad(pos_ids_1d, [(0, pad_N)])

        # 3. Pad cu_seqlens: add padding segment + pad to fixed length
        if cu_seqlens[-1] < self.max_total_patches:
            cu_seqlens = np.append(cu_seqlens, self.max_total_patches)
        max_cu_len = self.max_num_images + 2
        if len(cu_seqlens) < max_cu_len:
            cu_seqlens = np.pad(cu_seqlens,
                (0, max_cu_len - len(cu_seqlens)),
                constant_values=self.max_total_patches)

        # 4. Pad grid_thw: (actual_imgs, 3) -> (max_num_images, 3)
        if grid_thw.shape[0] < self.max_num_images:
            pad_grid = np.zeros((self.max_num_images - grid_thw.shape[0], 3), dtype=grid_thw.dtype)
            grid_thw = np.concatenate([grid_thw, pad_grid])

        return pixel_values, grid_thw, pos_ids_2d, pos_ids_1d, cu_seqlens

    def __call__(self, instances: Sequence[Dict]) -> Batch:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = np.array([0] + attention_mask, dtype=np.int32)
        cumsum_seq_lens = np.cumsum(seq_lens).astype(np.int32)
        input_ids = np.concatenate(input_ids, axis=1)
        labels = np.concatenate(labels, axis=1)
        position_ids = np.concatenate(position_ids, axis=2)

        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        image_pos_ids_2d = None
        image_pos_ids_1d = None
        image_cu_seqlens = None
        if len(images) != 0:
            concat_images = np.concatenate([image for image in images], axis=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = np.concatenate(grid_thw, axis=0)
            image_pos_ids_2d, image_pos_ids_1d = precompute_vision_position_ids(
                grid_thw, self.spatial_merge_size
            )
            image_cu_seqlens = precompute_vision_cu_seqlens(grid_thw)
            # Pad vision tensors to fixed shapes to avoid XLA recompilation
            concat_images, grid_thw, image_pos_ids_2d, image_pos_ids_1d, image_cu_seqlens = \
                self._pad_vision_inputs(concat_images, grid_thw, image_pos_ids_2d, image_pos_ids_1d, image_cu_seqlens)
        else:
            concat_images = None
            grid_thw = None

        video_pos_ids_2d = None
        video_pos_ids_1d = None
        video_cu_seqlens = None
        if len(videos) != 0:
            concat_videos = np.concatenate([video for video in videos], axis=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = np.concatenate(video_grid_thw, axis=0)
            video_pos_ids_2d, video_pos_ids_1d = precompute_vision_position_ids(
                video_grid_thw, self.spatial_merge_size
            )
            video_cu_seqlens = precompute_vision_cu_seqlens(video_grid_thw)
        else:
            concat_videos = None
            video_grid_thw = None

        return Batch(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
            pixel_values=concat_images,
            image_grid_thw=grid_thw,
            pixel_values_videos=concat_videos,
            video_grid_thw=video_grid_thw,
            image_pos_ids_2d=image_pos_ids_2d,
            image_pos_ids_1d=image_pos_ids_1d,
            image_cu_seqlens=image_cu_seqlens,
            video_pos_ids_2d=video_pos_ids_2d,
            video_pos_ids_1d=video_pos_ids_1d,
            video_cu_seqlens=video_cu_seqlens,
        )


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
