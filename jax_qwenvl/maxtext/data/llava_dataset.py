"""HuggingFace Dataset Builder for LLaVA-Instruct-150K.

读取 GCS 上的原始 LLaVA JSON + COCO 图片，转为 MaxText SFT 期望的
(query, label, images) 格式，无需预转换存储。

MaxText SFT config 引用方式：
    hf_path: "jax_qwenvl/maxtext/data/llava_dataset.py"

运行时通过 MaxText 的 config 传入路径：
    llava_json_path: gs://bucket/llava_instruct_150k.json
    llava_image_dir:  gs://bucket/train2017
"""

import json
import os
from pathlib import Path

import datasets
from datasets import GeneratorBasedBuilder, SplitGenerator, Version
from datasets import Features, Value, Image, Sequence, Split

# LLaVA image placeholder，与 Qwen3-VL / MaxText Qwen3-Omni processor 一致
IMAGE_PLACEHOLDER = "<|image|>"

_DESCRIPTION = "LLaVA-Instruct-150K formatted for MaxText multimodal SFT."
_VERSION = Version("1.0.0")


class LLaVADataset(GeneratorBasedBuilder):
    """HuggingFace dataset builder that reads LLaVA JSON + COCO images."""

    VERSION = _VERSION
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=_VERSION, description=_DESCRIPTION)
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=Features({
                "query":  Value("string"),   # prompt，包含 image placeholder
                "label":  Value("string"),   # completion（GPT 回答）
                "images": Sequence(Image()), # 对应图片列表
            }),
        )

    def _split_generators(self, dl_manager):
        # 路径从 MaxText config 传入，通过 dl_manager.manual_dir 或环境变量
        json_path  = os.environ.get("LLAVA_JSON_PATH",  "")
        image_dir  = os.environ.get("LLAVA_IMAGE_DIR",  "")

        if not json_path:
            raise ValueError(
                "Set LLAVA_JSON_PATH env var to the LLaVA JSON file path "
                "(e.g. gs://bucket/llava_instruct_150k.json)"
            )

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"json_path": json_path, "image_dir": image_dir, "split": "train"},
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={"json_path": json_path, "image_dir": image_dir, "split": "validation"},
            ),
        ]

    def _generate_examples(self, json_path, image_dir, split):
        """把 LLaVA 多轮对话展开为 (query, label) 对。"""
        data = _load_json(json_path)

        # 简单按 90/10 划分 train/validation（LLaVA 原始无官方 val split）
        n = len(data)
        if split == "train":
            items = data[:int(n * 0.9)]
        else:
            items = data[int(n * 0.9):]

        idx = 0
        for item in items:
            image_file = item.get("image", "")
            conversations = item.get("conversations", [])

            # 把对话列表转为 (human, gpt) 配对
            pairs = _extract_pairs(conversations)

            for turn_idx, (human_text, gpt_text) in enumerate(pairs):
                # 只在第一轮插入图片 placeholder
                if turn_idx == 0 and image_file:
                    query = f"{IMAGE_PLACEHOLDER}\n{_strip_image_tag(human_text)}"
                    image_path = _build_image_path(image_dir, image_file)
                    images = [{"path": image_path}]
                else:
                    # 后续轮：拼接历史上下文
                    history = _build_history(pairs, turn_idx)
                    query = history + _strip_image_tag(human_text)
                    images = []

                yield idx, {
                    "query":  query,
                    "label":  gpt_text,
                    "images": images,
                }
                idx += 1


# ── 工具函数 ────────────────────────────────────────────────────────────────

def _load_json(path: str) -> list:
    """支持 GCS 路径（gs://）和本地路径。"""
    if path.startswith("gs://"):
        from google.cloud import storage
        bucket_name, blob_path = path[5:].split("/", 1)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)
        content = blob.download_as_text()
        return json.loads(content)
    else:
        with open(path) as f:
            return json.load(f)


def _build_image_path(image_dir: str, image_file: str) -> str:
    """拼接图片路径，支持 GCS 和本地。"""
    if image_dir:
        return f"{image_dir.rstrip('/')}/{image_file}"
    return image_file


def _strip_image_tag(text: str) -> str:
    """去掉 LLaVA 原始 <image> 标签（MaxText 用 <|image|> 替代）。"""
    return text.replace("<image>\n", "").replace("<image>", "").strip()


def _extract_pairs(conversations: list) -> list[tuple[str, str]]:
    """从对话列表提取 (human, gpt) 配对。"""
    pairs = []
    i = 0
    while i + 1 < len(conversations):
        h = conversations[i]
        g = conversations[i + 1]
        if h.get("from") == "human" and g.get("from") == "gpt":
            pairs.append((h.get("value", ""), g.get("value", "")))
        i += 2
    return pairs


def _build_history(pairs: list[tuple[str, str]], current_turn: int) -> str:
    """拼接前几轮对话作为上下文。"""
    parts = []
    for h, g in pairs[:current_turn]:
        parts.append(f"User: {_strip_image_tag(h)}\nAssistant: {g}\n")
    return "".join(parts) + "User: "
