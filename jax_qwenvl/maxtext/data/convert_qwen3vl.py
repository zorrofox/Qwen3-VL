"""Convert Qwen3-VL-8B HuggingFace weights to MaxText Orbax checkpoint.

只跑一次，转换后存到 GCS，之后 MaxText SFT 直接 load_parameters_path 加载。

用法：
  python3 -m jax_qwenvl.maxtext.data.convert_qwen3vl \
    --hf_model_path gs://YOUR_GCS_BUCKET/models/Qwen3-VL-8B-Instruct/qwen3vl-8b \
    --output_path    gs://YOUR_GCS_BUCKET/models/Qwen3-VL-8B-Instruct/maxtext-ckpt \
    --model_size     qwen3vl-8b \
    --dry_run        True   # 先 dry run 确认 key 映射，再正式转换

HF 权重结构：
  model.language_model.*   ← 文本 decoder（36层）
  model.visual.*           ← 视觉 encoder（27层 ViT + merger）
  lm_head.weight

MaxText 权重结构（Orbax PyTree）：
  token_embedder.embedding
  decoder.layers.{i}.*
  decoder.decoder_norm.scale
  decoder.logits_dense.kernel
  vision_encoder.patch_embed.proj.kernel / bias
  vision_encoder.blocks.{i}.*
  vision_encoder.merger.*
"""

import argparse
import glob
import os
import sys
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

# ── 模型参数（Qwen3-VL-8B-Instruct）────────────────────────────────────────
MODEL_PARAMS = {
    "qwen3vl-8b": {
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "head_dim": 128,
        "vocab_size": 151936,
        "num_vit_layers": 27,
        "vit_hidden_size": 1152,
        "vit_num_heads": 16,
        "vit_head_dim": 64,  # 1152 / 16 / 3 (fused qkv)
        "deepstack_indexes": [8, 16, 24],
    }
}

# ── 文本 Decoder 权重映射 ────────────────────────────────────────────────────
# HF key → (MaxText key, 是否需要转置)
# 注意：Qwen3-VL 的文本部分比标准 Qwen3 多了 model.language_model. 前缀

def text_layer_mapping(layer_idx: int) -> dict:
    """单层文本 decoder 的 HF → MaxText 权重名映射。"""
    i = layer_idx
    pfx = f"model.language_model.layers.{i}"
    mt = f"decoder.layers.{i}"
    return {
        # Attention
        f"{pfx}.self_attn.q_proj.weight":    (f"{mt}.self_attention.query.kernel",              True),
        f"{pfx}.self_attn.k_proj.weight":    (f"{mt}.self_attention.key.kernel",                True),
        f"{pfx}.self_attn.v_proj.weight":    (f"{mt}.self_attention.value.kernel",              True),
        f"{pfx}.self_attn.o_proj.weight":    (f"{mt}.self_attention.out.kernel",                True),
        f"{pfx}.self_attn.q_norm.weight":    (f"{mt}.self_attention.query_norm.scale",          False),
        f"{pfx}.self_attn.k_norm.weight":    (f"{mt}.self_attention.key_norm.scale",            False),
        # MLP (SwiGLU: gate, up, down)
        f"{pfx}.mlp.gate_proj.weight":       (f"{mt}.mlp.wi_0.kernel",                         True),
        f"{pfx}.mlp.up_proj.weight":         (f"{mt}.mlp.wi_1.kernel",                         True),
        f"{pfx}.mlp.down_proj.weight":       (f"{mt}.mlp.wo.kernel",                           True),
        # LayerNorm
        f"{pfx}.input_layernorm.weight":     (f"{mt}.pre_self_attention_layer_norm.scale",      False),
        f"{pfx}.post_attention_layernorm.weight": (f"{mt}.post_self_attention_layer_norm.scale", False),
    }

def build_text_mapping(num_layers: int) -> dict:
    """构建完整文本 decoder 映射（embedding + 所有层 + norm + lm_head）。"""
    mapping = {
        "model.language_model.embed_tokens.weight": ("token_embedder.embedding",      False),
        "model.language_model.norm.weight":         ("decoder.decoder_norm.scale",    False),
        "lm_head.weight":                           ("decoder.logits_dense.kernel",   True),
    }
    for i in range(num_layers):
        mapping.update(text_layer_mapping(i))
    return mapping


# ── 视觉 Encoder 权重映射 ──────────────────────────────────────────────────
# HF Qwen3-VL ViT: blocks.{i}.attn.qkv（fused）→ MaxText Attention（split q,k,v）

def vit_layer_mapping(layer_idx: int, num_heads: int, head_dim: int) -> dict:
    """单层 ViT block 映射，qkv fused weight 需要手动拆分。"""
    i = layer_idx
    pfx = f"model.visual.blocks.{i}"
    mt = f"vision_encoder.blocks.{i}"
    # qkv 是 fused，下面用 SPLIT_QKV 标记，在转换时拆分
    return {
        f"{pfx}.attn.qkv.weight": (f"{mt}.attn.SPLIT_QKV", "split_qkv"),
        f"{pfx}.attn.qkv.bias":   (f"{mt}.attn.SPLIT_QKV_BIAS", "split_qkv_bias"),
        f"{pfx}.attn.proj.weight": (f"{mt}.attn.attn.out.kernel", True),
        f"{pfx}.attn.proj.bias":   (f"{mt}.attn.attn.out.bias",   False),
        f"{pfx}.mlp.linear_fc1.weight": (f"{mt}.mlp.kernel",     True),
        f"{pfx}.mlp.linear_fc1.bias":   (f"{mt}.mlp.bias",       False),
        f"{pfx}.mlp.linear_fc2.weight": (f"{mt}.mlp_out.kernel", True),
        f"{pfx}.mlp.linear_fc2.bias":   (f"{mt}.mlp_out.bias",   False),
        f"{pfx}.norm1.weight": (f"{mt}.norm1.scale", False),
        f"{pfx}.norm1.bias":   (f"{mt}.norm1.bias",  False),
        f"{pfx}.norm2.weight": (f"{mt}.norm2.scale", False),
        f"{pfx}.norm2.bias":   (f"{mt}.norm2.bias",  False),
    }

def build_vision_mapping(num_vit_layers: int, num_heads: int, head_dim: int,
                          deepstack_indexes: list) -> dict:
    """构建完整 ViT 权重映射。"""
    mapping = {
        "model.visual.patch_embed.proj.weight": ("vision_encoder.patch_embed.proj.kernel", False),
        "model.visual.patch_embed.proj.bias":   ("vision_encoder.patch_embed.proj.bias",   False),
        "model.visual.pos_embed.weight":        ("vision_encoder.pos_embed_interpolate.embedding", False),
        # Patch merger（输出端）
        "model.visual.merger.norm.weight":      ("vision_projector.merger.ln_q.scale", False),
        "model.visual.merger.norm.bias":        ("vision_projector.merger.ln_q.bias",  False),
        "model.visual.merger.linear_fc1.weight":("vision_projector.merger.mlp.wi_0.kernel", True),
        "model.visual.merger.linear_fc1.bias":  ("vision_projector.merger.mlp.wi_0.bias",  False),
        "model.visual.merger.linear_fc2.weight":("vision_projector.merger.mlp.wo.kernel",  True),
        "model.visual.merger.linear_fc2.bias":  ("vision_projector.merger.mlp.wo.bias",    False),
    }
    # DeepStack intermediate mergers（层 8, 16, 24 的中间特征提取）
    for ds_idx, layer_idx in enumerate(deepstack_indexes):
        pfx = f"model.visual.deepstack_merger_list.{ds_idx}"
        mt  = f"vision_encoder.deepstack_mergers.{ds_idx}"
        mapping.update({
            f"{pfx}.norm.weight":      (f"{mt}.ln_q.scale",       False),
            f"{pfx}.norm.bias":        (f"{mt}.ln_q.bias",        False),
            f"{pfx}.linear_fc1.weight":(f"{mt}.mlp.wi_0.kernel",  True),
            f"{pfx}.linear_fc1.bias":  (f"{mt}.mlp.wi_0.bias",    False),
            f"{pfx}.linear_fc2.weight":(f"{mt}.mlp.wo.kernel",    True),
            f"{pfx}.linear_fc2.bias":  (f"{mt}.mlp.wo.bias",      False),
        })
    # ViT blocks
    for i in range(num_vit_layers):
        mapping.update(vit_layer_mapping(i, num_heads, head_dim))
    return mapping


# ── 加载 HF safetensors（懒加载，按文件流式处理）─────────────────────────

def iter_hf_weights(model_path: str):
    """逐文件 yield (key, numpy_array)，峰值 RAM ≈ 单个文件大小（~4-5GB）。"""
    if model_path.startswith("gs://"):
        import subprocess, tempfile
        tmp = tempfile.mkdtemp()
        files = sorted(glob.glob(os.path.join(tmp, "model-*.safetensors")))
        if not files:
            subprocess.run(
                ["gcloud", "storage", "cp", f"{model_path}/model-*.safetensors", tmp],
                check=True
            )
            files = sorted(glob.glob(os.path.join(tmp, "model-*.safetensors")))
    else:
        files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))

    for path in tqdm(files, desc="处理 safetensors 文件"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key).float().numpy()


def load_hf_weights(model_path: str) -> dict:
    """一次性加载所有权重（内存充足时使用）。"""
    if model_path.startswith("gs://"):
        import subprocess, tempfile
        tmp = tempfile.mkdtemp()
        subprocess.run(["gcloud", "storage", "cp", "-r", f"{model_path}/*.safetensors", tmp], check=True)
        pattern = os.path.join(tmp, "*.safetensors")
    else:
        pattern = os.path.join(model_path, "*.safetensors")

    weights = {}
    for path in tqdm(sorted(glob.glob(pattern)), desc="加载 safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k).float().numpy()
    print(f"已加载 {len(weights)} 个权重张量")
    return weights


# ── 主转换逻辑 ────────────────────────────────────────────────────────────

def _apply_mapping(hf_k: str, w: np.ndarray, mt_k: str, op,
                   params: dict, maxtext_weights: dict):
    """将单个 HF 权重应用映射写入 maxtext_weights。"""
    num_heads    = params["num_attention_heads"]
    num_kv_heads = params["num_key_value_heads"]
    vit_hs       = params["vit_hidden_size"]

    if "SPLIT_QKV" in mt_k:
        is_bias = "BIAS" in mt_k
        base_mt = mt_k.replace("SPLIT_QKV_BIAS", "").replace("SPLIT_QKV", "")
        head_dim = params["head_dim"]
        if not is_bias:
            q, k, v = np.split(w, [
                num_heads * head_dim,
                num_heads * head_dim + num_kv_heads * head_dim,
            ], axis=0)
            maxtext_weights[f"{base_mt}attn.query.kernel"] = q.T
            maxtext_weights[f"{base_mt}attn.key.kernel"]   = k.T
            maxtext_weights[f"{base_mt}attn.value.kernel"] = v.T
        else:
            maxtext_weights[f"{base_mt}attn.query.bias"] = w[:vit_hs]
            maxtext_weights[f"{base_mt}attn.key.bias"]   = w[vit_hs:2*vit_hs]
            maxtext_weights[f"{base_mt}attn.value.bias"] = w[2*vit_hs:]
    elif op is True:
        maxtext_weights[mt_k] = w.T
    else:
        maxtext_weights[mt_k] = w


def convert_lazy(model_path: str, params: dict) -> dict:
    """懒加载转换：逐文件处理，峰值 RAM ≈ 单文件大小（~4GB）。"""
    num_layers     = params["num_hidden_layers"]
    num_vit_layers = params["num_vit_layers"]
    vit_num_heads  = params["vit_num_heads"]
    head_dim       = params["head_dim"]
    deepstack_idxs = params["deepstack_indexes"]

    full_map = {
        **build_text_mapping(num_layers),
        **build_vision_mapping(num_vit_layers, vit_num_heads, head_dim, deepstack_idxs),
    }

    maxtext_weights = {}
    unmapped = []
    for hf_k, w in iter_hf_weights(model_path):
        if hf_k in full_map:
            mt_k, op = full_map[hf_k]
            _apply_mapping(hf_k, w, mt_k, op, params, maxtext_weights)
        else:
            unmapped.append(hf_k)
        del w  # 立即释放

    if unmapped:
        print(f"\n⚠️  {len(unmapped)} 个 HF key 未映射：{unmapped[:5]}")
    print(f"懒加载转换完成：{len(maxtext_weights)} 个 MaxText 权重张量")
    return maxtext_weights


def convert(hf_weights: dict, params: dict, dry_run: bool = False) -> Optional[dict]:
    """一次性转换（内存充足时使用）。dry_run=True 只打印映射。"""
    num_layers     = params["num_hidden_layers"]
    num_vit_layers = params["num_vit_layers"]
    vit_num_heads  = params["vit_num_heads"]
    head_dim       = params["head_dim"]
    deepstack_idxs = params["deepstack_indexes"]

    text_map   = build_text_mapping(num_layers)
    vision_map = build_vision_mapping(num_vit_layers, vit_num_heads, head_dim, deepstack_idxs)
    full_map   = {**text_map, **vision_map}

    unmapped = set(hf_weights.keys()) - set(full_map.keys())
    if unmapped:
        print(f"\n⚠️  {len(unmapped)} 个 HF key 未映射：{sorted(unmapped)[:5]}")

    if dry_run:
        print("\n=== DRY RUN 映射预览（前20条）===")
        for hf_k, (mt_k, op) in list(full_map.items())[:20]:
            shape = hf_weights.get(hf_k, np.array([])).shape
            print(f"  {hf_k}\n    → {mt_k}  [{op}]  {shape}")
        print(f"\n共 {len(full_map)} 条映射（文本: {len(text_map)}，视觉: {len(vision_map)}）")
        return None

    maxtext_weights = {}
    for hf_k, (mt_k, op) in tqdm(full_map.items(), desc="转换权重"):
        if hf_k not in hf_weights:
            continue
        _apply_mapping(hf_k, hf_weights[hf_k], mt_k, op, params, maxtext_weights)

    print(f"转换完成：{len(maxtext_weights)} 个 MaxText 权重张量")
    return maxtext_weights


def save_orbax(maxtext_weights: dict, output_path: str):
    """将转换后的权重保存为 Orbax checkpoint。"""
    import orbax.checkpoint as ocp
    from etils import epath

    # 把 flat dict 还原成嵌套 dict（以 . 分层）
    def _nested(flat):
        tree = {}
        for key, val in flat.items():
            parts = key.split(".")
            d = tree
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val
        return tree

    nested = _nested(maxtext_weights)
    out = epath.Path(output_path)
    ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    ckptr.save(out, nested, force=True)
    print(f"Orbax checkpoint 已保存至：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-VL-8B HF → MaxText")
    parser.add_argument("--hf_model_path",  required=True,  help="HF 权重路径（本地或 gs://）")
    parser.add_argument("--output_path",    required=True,  help="MaxText Orbax 输出路径（gs://）")
    parser.add_argument("--model_size",     default="qwen3vl-8b")
    parser.add_argument("--dry_run", default="True",  help="只验证映射，不写文件")
    parser.add_argument("--lazy",    default="True",  help="懒加载（推荐，峰值 RAM ~4GB）")
    args = parser.parse_args()

    dry_run = args.dry_run.lower() in ("true", "1", "yes")
    lazy    = args.lazy.lower()    in ("true", "1", "yes")
    params  = MODEL_PARAMS[args.model_size]

    if dry_run:
        # dry_run 时仍需加载权重来显示 shape，但可以只加载第一个文件
        print(f"DRY RUN：加载 HF 权重（仅验证映射）...")
        hf_weights = load_hf_weights(args.hf_model_path)
        convert(hf_weights, params, dry_run=True)
    elif lazy:
        print(f"懒加载模式（峰值 RAM ≈ 4GB）：{args.hf_model_path}")
        maxtext_weights = convert_lazy(args.hf_model_path, params)
        save_orbax(maxtext_weights, args.output_path)
    else:
        print(f"全量加载（需要 ~32GB RAM）：{args.hf_model_path}")
        hf_weights = load_hf_weights(args.hf_model_path)
        maxtext_weights = convert(hf_weights, params)
        if maxtext_weights:
            save_orbax(maxtext_weights, args.output_path)


if __name__ == "__main__":
    main()
