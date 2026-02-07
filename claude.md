# Qwen3-VL 项目笔记

## 项目概览

Qwen3-VL 是阿里巴巴的多模态视觉语言模型仓库，支持图像理解、视频理解、OCR、UI 操控等能力。

### 目录结构

```
Qwen3-VL/
├── qwen-vl-utils/          # 视觉处理工具包（图像/视频加载、缩放）
│   └── src/qwen_vl_utils/
│       ├── __init__.py
│       └── vision_process.py   # 图像/视频预处理核心（smart_resize、fetch_image/video）
├── qwen-vl-finetune/       # 微调框架
│   ├── qwenvl/
│   │   ├── data/
│   │   │   ├── data_processor.py  # 数据加载、collation、padding
│   │   │   ├── rope2d.py          # 2D RoPE 位置编码（三个变体）
│   │   │   └── __init__.py        # 数据集注册表
│   │   └── train/
│   │       ├── train_qwen.py      # 训练入口，模型加载、LoRA 配置
│   │       ├── trainer.py         # 自定义 Trainer，Flash Attention patch、分组件 LR
│   │       └── argument.py        # 训练参数 dataclass
│   ├── scripts/             # 训练脚本 + DeepSpeed 配置（zero2/3/offload）
│   ├── demo/                # 示例数据
│   └── tools/               # 数据打包、bbox 转换工具
├── evaluation/              # 基准评测套件
│   ├── VideoMME/            # 视频多模态评测
│   ├── mmmu/                # MMMU 评测
│   ├── MathVision/          # 数学视觉推理
│   ├── ODinW-13/            # 目标检测
│   └── RealWorldQA/         # 真实世界 QA
├── cookbooks/               # Jupyter 示例（OCR、grounding、agent 等）
├── docker/                  # Docker 部署
└── web_demo_mm.py           # Gradio Web 演示界面
```

### 支持的模型

- Qwen2-VL → `Qwen2VLForConditionalGeneration`
- Qwen2.5-VL → `Qwen2_5_VLForConditionalGeneration`
- Qwen3-VL (Dense) → `Qwen3VLForConditionalGeneration`
- Qwen3-VL-MoE → `Qwen3VLMoeForConditionalGeneration`（仅支持 ZeRO-2）

规模：2B、4B、8B、32B、30B-A3B（MoE）、235B-A22B（MoE）
版本：Instruct（标准指令跟随）和 Thinking（增强推理）

### 核心依赖

- transformers (>=4.57.0) + torch — 模型加载与推理
- vLLM (>=0.11.0) — 高吞吐推理
- DeepSpeed (>=0.17.1) — 分布式训练（ZeRO-2/3）
- peft (>=0.17.1) — LoRA 微调
- Gradio — Web 界面
- av / decord — 视频解码

---

## 微调模块分析 (`qwen-vl-finetune/`)

### 支持的训练方法

| 方法 | 状态 | 说明 |
|------|------|------|
| SFT（监督微调） | 支持 | 唯一实现的训练目标，基于 next-token prediction |
| LoRA | 支持 | 通过 `--lora_enable True`，目标模块为 q/k/v/o_proj |
| 全参数微调 | 支持 | 可分组件控制（视觉编码器/MLP 投影器/LLM） |
| DeepSpeed ZeRO-2/3 | 支持 | 分布式训练优化 |

### 不支持的训练方法

- RLHF — 无实现
- DPO / PPO — 无偏好学习
- QLoRA（量化+LoRA） — 不支持 8-bit/4-bit 量化
- Prefix Tuning / Adapter — 只有 LoRA 一种 PEFT 方法
- 多任务训练 / 辅助损失 — 仅单一 causal LM 目标

### 组件级调控

通过三个标志位独立控制哪些部分参与训练：

```bash
--tune_mm_vision False   # 视觉编码器（默认冻结）
--tune_mm_mlp True       # MLP 投影层
--tune_mm_llm True       # 语言模型主体
```

支持分组件学习率：`--mm_projector_lr` 和 `--vision_tower_lr` 可独立于 `--learning_rate`。

---

## PyTorch/GPU 到 JAX/TPU 迁移计划

### 一、整体策略

有两条路线：

| 路线 | 方案 | 适合场景 |
|------|------|----------|
| A. 基于 Flax/NNX | 用 Flax 重写模型 + 自定义训练循环 | 需要极致 TPU 性能、长期维护 |
| B. 基于 MaxText | 在 Google 已有的 JAX LLM 训练框架上适配 | 快速落地、减少重复造轮子 |

建议路线 B：MaxText / EasyLM 已经解决了 TPU 分布式训练、sharding、checkpoint 等基础设施问题，只需关注模型结构和数据管道的适配。

### 二、逐模块迁移方案

#### 1. 模型定义（最核心、工作量最大）

**现状**：依赖 `transformers` 中的 `Qwen3VLForConditionalGeneration` 等 PyTorch 类。

**方案**：

- 用 Flax `nn.Module` 重写三个子组件：
  - Vision Encoder (ViT)：标准 ViT + DeepStack 多层特征融合
  - MLP Projector（Merger）：跨模态对齐层
  - LLM Backbone：Qwen3 的 decoder-only transformer
- 参考 Google 的 MaxText (https://github.com/google/maxtext) 中 Transformer 实现模式
- MoE 变体需要额外实现 expert routing，可参考 Flaxformer 中 Mixtral 的实现

**关键差异**：

```python
# PyTorch（当前）
class Attention(nn.Module):
    def forward(self, x):
        ...
        return output

# JAX/Flax（目标）
class Attention(nn.Module):
    @nn.compact
    def __call__(self, x, deterministic=False):
        ...
        return output
# JAX 模型是函数式的，无 in-place 操作，状态通过参数显式传递
```

#### 2. Attention 实现（关键难点）

**现状**：`trainer.py` 中大量 monkey-patch，用 `flash_attn_varlen_func` 处理变长序列。

涉及代码位置：
- `trainer.py` 中 `flash_attention_forward()` (lines 33-108)
- `qwen2vl_forward()` 和 `qwen3vl_forward()` (lines 112-200)
- `replace_qwen2_vl_attention_class()` (lines 215-252) monkey-patch 了以下类：
  - `Qwen2VLAttention.forward`
  - `Qwen2_5_VLAttention.forward`
  - `Qwen3VLTextAttention.forward`
  - `Qwen3VLMoeTextAttention.forward`

**方案**：

- TPU 上不需要 Flash Attention。TPU 的 HBM 带宽和计算模式不同于 GPU，JAX/XLA 编译器会自动优化 attention 的内存布局
- 使用 `jax.nn.dot_product_attention`（JAX 0.4.31+）或手写 `softmax(QK^T/sqrt(d))V`，XLA 会在 TPU 上做 kernel fusion
- 变长序列处理：用 attention mask + padding 或 BlockSparse 替代 `varlen_func`
- 如需进一步优化，可使用 Pallas (https://jax.readthedocs.io/en/latest/pallas/) 写自定义 TPU kernel（类似 Splash Attention）

#### 3. RoPE 位置编码（工作量中等）

**现状**：`rope2d.py` 中有三个变体（`get_rope_index_2`、`get_rope_index_25`、`get_rope_index_3`），纯 PyTorch 张量操作，无自定义 CUDA kernel。

涉及代码位置：
- `rope2d.py` 全文，三个函数分别对应 Qwen2-VL、Qwen2.5-VL、Qwen3-VL
- 使用的 PyTorch ops：`torch.arange`、`torch.cat`、`torch.stack`、`tensor.view`、`tensor.unsqueeze`、`tensor.cumsum`、`tensor.argwhere`、`tensor.masked_fill_()`

**方案**：

直接用 `jnp`（jax.numpy）替换 `torch` 操作，API 基本一一对应：

```
torch.arange       → jnp.arange
torch.cat           → jnp.concatenate
torch.stack         → jnp.stack
tensor.view         → jnp.reshape
tensor.unsqueeze    → jnp.expand_dims
tensor.cumsum       → jnp.cumsum
tensor.argwhere     → jnp.argwhere
masked_fill_()      → jnp.where(mask, value, array)  # JAX 无 in-place 操作
```

去掉所有 `.to(device)` 调用，JAX 通过 `jax.device_put` 和 sharding 注解管理设备放置。

硬编码的 token ID 常量可保留：
- `image_token_id = 151655`
- `video_token_id = 151656`
- `vision_start_token_id = 151652`

#### 4. 分布式训练（架构变化最大）

**现状**：DeepSpeed ZeRO-2/3 + `torchrun` 启动器。

涉及代码位置：
- `train_qwen.py`：`torch.cuda.synchronize()` (line 56)、`torch.distributed.get_rank()` (line 182)
- `scripts/*.sh`：`torchrun --nproc_per_node`
- `scripts/zero2.json`、`zero3.json`、`zero3_offload.json`

**方案**：

使用 JAX 的 SPMD（单程序多数据）+ `jax.sharding`：

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# 定义设备 mesh（如 4x2 的 TPU pod slice）
mesh = Mesh(jax.devices().reshape(4, 2), ('data', 'model'))

# 参数分片注解
param_sharding = NamedSharding(mesh, P('model', None))  # 列切分
data_sharding = NamedSharding(mesh, P('data', None))     # 数据并行
```

- ZeRO-3 的等价物：JAX 的 FSDP 通过 `P(None)` 全分片 + `jax.checkpoint` 实现类似效果
- 启动方式：TPU pod 上直接用 `python` 启动，JAX 自动检测 TPU 拓扑，不需要 `torchrun`

#### 5. 优化器（中等工作量）

**现状**：`trainer.py` 中自定义 `create_optimizer()`，支持分组件学习率和 weight decay 分组。

涉及代码位置：
- `trainer.py`：`Trainer.create_optimizer()` 覆写 (line 495)、参数分组逻辑 (lines 321-484)
- `argument.py`：`optim = "adamw_torch"` (line 31)、`mm_projector_lr` (line 38)、`vision_tower_lr` (line 39)

**方案**：

使用 Optax (https://github.com/google-deepmind/optax) 替代 `torch.optim.AdamW`：

```python
import optax

def create_optimizer(params, lr_llm, lr_vision, lr_projector):
    # 用 optax.multi_transform 对不同参数组应用不同优化器
    label_fn = flax.traverse_util.path_aware_map(
        lambda path, _: 'vision' if 'visual' in path
                        else 'projector' if 'merger' in path
                        else 'llm'
    )
    tx = optax.multi_transform({
        'llm': optax.adamw(lr_llm, weight_decay=0.0),
        'vision': optax.adamw(lr_vision),
        'projector': optax.adamw(lr_projector),
    }, label_fn)
    return tx
```

- Cosine schedule：`optax.warmup_cosine_decay_schedule()`
- Gradient clipping：`optax.clip_by_global_norm(1.0)`

#### 6. 数据管道（中等工作量）

**现状**：`LazySupervisedDataset(torch.utils.data.Dataset)` + 自定义 DataCollator。

涉及代码位置：
- `data_processor.py`：`LazySupervisedDataset` (line 244) 继承 `torch.utils.data.Dataset`
- `DataCollatorForSupervisedDataset` (line 535)
- `FlattenedDataCollatorForSupervisedDataset` (line 605)
- 大量 `torch.cat`、`torch.nn.functional.pad`、`torch.nn.utils.rnn.pad_sequence`

**方案**：

- 使用 `tf.data` 或 Grain（Google 推荐的 JAX 数据加载库）替代 PyTorch DataLoader
- 图像/视频预处理（`qwen-vl-utils`）可以保持 CPU 端 Python/PIL/NumPy 代码不变，最后转为 `jnp.array`
- Collator 中的 padding/concat 操作直接改为 `jnp` 等价操作
- 数据 packing（`tools/pack_data.py`）逻辑是纯 Python 的 bin-packing 算法，可直接复用

#### 7. LoRA（中等工作量）

**现状**：通过 `peft.get_peft_model()` 包装。

涉及代码位置：
- `train_qwen.py`：`peft.LoraConfig()` + `peft.get_peft_model()` (lines 170-177)
- 目标模块：`["q_proj", "k_proj", "v_proj", "o_proj"]` (line 174)
- 参数：`lora_r=64`、`lora_alpha=128`、`lora_dropout=0.0`

**方案**：

JAX 没有 HuggingFace PEFT 的直接等价物，需手动实现 LoRA：

```python
class LoRALinear(nn.Module):
    features: int
    r: int = 64
    alpha: float = 128

    @nn.compact
    def __call__(self, x):
        base = nn.Dense(self.features, use_bias=False, name='base')(x)
        lora_a = nn.Dense(self.r, use_bias=False, name='lora_a')(x)
        lora_b = nn.Dense(self.features, use_bias=False, name='lora_b')(lora_a)
        return base + (self.alpha / self.r) * lora_b
```

冻结 base 权重：在 `train_step` 中只对 LoRA 参数计算梯度：

```python
trainable_params, frozen_params = partition(is_lora_param, params)
grads = jax.grad(loss_fn)(trainable_params, frozen_params)
```

#### 8. Gradient Checkpointing

**现状**：`model.enable_input_require_grads()` + HuggingFace gradient checkpointing。

涉及代码位置：
- `train_qwen.py`：`model.enable_input_require_grads()` (line 147)、forward hook 注册 (lines 150-153)

**方案**：

JAX 原生支持 `jax.checkpoint`（也叫 `jax.remat`）：

```python
class TransformerBlock(nn.Module):
    @nn.compact
    @nn.remat  # 等价于 gradient checkpointing
    def __call__(self, x):
        ...
```

TPU 上 `nn.remat` 通常是必须的，因为模型参数占用大量 HBM。

#### 9. Checkpoint 保存/加载

**现状**：DeepSpeed 特殊的分布式 checkpoint + HuggingFace `safe_serialization`。

涉及代码位置：
- `train_qwen.py`：`trainer.deepspeed` 检测 (line 55)、`trainer.save_state()` (line 196)

**方案**：

- 使用 Orbax (https://github.com/google/orbax) 做 checkpoint 管理（支持异步保存、分片存储）
- 从 HuggingFace PyTorch checkpoint 初始化权重：

```python
import safetensors
pt_state = safetensors.numpy.load_file("model.safetensors")
jax_params = jax.tree.map(jnp.array, pt_state)
# 然后 reshape/rename 以匹配 Flax 模型结构
```

### 三、迁移阶段规划

```
Phase 1: 模型前向推理
  ├─ 用 Flax 重写 ViT + Projector + LLM
  ├─ 迁移 RoPE（纯张量操作替换）
  ├─ 加载 HF 预训练权重到 JAX
  └─ 验证：对比 PyTorch/JAX 前向输出是否一致（tolerance ~1e-5）

Phase 2: 训练循环
  ├─ Optax 优化器（分组件 LR）
  ├─ jax.grad + train_step
  ├─ Gradient checkpointing (nn.remat)
  └─ 单 TPU chip 上跑通 SFT

Phase 3: 分布式扩展
  ├─ SPMD sharding 策略
  ├─ 多 TPU chip / pod 训练
  └─ Orbax checkpoint

Phase 4: 高级特性
  ├─ LoRA 实现
  ├─ 数据 packing
  └─ MoE 模型支持
```

### 四、TPU 与 GPU 关键差异

| 维度 | GPU (现状) | TPU (目标) |
|------|-----------|-----------|
| 内存模型 | 显存有限，需 Flash Attention | HBM 更大但带宽模式不同，XLA 自动优化 |
| 精度 | bf16/fp16 混合精度 | TPU 原生 bf16，不支持 fp16，只用 bf16 |
| 随机数 | `torch.manual_seed` | JAX 需显式传递 PRNG key，`jax.random.PRNGKey` |
| 动态 shape | 支持 | 尽量避免，XLA 编译要求静态 shape，动态部分用 padding + mask |
| 编译 | 即时执行 | `jax.jit` 首次编译慢但后续快，shape 变化触发重编译 |

**最关键的一点**：JAX/TPU 要求尽量静态 shape。当前代码中变长序列的处理（variable-length attention、dynamic frame count）需要改为固定长度 + padding + attention mask 的方式，否则会频繁触发 XLA 重编译导致性能极差。

### 五、PyTorch/GPU 耦合点速查表

| 文件 | 耦合点 | 迁移难度 |
|------|--------|----------|
| `trainer.py` | Flash Attention monkey-patch、自定义 optimizer、`torch.Tensor` 操作 | 高 |
| `train_qwen.py` | HF Trainer、DeepSpeed 集成、CUDA sync、LoRA (peft) | 高 |
| `data_processor.py` | `torch.utils.data.Dataset`、torch tensor 创建/padding/collation | 中 |
| `rope2d.py` | 纯 torch 张量操作（无 CUDA kernel），API 可一一对应迁移 | 中 |
| `argument.py` | DeepSpeed 参数、`adamw_torch` 优化器指定 | 低 |
| `scripts/*.sh` | `torchrun` 启动器、DeepSpeed JSON 配置 | 低（直接替换） |
| `qwen-vl-utils/` | CPU 端处理为主，少量 torch tensor 转换 | 低 |

---

## 多 Agent 并行迁移方案

### 一、Agent 拆分方案

基于代码依赖关系，拆分为 6 个 Agent，分 3 层执行：

```
Layer 0 (无依赖，可并行)
  ├─ Agent 1: RoPE 迁移
  ├─ Agent 2: 数据管道迁移
  └─ Agent 3: 工具函数迁移 (qwen-vl-utils)

Layer 1 (依赖 Layer 0)
  ├─ Agent 4: 模型定义 (ViT + Projector + LLM + LoRA)
  └─ Agent 5: 优化器 + 训练循环

Layer 2 (依赖 Layer 0+1)
  └─ Agent 6: 分布式训练 + Checkpoint
```

### 二、各 Agent 职责与接口契约

#### Agent 1: RoPE 迁移

**输入**：`qwenvl/data/rope2d.py`

**输出**：`jax_qwenvl/data/rope2d.py`

**职责**：
- 将三个 `get_rope_index_*` 函数从 torch 翻译为 jnp
- 消除所有 in-place 操作（`masked_fill_` → `jnp.where`）
- 消除 `.to(device)` 调用

**接口契约**：

```python
# 函数签名保持不变，类型从 torch.Tensor 改为 jax.Array
def get_rope_index_3(
    input_ids: jax.Array,          # [batch, seq_len]
    image_grid_thw: jax.Array,     # [num_images, 3]
    video_grid_thw: jax.Array,     # [num_videos, 3]
    attention_mask: jax.Array,     # [batch, seq_len]
) -> tuple[jax.Array, jax.Array]: # (position_ids [3, batch, seq_len], mrope_position_deltas)
```

**验证标准**：给定相同输入，PyTorch 版和 JAX 版的输出数值差 < 1e-6

**注意事项**：
- `jnp.argwhere` 返回的形状是静态 padded 的，行为与 `torch.argwhere` 不同，需要特殊处理
- 涉及动态长度循环（遍历每个 image/video 的 grid），可能需要 `jax.lax.scan` 或 `jax.lax.fori_loop` 替代 Python for 循环，或者接受这部分在 CPU 上预计算（不进 `jax.jit`）

#### Agent 2: 数据管道迁移

**输入**：`qwenvl/data/data_processor.py`、`qwenvl/data/__init__.py`

**输出**：`jax_qwenvl/data/data_processor.py`

**职责**：
- `LazySupervisedDataset` 改为 Grain DataSource 或 `tf.data` pipeline
- DataCollator 中的 padding/concat 从 `torch.nn.functional.pad` 改为 `jnp.pad`
- 保留数据集注册表逻辑
- 保留数据 packing 逻辑（纯 Python）

**接口契约**：

```python
# 每个 batch 输出的字典结构
@dataclass
class Batch:
    input_ids: jax.Array       # [batch, max_seq_len], padded
    labels: jax.Array          # [batch, max_seq_len], -100 for ignored
    attention_mask: jax.Array  # [batch, max_seq_len]
    pixel_values: jax.Array    # [total_patches, channels, h, w]
    image_grid_thw: jax.Array  # [num_images, 3]
    video_grid_thw: jax.Array  # [num_videos, 3]
    position_ids: jax.Array    # [3, batch, max_seq_len]
```

**关键决策**：
- 序列长度需要统一为静态值（如 `model_max_length=8192`），避免 XLA 重编译
- `pixel_values` 的图像数量在 batch 间可能不同，需要 padding 到固定数量或使用 per-sample 处理

#### Agent 3: 工具函数迁移 (qwen-vl-utils)

**输入**：`qwen-vl-utils/src/qwen_vl_utils/vision_process.py`

**输出**：`jax_qwenvl/utils/vision_process.py`

**职责**：
- `smart_resize`、`fetch_image`、`fetch_video` 这些是 CPU 端 PIL/NumPy 操作，基本不需要改
- 只需将最终的 `torch.tensor()` 输出改为 `np.array`（保持在 CPU，后续由数据管道转到 TPU）
- 移除 `torchvision.transforms` 依赖，用 PIL 或 NumPy 等价操作替代

**接口契约**：

```python
def process_vision_info(messages: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """返回 (image_list, video_list)，均为 numpy array"""
```

工作量最小，适合作为热身任务。

#### Agent 4: 模型定义（工作量最大）

**依赖**：Agent 1 (RoPE)、Agent 3 (vision utils)

**输入**：HuggingFace transformers 中 Qwen3-VL 的模型代码（非本仓库内，来自 transformers 库）

**输出**：

```
jax_qwenvl/model/
├── vit.py          # Vision Encoder
├── projector.py    # MLP Merger
├── llm.py          # Qwen3 LLM backbone
├── qwen3_vl.py     # 组合模型
└── lora.py         # LoRA 层实现
```

**职责**：
- 用 Flax `nn.Module` 重写完整模型
- 实现 Attention（使用 `jax.nn.dot_product_attention`，不需要 Flash Attention）
- 集成 RoPE（调用 Agent 1 的输出）
- 实现 LoRA 层（替代 peft 库）
- 实现权重加载函数：HuggingFace safetensors → Flax params

**接口契约**：

```python
class Qwen3VLForCausalLM(nn.Module):
    config: Qwen3VLConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jax.Array,
        pixel_values: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        image_grid_thw: jax.Array,
        video_grid_thw: jax.Array,
        deterministic: bool = True,
    ) -> CausalLMOutput:  # logits: [batch, seq_len, vocab_size]
```

建议进一步拆成 ViT / LLM / 组合三个子任务。

#### Agent 5: 优化器 + 训练循环

**依赖**：Agent 4 (模型定义)、Agent 2 (数据管道)

**输入**：`qwenvl/train/trainer.py`、`qwenvl/train/argument.py`

**输出**：`jax_qwenvl/train/train_step.py`、`jax_qwenvl/train/optimizer.py`

**职责**：
- Optax 优化器（分组件 LR + cosine schedule + gradient clipping）
- `train_step` 函数（`jax.jit` + `jax.grad`）
- Gradient checkpointing（`nn.remat`）
- 训练入口脚本

**接口契约**：

```python
@jax.jit
def train_step(
    state: TrainState,         # 包含 params + opt_state + step
    batch: Batch,
    rng: jax.random.PRNGKey,
) -> tuple[TrainState, dict]:  # (updated_state, metrics_dict)
```

#### Agent 6: 分布式训练 + Checkpoint

**依赖**：Agent 5 (训练循环)

**输入**：`scripts/zero*.json`、`scripts/sft*.sh`

**输出**：`jax_qwenvl/train/sharding.py`、`jax_qwenvl/train/checkpoint.py`、`scripts/tpu_train.sh`

**职责**：
- 设计 SPMD sharding 策略（数据并行 + tensor 并行）
- Orbax checkpoint 保存/恢复
- TPU pod 启动脚本
- MoE 模型的 expert 并行策略

### 三、Agent 间协作原则

#### 1. 共享类型定义文件

创建一个所有 Agent 共用的 `jax_qwenvl/types.py`，在最开始就定好：

```python
# types.py — 所有 Agent 的共享契约
from typing import NamedTuple
import jax
import jax.numpy as jnp

class Batch(NamedTuple):
    input_ids: jax.Array
    labels: jax.Array
    attention_mask: jax.Array
    pixel_values: jax.Array
    image_grid_thw: jax.Array
    video_grid_thw: jax.Array
    position_ids: jax.Array

class CausalLMOutput(NamedTuple):
    logits: jax.Array
    loss: jax.Array | None = None
```

#### 2. 每个 Agent 必须写数值验证测试

```python
# 每个模块都需要这种对比测试
def test_rope_equivalence():
    # 相同输入
    input_ids = ...
    # PyTorch 版结果
    pt_result = pt_get_rope_index_3(torch_input)
    # JAX 版结果
    jax_result = jax_get_rope_index_3(jax_input)
    # 对比
    np.testing.assert_allclose(pt_result.numpy(), np.array(jax_result), atol=1e-5)
```

#### 3. 静态 shape 约定

在开始前统一所有 Agent 遵循的 shape 约定：

```python
# config.py — 全局静态 shape 约定
MAX_SEQ_LEN = 8192
MAX_IMAGES_PER_SAMPLE = 8
MAX_VIDEO_FRAMES = 64
IMAGE_PATCH_SIZE = 28
MAX_PIXELS = 576 * 28 * 28
```

#### 4. 避免的反模式

| 反模式 | 问题 | 正确做法 |
|--------|------|----------|
| Agent 间通过口头描述接口 | 集成时类型不匹配 | 先写 `types.py` 再开工 |
| 每个 Agent 独立选择 Flax API 风格 | `nn.compact` vs `setup()` 混用 | 统一用 `nn.compact` |
| 跳过数值验证直接联调 | 错误传播难以定位 | 每个模块独立验证后再组合 |
| 在 JAX 中保留动态 shape | XLA 反复重编译 | 所有维度 padding 到静态值 |
| Agent 4 等 Agent 1 完全做完才开始 | 串行瓶颈 | Agent 4 先用 mock RoPE 开发，后替换 |

### 四、执行顺序建议

```
Week 1:
  [全体] 定义 types.py + config.py + 项目骨架
  [并行] Agent 1 (RoPE) + Agent 2 (数据管道) + Agent 3 (工具函数)

Week 2-3:
  [并行] Agent 4 (模型，用 mock RoPE/data) + Agent 5 前半部分 (optimizer)
  [串行] Agent 1/2/3 完成 → 数值验证 → 交付给 Agent 4/5

Week 3-4:
  [串行] Agent 4 + Agent 5 集成 → 单 TPU 跑通 SFT
  [并行] Agent 6 (分布式) 开始

Week 4-5:
  [串行] 全链路集成 → 多 TPU 训练验证 → 与 PyTorch 版 loss 曲线对比
```

### 五、风险点

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| `jnp.argwhere` 的动态 shape 行为 | RoPE 迁移卡住 | RoPE 预计算放在 CPU，不进 `jit` |
| 模型权重 naming 不匹配 | 权重加载失败 | 专门写一个 key mapping 函数，充分测试 |
| 多模态 token 数量 batch 间不一致 | XLA 重编译 | 固定 `pixel_values` shape，不足的 pad |
| MoE expert routing 的 top-k 是动态的 | 编译失败或性能差 | 用 `jax.lax.top_k`（静态 k），参考 MaxText 的 MoE 实现 |
| Flax 模型与 HF tokenizer 的整合 | 输入格式不兼容 | tokenizer 保持用 HF 的，只是输出转 jnp |

### 六、目标输出目录结构

```
jax_qwenvl/
├── types.py                # 共享类型定义（Batch, CausalLMOutput）
├── config.py               # 全局静态 shape 约定
├── data/
│   ├── rope2d.py           # Agent 1 输出
│   ├── data_processor.py   # Agent 2 输出
│   └── __init__.py         # 数据集注册表
├── utils/
│   └── vision_process.py   # Agent 3 输出
├── model/
│   ├── vit.py              # Agent 4 输出：Vision Encoder
│   ├── projector.py        # Agent 4 输出：MLP Merger
│   ├── llm.py              # Agent 4 输出：Qwen3 LLM backbone
│   ├── qwen3_vl.py         # Agent 4 输出：组合模型
│   ├── lora.py             # Agent 4 输出：LoRA 层
│   └── weight_loader.py    # Agent 4 输出：HF → Flax 权重转换
├── train/
│   ├── optimizer.py        # Agent 5 输出
│   ├── train_step.py       # Agent 5 输出
│   ├── sharding.py         # Agent 6 输出
│   └── checkpoint.py       # Agent 6 输出
├── scripts/
│   └── tpu_train.sh        # Agent 6 输出
└── tests/
    ├── test_rope.py        # Agent 1 数值验证
    ├── test_data.py        # Agent 2 验证
    ├── test_model.py       # Agent 4 前向对比
    └── test_train_step.py  # Agent 5 验证
```

---

## Layer 0 迁移实施记录

### 完成状态

Layer 0 迁移已完成。3 个 Agent 并行执行，所有模块从 PyTorch 迁移到纯 NumPy。

### 生成的文件

| 文件 | 行数 | Agent | 说明 |
|------|------|-------|------|
| `jax_qwenvl/__init__.py` | 1 | 共享 | 包入口 |
| `jax_qwenvl/types.py` | 30 | 共享 | `Batch`、`CausalLMOutput` NamedTuple，所有字段 `np.ndarray` |
| `jax_qwenvl/config.py` | 31 | 共享 | token ID、shape 常量、视觉处理默认值 |
| `jax_qwenvl/data/__init__.py` | 65 | Agent 2 | 数据集注册表（纯 Python 配置，从源复制） |
| `jax_qwenvl/data/rope2d.py` | 494 | Agent 1 | 3 个 RoPE 函数（`get_rope_index_3/25/2`），纯 numpy |
| `jax_qwenvl/data/data_processor.py` | 727 | Agent 2 | Dataset + 2 个 Collator，返回 `Batch` NamedTuple |
| `jax_qwenvl/utils/__init__.py` | 7 | Agent 3 | 公共 API 导出 |
| `jax_qwenvl/utils/vision_process.py` | 492 | Agent 3 | 图像/视频加载，无 torch/torchvision 依赖 |

### 验证结果

- 零 `import torch` / `from torch` 顶层引用
- 仅有 2 处 `np.int64`（视频帧索引，`.tolist()` 后立即转为 Python int，不进 TPU）
- `python3 -c "import jax_qwenvl"` 导入成功
- 所有整数 dtype 使用 `np.int32`（TPU 兼容）
- 所有浮点 dtype 使用 `np.float32`

### 关键设计决策

1. **纯 NumPy 而非 JAX NumPy**：Layer 0 是 CPU 端数据预处理，不需要 `jax.jit`，使用纯 numpy 避免动态 shape 问题
2. **`_ensure_numpy()` 辅助函数**：处理 HF processor 可能返回 torch tensor 的情况，通过 duck typing (`hasattr(val, 'numpy')`) 而非 import torch
3. **删除 torchvision 视频后端**：decord 为主、torchcodec 为备（lazy import），无 torchvision fallback
4. **PIL 逐帧 resize 替代 torchvision resize**：CPU 端预处理性能足够
5. **`_pad_sequence()` 自实现**：替代 `torch.nn.utils.rnn.pad_sequence()`
6. **Collator 返回 `Batch` NamedTuple**：确保下游 `jax.device_put()` 兼容

---

## Layer 1 迁移完成记录

**完成时间**：2026-02-07
**迁移内容**：模型定义（Flax nn.Module）+ 训练基础设施（Optax）

### Agent 4：模型定义（8 个文件）

| 文件 | 行数 | 内容 |
|---|---|---|
| `jax_qwenvl/model/__init__.py` | 5 | 导出 Config, Model, load_hf_weights |
| `jax_qwenvl/model/config.py` | 156 | Qwen3VLConfig / VisionConfig / TextConfig dataclass，含 `from_pretrained` |
| `jax_qwenvl/model/layers.py` | 105 | RMSNorm, SwiGLUMLP, VisionMLP, LoRADense |
| `jax_qwenvl/model/rope.py` | 191 | Vision RoPE（2D 空间位置查找），Text MRoPE（3D 交错频率），apply_rotary_pos_emb |
| `jax_qwenvl/model/vit.py` | 421 | PatchEmbed3D, VisionAttention, VisionBlock, PatchMerger, VisionModel（含 DeepStack） |
| `jax_qwenvl/model/llm.py` | 309 | TextAttention（GQA + q/k_norm）, DecoderLayer, TextModel（含 DeepStack 注入） |
| `jax_qwenvl/model/qwen3_vl.py` | 259 | Qwen3VLForConditionalGeneration, cross_entropy_loss, _scatter_embeddings |
| `jax_qwenvl/model/weight_loader.py` | 308 | HF safetensors → Flax params 转换，key 映射 + 转置 + 分片加载 |

### Agent 5：训练基础设施（5 个文件）

| 文件 | 行数 | 内容 |
|---|---|---|
| `jax_qwenvl/train/__init__.py` | 3 | 导出 create_optimizer, TrainState, train_step |
| `jax_qwenvl/train/optimizer.py` | 349 | optax.multi_transform 6+1 参数组，warmup+cosine schedule |
| `jax_qwenvl/train/train_state.py` | 45 | 扩展 Flax TrainState |
| `jax_qwenvl/train/train_step.py` | 92 | @jax.jit train_step + cross_entropy_loss |
| `jax_qwenvl/train/train.py` | 301 | 训练入口：参数解析、模型加载、权重合并、训练循环 |

### 验证结果

- 零 `import torch` 在 model/ 和 train/ 中
- 所有模块导入成功
- 模型 `init` + `forward` 通过（text-only、vision+text、tied embeddings、LoRA 四种路径）
- `Qwen3VLConfig.from_pretrained()` 正确解析 HF config.json
- 优化器 6+1 参数组正确分类（vision_decay/nodecay, proj_decay/nodecay, llm_decay/nodecay, frozen）
- 10 步训练 loss 从 6.10 下降到 2.67（warmup 后开始收敛）
- LoRA 参数路径正确：`q_proj/base/kernel`, `q_proj/lora_A/kernel`, `q_proj/lora_B/kernel`

### 关键架构实现

1. **Vision RoPE**：2D 空间位置查找表 + flatten + duplicate，匹配 HF `rot_pos_emb` 方法
2. **Text MRoPE**：3D 交错频率布局 `[THWTHW...]`，匹配 HF `apply_interleaved_mrope`
3. **DeepStack**：ViT 在指定层（如 [8,16,24]）提取中间特征，注入 LLM 早期层
4. **q_norm / k_norm**：Qwen3 特有的 per-head RMSNorm（在 projection 后、RoPE 前）
5. **GQA**：通过 `jnp.repeat` 展开 K/V heads 到 Q heads 数量
6. **3D Conv PatchEmbed**：Flax `nn.Conv` 实现 `(temporal_patch_size, patch_size, patch_size)` 3D 卷积
7. **cu_seqlens → block-diagonal mask**：替代 Flash Attention varlen，用 JAX 实现
8. **LoRADense**：内联 LoRA（lora_rank=0 时退化为普通 Dense），初始化 lora_B=zeros

---

## Layer 2 迁移完成记录

**完成时间**：2026-02-07
**迁移内容**：分布式训练（SPMD）、Orbax Checkpoint、梯度累积、梯度检查点

### Agent 6：分布式训练 + Checkpoint

#### 新建文件

| 文件 | 行数 | 内容 |
|---|---|---|
| `jax_qwenvl/train/sharding.py` | 132 | `create_device_mesh()`, `get_param_sharding_rules()`, `shard_params()`, `shard_batch()` |
| `jax_qwenvl/train/checkpoint.py` | 83 | `CheckpointManager` 封装 Orbax `ocp.CheckpointManager` |

#### 修改文件

| 文件 | 行数 | 变更 |
|---|---|---|
| `jax_qwenvl/train/train_step.py` | 92 → 156 | 添加 `train_step_with_accumulation`（`jax.lax.scan` 梯度累积） |
| `jax_qwenvl/train/train.py` | 301 → 416 | 集成 mesh/sharding/checkpoint/gradient_accumulation + 新增 TrainingArguments 字段 |
| `jax_qwenvl/train/__init__.py` | 3 → 5 | 新增 sharding/checkpoint/accumulation 导出 |
| `jax_qwenvl/model/llm.py` | 309 → 318 | TextModel 添加 `gradient_checkpointing` + `nn.remat(DecoderLayer)` |
| `jax_qwenvl/model/vit.py` | 421 → 427 | VisionModel 添加 `gradient_checkpointing` + `nn.remat(VisionBlock)` |
| `jax_qwenvl/model/qwen3_vl.py` | 259 → 265 | 传递 `gradient_checkpointing` 到 VisionModel 和 TextModel |

### 新增 TrainingArguments 字段

```python
gradient_accumulation_steps: int = 1    # micro-batch 累积数
gradient_checkpointing: bool = False    # nn.remat 激活重算
fsdp: bool = False                      # FSDP 模式（否则纯 DP）
max_checkpoints: int = 3               # 保留的 checkpoint 数量
resume_from_checkpoint: Optional[str]   # checkpoint 恢复路径
```

### 关键实现

1. **SPMD Mesh**：3 轴 `('dp', 'fsdp', 'tp')`，DP 模式参数全复制 `P()`，FSDP 模式 2D kernel 沿 fsdp 轴分片 `P('fsdp', None)`
2. **Batch 分片**：`position_ids` 特殊处理（shape `(3, B, L)`，batch 在 axis=1：`P(None, 'dp', None)`）
3. **梯度累积**：`jax.lax.scan` 在 JIT 内循环累积，平均后 `apply_gradients`
4. **梯度检查点**：`nn.remat(DecoderLayer, policy=nothing_saveable)` 和 `nn.remat(VisionBlock)`
5. **Checkpoint**：Orbax `StandardSave`/`StandardRestore`，支持分片参数自动处理

### 验证结果

- 所有导入 OK（sharding, checkpoint, train_step_with_accumulation）
- 零 `import torch` 在 train/
- Mesh 创建正确（单设备 dp=1, fsdp=1, tp=1）
- DP/FSDP 分片规则正确
- `shard_batch` 和 `shard_params` 单设备通过
- 梯度检查点：tiny 模型 remat vs non-remat 输出完全一致（diff=0.0）
- Checkpoint save/restore round-trip：参数完全恢复（diff=0.0），步数正确
- CheckpointManager 创建和 should_save 逻辑正确

---

## Layer 3 迁移完成记录

**完成时间**：2026-02-07
**迁移内容**：权重导出（Flax → HF safetensors）、LoRA 合并、wandb/tensorboard 日志、数据 packing 集成、TPU 启动脚本

### Agent 7：生产化功能

#### 新建文件

| 文件 | 行数 | 内容 |
|---|---|---|
| `jax_qwenvl/model/weight_exporter.py` | 416 | LoRA 合并 + Flax→HF safetensors 导出（反向 key 映射 + 转置 + 分片保存） |
| `jax_qwenvl/train/metrics_logger.py` | 60 | wandb + tensorboard 统一日志封装（report_to="none" 时为 no-op） |
| `jax_qwenvl/scripts/train_tpu.sh` | 102 | TPU 训练启动脚本（环境变量覆盖所有参数） |

#### 修改文件

| 文件 | 行数 | 变更 |
|---|---|---|
| `jax_qwenvl/model/qwen3_vl.py` | 265 → 301 | 添加 `_make_packed_causal_mask()`（cu_seqlens → block-diagonal + causal mask）+ ndim==1 分支 |
| `jax_qwenvl/train/train.py` | 416 → 487 | 集成 MetricsLogger + FlattenedDataCollator + warmup_ratio + 训练后权重导出 + processor 保存 |
| `jax_qwenvl/train/__init__.py` | 5 → 6 | 新增 MetricsLogger 导出 |

### 新增 TrainingArguments 字段

```python
report_to: str = "none"        # "wandb", "tensorboard", "wandb,tensorboard", "none"
run_name: str = ""              # wandb/tensorboard run 名称
warmup_ratio: float = 0.0      # 若 > 0 且 warmup_steps == 0，则从 ratio 计算 warmup
```

### 关键实现

1. **LoRA 合并**：递归遍历参数树，检测 `{base, lora_A, lora_B}` 子结构，合并为 `kernel = base + (A @ B) * (alpha/rank)`，移除 LoRA 子键
2. **Flax → PyTorch key 映射**：反向执行 `weight_loader.py` 的映射规则（`blocks_N` → `blocks.N`，`kernel` → `weight`，vision norm `scale` → `weight`）
3. **反向转置**：Dense `(in,out)→(out,in)`，Conv3D `(T,H,W,in,out)→(out,in,T,H,W)`，embedding/norm/bias 不转置
4. **分片保存**：超过 `max_shard_size`（默认 5GB）时自动分片，生成 `model.safetensors.index.json`
5. **Packed causal mask**：从 `cu_seqlens`（1D cumsum）构建 segment_ids，然后 `same_segment & causal` 生成 block-diagonal mask
6. **数据 packing 集成**：`data_flatten=True` 或 `data_packing=True` 时使用 `FlattenedDataCollatorForSupervisedDataset`
7. **MetricsLogger**：lazy import wandb/tensorboard，`report_to="none"` 时完全无副作用
8. **训练后导出**：自动导出 HF safetensors + 保存 processor/tokenizer

### 验证结果

- 所有导入 OK（weight_exporter, MetricsLogger, _make_packed_causal_mask, train __init__）
- 零 `import torch` 在 jax_qwenvl/ 中
- Packed causal mask 正确：`cu_seqlens=[0,3,5], seq_len=5` → 正确的 block-diagonal + causal 矩阵
- MetricsLogger no-op 模式无报错
- LoRA 合并正确：`merged = base + (A @ B) * scaling`，LoRA 子键正确移除
- Key 映射正确：18 个测试用例全部通过（lm_head, embed_tokens, visual blocks/merger/ln_post, text layers/norm）
- 反向转置正确：Dense/Conv3D/Embedding/Bias 全部通过
- 权重导出 round-trip：export → reload → compare，28 个参数 max_diff=0.0
- train.py 语法正确（AST parse OK）
- train_tpu.sh 可执行权限正确

## TPU 硬件验证完成记录

**日期**：2026-02-07
**环境**：v6e-4 spot (us-central1-b)，4 chips，runtime=v2-alpha-tpuv6e
**JAX**：0.6.2 + libtpu 0.0.17 + flax 0.10.7 + orbax-checkpoint 0.11.15
**模型**：Qwen/Qwen3-VL-2B-Instruct（2.1B 参数，float32）

### 验证结果：6/8 通过

| 测试 | 结果 | 详情 |
|------|------|------|
| 1. TPU 设备检测 | ✅ PASS | backend=tpu, 4 chips (topology 2x2) |
| 2. 模型权重加载 | ✅ PASS | 2,127,532,032 params, HuggingFace Hub 下载 |
| 3. 前向推理 | ✅ PASS | logits=(1,32,151936), no NaN/Inf, 5.4s |
| 4. SPMD 分片 (DP) | ✅ PASS | dp=4, fsdp=1, 参数复制到 4 chips |
| 5. 单步训练 | ✅ PASS | loss=10.2661, step=1, 29.6s |
| 6. Checkpoint | ❌ FAIL | OOM (float32 replicated 2B 模型占满 HBM) |
| 7. 梯度累积 | ❌ FAIL | OOM (需 32.35G，仅有 31.25G HBM/chip) |
| 8. FSDP 训练 | ✅ PASS | loss=10.2678, params sharded across 4 chips |

### 发现的代码 Bug 及修复

1. **weight_loader.py key mapping**：Qwen3-VL safetensors 使用 `model.language_model.` 前缀（而非 `model.`），导致 text model 权重全部丢失。已修复，支持两种命名。
2. **weight_loader.py embed_tokens**：`embed_tokens` 需映射为直接叶节点（`self.param()` 而非 `nn.Embed`）。
3. **sharding.py FSDP 兼容**：参数第一维不能被 FSDP 设备数整除时（如 Conv3D shape=(2,16,16,3,1024)），自动回退到 replicated。
4. **sharding.py embed_tokens**：`shard_params` 的 kernel 检测增加 `embed_tokens` 匹配。
5. **config.py tie_word_embeddings**：Qwen3-VL-2B-Instruct 的 `tie_word_embeddings=True`，无独立 lm_head。
6. **TPU runtime**：v6e TPU 需使用 `v2-alpha-tpuv6e` runtime（而非 `tpu-ubuntu2204-base`）。

### OOM 分析

float32 2B 模型在 DP 模式下内存使用：
- 参数：~8.4GB（2.1B × 4 bytes）
- Adam 优化器状态：~16.8GB（2 × params）
- 总计：~25GB（replicated per chip）
- v6e HBM/chip：~31.25GB
- 剩余空间不足以容纳 checkpoint restore 或 gradient accumulation 的额外缓冲

**解决方案**（生产环境）：使用 FSDP 分片 + bfloat16 + gradient checkpointing

---

## bfloat16 训练支持完成记录

**日期**：2026-02-07
**环境**：v6e-4 spot (us-central1-b)，4 chips，runtime=v2-alpha-tpuv6e
**JAX**：0.6.2 + libtpu 0.0.17 + flax 0.10.7 + orbax-checkpoint 0.11.15
**模型**：Qwen/Qwen3-VL-2B-Instruct（2.1B 参数，**bfloat16**）

### 验证结果：8/8 通过

| 测试 | 结果 | 详情 |
|------|------|------|
| 1. TPU 设备检测 | ✅ PASS | backend=tpu, 4 chips (topology 2x2) |
| 2. 模型权重加载 (bf16) | ✅ PASS | 2.1B params, dtype=bfloat16, memory=4.26GB |
| 3. 前向推理 (bf16) | ✅ PASS | logits=(1,32,151936), dtype=bfloat16, no NaN/Inf |
| 4. SPMD 分片 (DP) | ✅ PASS | dp=4, fsdp=1, dtype=bfloat16 |
| 5. 单步训练 (bf16) | ✅ PASS | loss=10.1919, opt_dtype=bfloat16, opt_mem=6.88GB |
| 6. Checkpoint | ✅ PASS | max_diff=0.0, save+restore 40s, dtype=bfloat16 |
| 7. 梯度累积 | ✅ PASS | loss=10.1919, accum_steps=2, 54.8s |
| 8. FSDP 训练 (bf16) | ✅ PASS | loss=10.2434, params sharded across 4 chips |

### 内存对比（float32 vs bfloat16，DP 模式 per chip）

| 项目 | float32 | bfloat16 | 节省 |
|------|---------|----------|------|
| 参数 | 8.4GB | 4.2GB | 50% |
| Adam 优化器状态 | 16.8GB | 6.88GB | 59% |
| 总计 | ~25GB | ~11GB | 56% |
| 剩余 HBM (31.25GB/chip) | ~6GB | ~20GB | — |

### 代码修改

1. **`train.py`**：连接 `bf16` 标志到实际 dtype 转换，加载权重后通过 `jax.tree_util.tree_map` 将所有 float 参数从 float32 转为 bfloat16
2. **`qwen3_vl.py`**：在 VisionModel 调用前，将 `pixel_values` 和 `pixel_values_videos` 转为与 `inputs_embeds` 相同的 dtype（确保 vision encoder 全程 bf16）
3. **`tpu_validate.py`**：所有测试使用 bf16 参数；修复 `donate_argnums` 导致的 state buffer 复用问题（返回 `new_state` 而非已 donated 的 `state`）

### 混合精度策略

bfloat16 训练中，以下操作保持 float32 以确保数值稳定性（已在 Layer 1 中实现）：
- `RMSNorm`：variance 计算在 float32，结果 cast 回 bf16
- `Softmax`（attention 和 loss）：在 float32 中计算
- `RoPE`：cos/sin 计算在 float32，结果 cast 回 bf16
- `cross_entropy_loss`：logits 和 log_softmax 在 float32 中计算

---

### TPU 资源创建关键注意事项

#### 1. TPU Runtime 版本选择（最关键）

v6e TPU **必须**使用 `v2-alpha-tpuv6e` runtime，不能使用通用的 `tpu-ubuntu2204-base`：

```bash
# ✅ 正确：v6e 专用 runtime
gcloud compute tpus tpu-vm create qwen3vl-test \
    --zone=us-central1-b \
    --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e \
    --spot

# ❌ 错误：通用 runtime，JAX 无法初始化 TPU backend
gcloud compute tpus tpu-vm create qwen3vl-test \
    --zone=us-central1-b \
    --accelerator-type=v6e-4 \
    --version=tpu-ubuntu2204-base \
    --spot
```

错误 runtime 的症状：
- `/dev/accel*` 设备文件不存在（v6e 使用 `/dev/vfio/0,1,2,3`）
- `jax.devices()` 报错 `Failed to get global TPU topology`
- TPU runtime Docker 容器运行在 8470 端口但 JAX 无法连接

#### 2. 网络和防火墙配置

如果 TPU VM 在自定义 VPC 中（而非 default），需要：

```bash
# 指定网络和子网
gcloud compute tpus tpu-vm create qwen3vl-test \
    --zone=us-central1-b \
    --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e \
    --spot \
    --network=dify-vpc \
    --subnetwork=dify-subnet

# 如果 SSH 超时，检查防火墙规则
# 方式 1：使用 IAP tunnel（推荐）
gcloud compute tpus tpu-vm ssh qwen3vl-test --zone=us-central1-b --tunnel-through-iap

# 方式 2：添加临时防火墙规则（测试用）
gcloud compute firewall-rules create allow-tpu-ssh-test \
    --network=dify-vpc \
    --allow=tcp:22 \
    --source-ranges=<your-ip>/32 \
    --target-tags=<tpu-network-tag>
```

#### 3. 代码上传和环境准备

```bash
# 打包代码（排除大文件）
cd /home/greg_greghuang_altostrat_com/Qwen3-VL
tar czf /tmp/jax_qwenvl.tar.gz jax_qwenvl/

# 上传到 TPU VM
gcloud compute tpus tpu-vm scp /tmp/jax_qwenvl.tar.gz qwen3vl-test:~ \
    --zone=us-central1-b

# 解压
gcloud compute tpus tpu-vm ssh qwen3vl-test --zone=us-central1-b \
    --command="tar xzf jax_qwenvl.tar.gz"
```

#### 4. 清理资源

```bash
# 删除 TPU VM（spot 实例也建议手动清理）
gcloud compute tpus tpu-vm delete qwen3vl-test --zone=us-central1-b --quiet

# 清理临时防火墙规则
gcloud compute firewall-rules delete allow-tpu-ssh-test --quiet
```

### Package 依赖兼容性关键记录

#### 1. JAX TPU 安装

```bash
# v6e TPU 上安装 JAX（截至 2026-02-07 验证通过的版本）
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# 安装结果：jax==0.6.2 + jaxlib==0.6.2 + libtpu==0.0.17
```

#### 2. orbax-checkpoint 版本兼容性（踩坑重点）

与 JAX 0.6.2 兼容的 orbax-checkpoint 版本**非常有限**，实测如下：

| orbax-checkpoint 版本 | 兼容性 | 错误信息 |
|---|---|---|
| **0.11.15** | ✅ 可用 | — |
| 0.11.32 (最新) | ❌ 不可用 | `jax.sharding.set_mesh` 不是 context manager |
| 0.10.0 | ❌ 不可用 | `jax._src.config.enable_memories` 缺失 |
| 0.9.1 | ❌ 不可用 | `jax.lib.xla_extension.XlaRuntimeError` 已移除 |

```bash
# 安装兼容版本
pip install orbax-checkpoint==0.11.15
```

#### 3. 其他依赖安装

```bash
# 核心依赖（一条命令安装）
pip install flax optax orbax-checkpoint==0.11.15 safetensors transformers huggingface_hub

# 可选：视觉处理（如果需要图像/视频测试）
pip install qwen-vl-utils
```

#### 4. 完整的已验证依赖版本快照

```
jax==0.6.2
jaxlib==0.6.2
libtpu==0.0.17
flax==0.10.7
optax==0.2.5 (或更高)
orbax-checkpoint==0.11.15
safetensors==0.5.x
transformers==4.51.x (或更高)
huggingface_hub==0.30.x
```

#### 5. 常见问题排查

| 问题 | 症状 | 解决方案 |
|---|---|---|
| JAX 无法检测 TPU | `Failed to get global TPU topology` | 使用 `v2-alpha-tpuv6e` runtime 重建 VM |
| orbax checkpoint 崩溃 | `set_mesh` / `enable_memories` 错误 | 降级到 `orbax-checkpoint==0.11.15` |
| HuggingFace 下载失败 | 网络超时或 401 | 设置 `HF_TOKEN` 环境变量或检查网络 |
| OOM（float32 2B 模型） | `RESOURCE_EXHAUSTED` | 使用 FSDP + bfloat16 + gradient checkpointing |
| SSH 连接超时 | `Connection timed out` | 检查防火墙规则或使用 `--tunnel-through-iap` |
| Checkpoint async 错误 | `Array has been deleted` | 使用 `enable_async_checkpointing=False` |

### 下一步：Layer 4+

可能的后续工作：
- MoE 模型支持（Expert Parallelism）
- 性能调优（XLA 编译优化、通信与计算重叠）
- Qwen2.5-VL 支持
- 推理/生成模式
- bfloat16 训练支持（减少内存占用）

---

### 参考资源

- MaxText (Google JAX LLM 训练框架): https://github.com/google/maxtext
- Optax (JAX 优化器库): https://github.com/google-deepmind/optax
- Orbax (JAX checkpoint 管理): https://github.com/google/orbax
- Grain (JAX 数据加载库): https://github.com/google/grain
- Pallas (JAX 自定义 TPU kernel): https://jax.readthedocs.io/en/latest/pallas/
- Flax NNX: https://flax.readthedocs.io/en/latest/
