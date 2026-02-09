# Training Loop Conversion

## Table of Contents

1. [Loss Function](#loss-function)
2. [Training Step](#training-step)
3. [Gradient Accumulation](#gradient-accumulation)
4. [Optimizer (Multi-Group LR)](#optimizer)
5. [Learning Rate Schedule](#learning-rate-schedule)
6. [TrainState](#trainstate)
7. [Gotchas](#gotchas)

## Loss Function

### PyTorch

```python
loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='mean')
```

### JAX (manual implementation)

```python
def cross_entropy_loss(logits, labels, ignore_index=-100):
    valid_mask = (labels != ignore_index).astype(jnp.float32)
    safe_labels = jnp.where(labels != ignore_index, labels, 0)
    # log_softmax in float32 for stability
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    nll = -jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    return (nll * valid_mask).sum() / jnp.maximum(valid_mask.sum(), 1.0)
```

Key points:
- Replace masked labels with 0 before indexing (avoids out-of-bounds)
- Float32 for `log_softmax` prevents overflow
- Manual mean over valid tokens only

## Training Step

### PyTorch (HuggingFace Trainer style)

```python
def training_step(self, batch):
    self.model.train()
    outputs = self.model(**batch)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
    self.optimizer.step()
    self.scheduler.step()
    self.optimizer.zero_grad()
    return loss.item()
```

### JAX

```python
@partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    def loss_fn(params):
        logits, loss = state.apply_fn(
            {"params": params},
            input_ids=batch.input_ids,
            pixel_values=batch.pixel_values,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            position_ids=batch.position_ids,
            image_grid_thw=batch.image_grid_thw,
            # ... other fields
        )
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    # Gradient clipping built into optax chain
    new_state = state.apply_gradients(grads=grads)
    return new_state, {"loss": loss, "step": state.step}
```

Key differences:
- `@jax.jit` compiles entire step for TPU
- `donate_argnums=(0,)` allows JAX to reuse `state` memory (freed after use)
- `jax.value_and_grad` computes loss AND gradients in one call
- Returns new state (immutable); no `optimizer.zero_grad()`
- Gradient clipping is part of the optax optimizer chain, not a separate call

## Gradient Accumulation

### PyTorch

```python
for i, micro_batch in enumerate(micro_batches):
    loss = model(micro_batch).loss / num_accum
    loss.backward()  # accumulates into .grad
if (i + 1) % num_accum == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### JAX (jax.lax.scan)

```python
@partial(jax.jit, donate_argnums=(0, 1))
def train_step_with_accumulation(state, micro_batches, num_accum):
    def micro_step(carry, micro_batch):
        acc_grads, acc_loss = carry

        def loss_fn(params):
            _, loss = state.apply_fn({"params": params}, ...)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        acc_grads = jax.tree_util.tree_map(lambda a, g: a + g, acc_grads, grads)
        acc_loss = acc_loss + loss
        return (acc_grads, acc_loss), None

    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    (total_grads, total_loss), _ = jax.lax.scan(
        micro_step, (zero_grads, jnp.float32(0.0)), micro_batches
    )

    avg_grads = jax.tree_util.tree_map(lambda g: g / num_accum, total_grads)
    new_state = state.apply_gradients(grads=avg_grads)
    return new_state, {"loss": total_loss / num_accum}
```

Key points:
- `jax.lax.scan` replaces Python loop — compiled into XLA, no retracing
- `micro_batches` must be stacked as `(num_accum, B, ...)` before calling
- Zero-initialize gradient accumulator via `tree_map(jnp.zeros_like, ...)`
- Average gradients after accumulation, not divide loss

## Optimizer

### PyTorch (parameter groups)

```python
param_groups = [
    {"params": vision_params, "lr": 1e-5, "weight_decay": 0.01},
    {"params": text_decay_params, "lr": 2e-5, "weight_decay": 0.01},
    {"params": text_nodecay_params, "lr": 2e-5, "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(param_groups)
```

### JAX/Optax (label tree + multi_transform)

```python
def create_optimizer(params, lr, vision_lr=None, wd=0.01, ...):
    # 1. Define per-group optimizer
    transforms = {
        "frozen": optax.set_to_zero(),
        "llm_decay": optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(lr, weight_decay=wd),
        ),
        "llm_nodecay": optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(lr, weight_decay=0.0),
        ),
        "vision_decay": optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(vision_lr or lr, weight_decay=wd),
        ),
        # ... more groups
    }

    # 2. Label each parameter
    def label_fn(path, leaf):
        path_str = "/".join(str(p) for p in path)
        if "vision" in path_str:
            if leaf.ndim >= 2:
                return "vision_decay"
            return "vision_nodecay"
        if leaf.ndim < 2:  # bias, norm, embedding
            return "llm_nodecay"
        return "llm_decay"

    label_tree = jax.tree_util.tree_map_with_path(label_fn, params)

    # 3. Create multi-transform optimizer
    optimizer = optax.multi_transform(transforms, label_tree)
    return optimizer
```

Key differences:
- No param groups list — parameter tree is labeled in parallel
- `optax.multi_transform` routes each parameter to its optimizer
- Weight decay 0 for bias, norm, embedding (via separate label)
- Frozen parameters use `optax.set_to_zero()`
- Gradient clipping is part of the chain, not a separate call

## Learning Rate Schedule

### PyTorch

```python
warmup = LinearLR(optimizer, start_factor=0.0, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, [warmup, cosine], [warmup_steps])
```

### Optax

```python
def create_schedule(peak_lr, warmup_steps, total_steps):
    warmup = optax.linear_schedule(0.0, peak_lr, warmup_steps)
    decay = optax.cosine_decay_schedule(peak_lr, max(total_steps - warmup_steps, 1))
    return optax.join_schedules([warmup, decay], [warmup_steps])

# Usage: pass schedule as learning_rate to adamw
optimizer = optax.adamw(learning_rate=schedule, ...)
```

Key difference: schedule is a function `step → lr`, passed directly to optimizer.

## TrainState

### PyTorch (mutable objects)

```python
model = MyModel()
optimizer = AdamW(model.parameters())
# State is implicit: model.parameters(), optimizer.state_dict()
```

### Flax (immutable TrainState)

```python
from flax.training.train_state import TrainState

state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer,
)
# state.step, state.params, state.opt_state — all immutable
# state = state.apply_gradients(grads=grads) creates NEW state
```

Extend for extra fields:

```python
class CustomTrainState(TrainState):
    epoch: int = 0
    global_step: int = 0
```

## Gotchas

1. **donate_argnums**: Mark state and batch as donatable to free memory early. Order matters — `(0,)` for state only, `(0, 1)` for state + batch.

2. **Loss must be scalar**: `jax.value_and_grad` expects a scalar loss. If model returns `(logits, loss)`, extract loss in `loss_fn`.

3. **No Python control flow in JIT**: Inside `@jax.jit`, can't use Python `if/for` with JAX values. Use `jax.lax.cond`, `jax.lax.scan` instead.

4. **Optimizer state dtype**: Optax Adam stores `mu` and `nu` in the same dtype as params. For bfloat16 training, optimizer states are also bfloat16 (~60% memory savings vs float32).

5. **Gradient clipping location**: In PyTorch, `clip_grad_norm_` is called after `.backward()`. In Optax, `clip_by_global_norm` is part of the chain BEFORE `adamw`.

6. **Step counting**: `state.step` is auto-incremented by `apply_gradients`. No manual `step += 1`.
