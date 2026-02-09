# Distribution: DeepSpeed/DDP → JAX SPMD

## Table of Contents

1. [Concepts](#concepts)
2. [Device Mesh](#device-mesh)
3. [Parameter Sharding](#parameter-sharding)
4. [Batch Sharding](#batch-sharding)
5. [Multi-Host (Pod Slices)](#multi-host)
6. [Gotchas](#gotchas)

## Concepts

| PyTorch | JAX SPMD |
|---------|----------|
| `torch.distributed.init_process_group` | `jax.distributed.initialize()` |
| DeepSpeed ZeRO-2 (optimizer sharding) | FSDP with `P('fsdp', None)` |
| DeepSpeed ZeRO-3 (param+optimizer sharding) | FSDP with `P('fsdp', None)` |
| DDP (all-reduce gradients) | DP with `P()` (replicated) |
| `model.to(device)` | `jax.device_put(params, sharding)` |
| Explicit `all_reduce` / `broadcast` | Implicit via SPMD annotations |

JAX SPMD key idea: annotate HOW data is distributed, JAX inserts communication automatically.

## Device Mesh

### PyTorch (no explicit mesh)

```python
torch.distributed.init_process_group(backend="nccl")
model = DDP(model, device_ids=[local_rank])
```

### JAX (explicit mesh)

```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def create_device_mesh(dp=-1, fsdp=1, tp=1):
    devices = jax.devices()
    num_devices = len(devices)
    if dp == -1:
        dp = num_devices // (fsdp * tp)
    device_array = np.array(devices).reshape(dp, fsdp, tp)
    return Mesh(device_array, axis_names=('dp', 'fsdp', 'tp'))
```

Axis meanings:
- `dp`: data parallel — batch is split, params replicated
- `fsdp`: fully-sharded data parallel — params split across devices
- `tp`: tensor parallel — individual tensors split (reserved for future)

Examples:
```python
# 4 chips, pure DP: dp=4, fsdp=1
mesh = create_device_mesh(dp=4, fsdp=1)  # 4x1x1

# 4 chips, FSDP: dp=1, fsdp=4
mesh = create_device_mesh(dp=1, fsdp=4)  # 1x4x1

# 16 chips, DP+FSDP: dp=4, fsdp=4
mesh = create_device_mesh(dp=4, fsdp=4)  # 4x4x1
```

## Parameter Sharding

### DP mode (replicated)

```python
# All parameters fully replicated on every device
param_sharding = NamedSharding(mesh, P())  # P() = replicate all axes

# Apply to all params
sharded_params = jax.device_put(params, param_sharding)
```

### FSDP mode (shard first axis)

```python
def shard_params(params, mesh, mode='dp'):
    if mode == 'fsdp':
        fsdp_sharding = NamedSharding(mesh, P('fsdp', None))
        replicated = NamedSharding(mesh, P())

    def _shard_leaf(path_str, param):
        if mode == 'dp':
            return jax.device_put(param, NamedSharding(mesh, P()))
        # FSDP: shard 2D+ kernels on first axis
        if param.ndim >= 2 and "kernel" in path_str:
            if param.shape[0] % fsdp_size == 0:
                return jax.device_put(param, fsdp_sharding)
        # Fallback: replicate
        return jax.device_put(param, replicated)

    return jax.tree_util.tree_map_with_path(
        lambda path, p: _shard_leaf(str(path), p), params
    )
```

Key rules:
- **kernel (2D)**: shard first axis if divisible by FSDP size
- **embedding**: shard first axis (vocab dimension)
- **bias, norm, 1D params**: always replicate
- **Conv3D kernel (5D)**: first axis may not be divisible (e.g., temporal_patch_size=2) — replicate

## Batch Sharding

### Single-host

```python
def shard_batch(batch, mesh):
    dp_sharding = NamedSharding(mesh, P('dp'))
    dp_none = NamedSharding(mesh, P('dp', None))
    none_dp_none = NamedSharding(mesh, P(None, 'dp', None))

    for name, tensor in batch._asdict().items():
        if tensor is None:
            continue
        if name == "position_ids":
            # Shape (3, B, L) — batch axis is dim=1
            sharded = jax.device_put(tensor, none_dp_none)
        elif tensor.ndim >= 2:
            # Shape (B, ...) — batch axis is dim=0
            sharded = jax.device_put(tensor, dp_none)
        elif tensor.ndim == 1:
            sharded = jax.device_put(tensor, dp_sharding)
        # ...
```

### Multi-host

```python
from jax.experimental.multihost_utils import host_local_array_to_global_array

def shard_batch_multihost(batch, mesh):
    process_idx = jax.process_index()
    num_processes = jax.process_count()

    for name, tensor in batch._asdict().items():
        # Each host has the FULL batch — slice to local shard
        B = tensor.shape[batch_axis]
        local_B = B // num_processes
        start = process_idx * local_B
        local_tensor = tensor[start : start + local_B]  # (adjust axis for position_ids)

        # Assemble global array from all hosts' local shards
        sharded = host_local_array_to_global_array(
            local_tensor, mesh, partition_spec
        )
```

Auto-detect mode:

```python
def shard_batch(batch, mesh):
    if jax.process_count() > 1:
        return _shard_batch_multihost(batch, mesh)
    return _shard_batch_single(batch, mesh)
```

## Multi-Host

### Initialization

```python
# MUST be called before any JAX operation on multi-host
try:
    jax.distributed.initialize()
except Exception as e:
    # Single-host: silently skip
    logger.warning("Skipped distributed init (single-host): %s", e)
```

### Running on all workers

```bash
# All workers must run simultaneously
gcloud compute tpus tpu-vm ssh VM --zone=ZONE --worker=all \
    --command='python3 train.py --output_dir /abs/path/output'
```

### Guard main-process operations

```python
is_main_process = jax.process_index() == 0

# Only main process logs, saves, exports
if is_main_process:
    logger.info("Step %d, loss %.4f", step, loss)
    checkpoint_manager.save(step, state)
    export_weights(params, output_dir)

# MetricsLogger: non-main uses report_to="none"
effective_report_to = report_to if is_main_process else "none"
metrics_logger = MetricsLogger(output_dir, report_to=effective_report_to)
```

### Absolute paths required

```python
# Orbax checkpoint fails with relative paths in multi-host mode
output_dir = os.path.abspath(output_dir)
```

## Gotchas

1. **Worker 0 != JAX process 0**: On TPU pod slices, `gcloud --worker=0` is NOT necessarily `jax.process_index() == 0`. JAX assigns process indices internally.

2. **FSDP divisibility**: If param's first axis is not divisible by FSDP device count (e.g., Conv3D kernel `(2,16,16,3,1024)`), it MUST be replicated. Auto-fallback is essential.

3. **Collective operations**: Every JAX operation is implicitly collective in SPMD. If one host skips an operation (e.g., conditional save), other hosts will hang waiting for the collective.

4. **Vision model not sharded**: In this codebase, the vision model is replicated across all devices. Each device processes ALL patches from the global batch. This wastes compute but simplifies sharding.

5. **Batch must be identically shaped across hosts**: For `host_local_array_to_global_array`, each host's local shard must have the same shape. Fixed-shape padding ensures this.

6. **position_ids special shape**: `(3, B, L)` has batch on axis=1, not axis=0. Sharding spec must be `P(None, 'dp', None)`, not `P('dp', None, None)`.
