# Multi-Host TPU Pod Slices

## Table of Contents

1. [Overview](#overview)
2. [Creating Pod Slices](#creating-pod-slices)
3. [Running Commands on All Workers](#running-commands-on-all-workers)
4. [JAX Distributed Initialization](#jax-distributed-initialization)
5. [Batch Sharding](#batch-sharding)
6. [Checkpointing](#checkpointing)
7. [Logging](#logging)
8. [Single-Host Compatibility](#single-host-compatibility)

## Overview

Multi-host TPU pod slices have multiple VMs (hosts), each with 4 TPU chips. All hosts must run the training script simultaneously for distributed JAX to work.

| Accelerator | Chips | Hosts | Topology |
|-------------|-------|-------|----------|
| v6e-4 | 4 | 1 | Single-host |
| v6e-16 | 16 | 4 | 4 hosts x 4 chips |
| v6e-32 | 32 | 8 | 8 hosts x 4 chips |
| v6e-64 | 64 | 16 | 16 hosts x 4 chips |

**Important**: Worker ordering in `gcloud` (worker 0,1,2,3) does NOT correspond to JAX process ordering. JAX process 0 may be on any gcloud worker.

## Creating Pod Slices

```bash
gcloud compute tpus tpu-vm create VM_NAME \
    --zone=ZONE \
    --accelerator-type=v6e-16 \
    --version=v2-alpha-tpuv6e \
    --spot
```

## Running Commands on All Workers

All multi-host operations use `--worker=all`:

```bash
# Run command on all workers
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all --command='CMD'

# Upload files to all workers
gcloud compute tpus tpu-vm scp FILE VM_NAME:~/PATH --zone=ZONE --worker=all

# SSH to a specific worker (e.g., worker 0)
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=0
```

### Launch training on all workers

```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all --command='
cd ~/PROJECT_DIR && python3 -m module.train \
    --model_name_or_path MODEL \
    --output_dir /absolute/path/to/output \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing True
'
```

**Critical**: Use absolute paths for `--output_dir`. Orbax rejects relative paths in multi-host mode:
```
ValueError: Checkpoint path should be absolute. Got output/...
```

## JAX Distributed Initialization

Multi-host JAX requires explicit distributed initialization before any JAX operation:

```python
# Must be called before jax.devices(), model creation, etc.
try:
    jax.distributed.initialize()
except Exception as e:
    # Single-host: no coordinator available, safe to skip
    logger.warning("jax.distributed.initialize() skipped: %s", e)
```

After initialization:
- `jax.process_count()` returns total number of hosts (e.g., 4 for v6e-16)
- `jax.process_index()` returns this host's index (0 to N-1)
- `jax.device_count()` returns total chips across all hosts (e.g., 16)
- `jax.local_device_count()` returns chips on this host (e.g., 4)

## Batch Sharding

In multi-host mode, each host prepares its own local batch from the DataLoader. The local batches must be assembled into a global array:

```python
from jax.experimental.multihost_utils import host_local_array_to_global_array

if jax.process_count() > 1:
    # Multi-host: each host has local_batch, assemble into global
    global_array = host_local_array_to_global_array(local_array, mesh, sharding)
else:
    # Single-host: just device_put with sharding
    global_array = jax.device_put(local_array, sharding)
```

Auto-detect multi-host via `jax.process_count() > 1`.

## Checkpointing

### Requirements

1. **Absolute paths only** — Orbax rejects relative paths in multi-host mode
2. **All hosts participate** — Orbax checkpoint is a collective operation
3. **`jax.distributed.initialize()` must be called first**

### Guard saves to main process only

```python
is_main_process = jax.process_index() == 0

if is_main_process and step % save_steps == 0:
    ckpt_manager.save(step, train_state)
```

### Async checkpoint caveat

Async checkpointing (`enable_async_checkpointing=True`) may cause `Array has been deleted` errors. Use `enable_async_checkpointing=False` if this occurs.

## Logging

Guard all logging/saving to main process to avoid duplicate output:

```python
is_main_process = jax.process_index() == 0

# Logging
if is_main_process:
    logger.info("Step %d, loss=%.4f", step, loss)
    metrics_logger.log({"loss": loss}, step=step)

# Weight export
if is_main_process:
    export_weights(params, output_dir)
```

### Tensorboard to GCS

`torch.utils.tensorboard.SummaryWriter` cannot append to GCS paths. Write to a local temp directory and sync to GCS after training:

```python
import tempfile, subprocess

if tb_dir.startswith("gs://"):
    gcs_dir = tb_dir
    local_dir = tempfile.mkdtemp(prefix="tb_logs_")
    writer = SummaryWriter(log_dir=local_dir)
    # ... training ...
    # On finish:
    subprocess.run(["gsutil", "-m", "cp", "-r", f"{local_dir}/*", gcs_dir])
```

## Single-Host Compatibility

All multi-host code must be backward compatible with single-host. Patterns:

| Feature | Multi-host guard |
|---------|-----------------|
| `jax.distributed.initialize()` | Wrap in try-except |
| Batch sharding | `if process_count() > 1` auto-detect |
| Logging | `if is_main_process` (always True on single-host) |
| Checkpoint | Same code works for both |
| MetricsLogger | Non-main process uses `report_to="none"` |
