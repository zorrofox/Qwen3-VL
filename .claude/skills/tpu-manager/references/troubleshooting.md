# TPU Troubleshooting

## Table of Contents

1. [JAX Cannot Detect TPU](#jax-cannot-detect-tpu)
2. [OOM Errors](#oom-errors)
3. [SSH Connection Issues](#ssh-connection-issues)
4. [Checkpoint Errors](#checkpoint-errors)
5. [XLA Recompilation](#xla-recompilation)
6. [Training Numerical Issues](#training-numerical-issues)
7. [Dependency Conflicts](#dependency-conflicts)
8. [Multi-Host Issues](#multi-host-issues)

## JAX Cannot Detect TPU

### Symptom
```
Failed to get global TPU topology
```
Or `jax.devices()` returns only CPU devices.

### Cause
Wrong TPU runtime version.

### Fix
Recreate the VM with the correct runtime:
```bash
# v6e requires v2-alpha-tpuv6e
gcloud compute tpus tpu-vm delete VM_NAME --zone=ZONE --quiet
gcloud compute tpus tpu-vm create VM_NAME \
    --zone=ZONE --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e --spot
```

### Verify
```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE \
    --command='python3 -c "import jax; print(jax.devices()); print(jax.device_count())"'
```

## OOM Errors

### Symptom
```
RESOURCE_EXHAUSTED: Out of memory
```

### Common causes and fixes

| Cause | Fix |
|-------|-----|
| float32 model too large | Use bfloat16: `--dtype bfloat16` |
| Sequence length too long | Reduce `model_max_length` (e.g., 1024) |
| Batch size too large | Reduce `per_device_train_batch_size` |
| No gradient checkpointing | Enable: `--gradient_checkpointing True` |
| Vision attention matrix too large | Reduce `--max_pixels` (e.g., 50176) |

### Memory budget (v6e, 31.25 GB HBM/chip)

| Config | Params | Optimizer | Activations | Total |
|--------|--------|-----------|-------------|-------|
| 2B fp32, DP | 8.4 GB | 16.8 GB | ~6 GB | ~31 GB (borderline) |
| 2B bf16, DP | 4.2 GB | 6.9 GB | ~3 GB | ~14 GB (safe) |
| 2B bf16, FSDP-4 | 1.1 GB | 1.7 GB | ~3 GB | ~6 GB (safe) |

### Vision attention OOM

The vision model processes ALL patches from the global batch on each chip (not DP-sharded). Attention matrix size = `(num_heads, N, N)` where N = total patches.

```
max_pixels=451584 + batch=64 → N=112896 → attention bf16[16,112896,112896] = 406 GB → OOM
max_pixels=50176  + batch=64 → N=12544  → attention bf16[16,12544,12544]   = ~5 GB  → OK
```

Fix: reduce `--max_pixels` (e.g., 50176).

## SSH Connection Issues

### Symptom
```
Connection timed out
```

### Fixes

1. **Use IAP tunnel** (works regardless of firewall):
   ```bash
   gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --tunnel-through-iap
   ```

2. **Add firewall rule** (if using custom VPC):
   ```bash
   gcloud compute firewall-rules create allow-tpu-ssh \
       --network=VPC_NAME --allow=tcp:22 --source-ranges=0.0.0.0/0
   ```

3. **Check TPU state**:
   ```bash
   gcloud compute tpus tpu-vm describe VM_NAME --zone=ZONE
   ```
   State should be `READY`. If `CREATING` or `REPAIRING`, wait.

## Checkpoint Errors

### "Checkpoint path should be absolute"

```
ValueError: Checkpoint path should be absolute. Got output/...
```

**Fix**: Use absolute path for output directory:
```python
output_dir = os.path.abspath(output_dir)
```

### "Array has been deleted"

Async checkpoint race condition.

**Fix**: Disable async checkpointing:
```python
CheckpointManager(directory, options=CheckpointManagerOptions(
    enable_async_checkpointing=False
))
```

### "set_mesh is not a context manager"

Wrong orbax-checkpoint version.

**Fix**: Pin to 0.11.15:
```bash
pip install orbax-checkpoint==0.11.15
```

### "enable_memories" / "XlaRuntimeError" import errors

Wrong orbax-checkpoint version (too old).

**Fix**: Pin to 0.11.15.

## XLA Recompilation

### Symptom
Step time does not decrease after step 1-2. Every step takes 60-100+ seconds.

### Cause
Input tensor shapes vary between batches. XLA compiles a new computation graph for each unique shape combination.

### Fix
Pad ALL input tensors to fixed shapes in the DataCollator:

**Text tensors**: Pad to `model_max_length` (not tokenizer default which may be 262144)
```python
# Use training_args.model_max_length, NOT tokenizer.model_max_length
pad_to = training_args.model_max_length  # e.g., 1024
```

**Vision tensors**: Pad to computed maximums:
```python
max_patches_per_image = max_pixels // (patch_size ** 2)
max_total_patches = batch_size * max_patches_per_image
max_num_images = batch_size  # assume max 1 image per sample
```

### Verify fix
After fixing, step times should drop dramatically:
- Step 1: ~80s (XLA compilation, expected)
- Step 2: ~80s (second trace, expected)
- Step 3+: <1s (no recompilation)

## Training Numerical Issues

### NaN/Inf in loss

Ensure mixed-precision operations use float32 where needed:

| Operation | Must be float32 |
|-----------|-----------------|
| RMSNorm variance | Yes |
| Softmax (attention) | Yes |
| Softmax (cross-entropy) | Yes |
| RoPE cos/sin | Yes |
| All other matmuls | bfloat16 OK |

### Loss not decreasing

Check:
1. Learning rate not too high (try 1e-5 for fine-tuning)
2. Vision encoder weights loaded correctly (check weight_loader key mapping)
3. Labels masked correctly (padding tokens should be -100)

## Dependency Conflicts

### orbax-checkpoint version matrix (JAX 0.6.2)

| Version | Status | Error |
|---------|--------|-------|
| **0.11.15** | Works | — |
| 0.11.32+ | Broken | `set_mesh` not context manager |
| 0.10.x | Broken | `enable_memories` missing |
| 0.9.x | Broken | `XlaRuntimeError` removed |

### HuggingFace download failures

```
401 Client Error: Unauthorized
```

**Fix**: Set `HF_TOKEN` environment variable:
```bash
export HF_TOKEN=hf_your_token_here
```

### tokenizer model_max_length too large

Qwen3-VL tokenizer has `model_max_length=262144` (256K). Using this for padding will hang compilation or OOM.

**Fix**: Override with training argument:
```bash
--model_max_length 1024
```

## Multi-Host Issues

### Training hangs at startup

All workers must run the training script. If using `--worker=all`, check all workers started:

```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=0 --command='ps aux | grep python'
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=1 --command='ps aux | grep python'
```

### One worker exits, others hang

This is expected. JAX multi-host operations are collective — if one process crashes, others wait forever. Kill all and restart:

```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all \
    --command='pkill -f python3'
```

### Spot preemption

```
The TPU was preempted
```

The VM is deleted. Recreate and resume from checkpoint:
```bash
gcloud compute tpus tpu-vm create VM_NAME \
    --zone=ZONE --accelerator-type=v6e-16 \
    --version=v2-alpha-tpuv6e --spot
# Re-upload code, install deps, resume with --resume_from_checkpoint
```
