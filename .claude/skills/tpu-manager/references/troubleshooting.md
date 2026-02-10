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

### Symptom: Connection timed out
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

### Symptom: "exited with return code [255]" in background/subagent

```
Failed to execute command on multiple workers. This may have happened if you have not
added your SSH key to your ssh-agent using "ssh-add ~/.ssh/google_compute_engine".
SSH command error: [/usr/bin/ssh] exited with return code [255].
```

**Cause**: Background processes (subagents, `&` commands, screen/tmux sessions) do NOT inherit the parent shell's `ssh-agent` context. The `gcloud compute tpus tpu-vm ssh` command internally invokes `/usr/bin/ssh`, which needs access to the SSH key via the agent.

**Fix**: Always run `gcloud compute tpus tpu-vm ssh` commands in the **foreground** of the main shell session — never in background subprocesses or detached agents. If you must run SSH commands from a background process:

```bash
# Option 1: Add key to agent explicitly before the command
eval $(ssh-agent) && ssh-add ~/.ssh/google_compute_engine
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all --command='CMD'

# Option 2: Use SSH_AUTH_SOCK from the parent process
SSH_AUTH_SOCK=/tmp/ssh-XXXXX/agent.XXXXX gcloud compute tpus tpu-vm ssh ...
```

**For Claude Code specifically**: When using the Task tool or background bash commands to run gcloud SSH, the subagent/subprocess does NOT have access to `ssh-agent`. Run gcloud SSH commands in the main conversation's Bash tool (foreground), not via Task subagents.

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

### Orbax multi-host deadlock (CheckpointManager or StandardCheckpointer)

**Symptom**: Training hangs at checkpoint save. Logs show:
```
Waiting for Save Finalize thread (save_finalize) to complete
```
Or `sync_global_devices name mismatch` assertion error, or `cannot schedule new futures after shutdown`.

**Root cause**: Orbax 0.11.15 requires a **shared filesystem** (GCS or NFS) visible to all hosts for multi-host checkpointing. When `output_dir` is a local path (e.g., `$HOME/output`), only the host that writes can see it. Orbax's internal barrier and finalization logic deadlocks because other hosts cannot access the checkpoint directory.

Both `ocp.CheckpointManager` and `ocp.StandardCheckpointer` are multi-host aware and internally call `sync_global_devices`. Calling them from only process 0 causes barrier mismatches.

**Fix options**:

1. **Recommended (long-term)**: Use GCS path directly for checkpoint directory (MaxText approach):
   ```python
   # All hosts participate, Orbax handles coordination
   ckpt_manager = ocp.CheckpointManager("gs://bucket/checkpoints", options=options)
   ```

2. **Workaround (current)**: Bypass Orbax entirely for multi-host DP mode. Save from process 0 only using `jax.device_get()` + `flax.serialization.to_bytes()`:
   ```python
   if jax.process_index() == 0:
       state_host = jax.device_get(state)
       state_bytes = flax.serialization.to_bytes(state_host)
       with open("state.msgpack", "wb") as f:
           f.write(state_bytes)
   # No barrier needed — next train_step collective provides implicit sync
   ```
   This only works in DP mode (all hosts have full parameter copies).

### Stale `.tmp` checkpoint directories

**Symptom**: `[Errno 39] Directory not empty: '/path/100.tmp' -> '/path/100'` on final checkpoint save.

**Cause**: If the same step was already saved during the training loop, the final forced save tries `os.rename(tmp_path, path)` but the target directory already exists.

**Fix**: Remove existing directory before rename:
```python
if os.path.exists(path):
    shutil.rmtree(path)
os.rename(tmp_path, path)
```

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

## Cleaning Up After Failed Training

When a multi-host training run crashes or is interrupted, stale processes and files can accumulate and cause problems for subsequent runs. Follow this checklist:

### 1. Kill all stale Python processes on ALL workers

**Critical**: Always kill on ALL workers, not just one. Leftover processes on any worker will hold TPU devices and block new runs.

```bash
# Kill all python processes on all workers
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all \
    --command='pkill -9 -f python3 || true'
```

**Verify cleanup**:
```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all \
    --command='ps aux | grep python | grep -v grep | wc -l'
```
Expected output: `0` on all workers.

**Important**: If you run `--worker=all` SSH and one worker fails, the other workers still received the command. Don't assume nothing happened. Check each worker individually if needed:
```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=0 --command='ps aux | grep python'
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=1 --command='ps aux | grep python'
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=2 --command='ps aux | grep python'
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=3 --command='ps aux | grep python'
```

### 2. Clean stale checkpoint temporary directories

Failed saves leave behind `.tmp` directories that interfere with subsequent saves:

```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=0 \
    --command='rm -rf $HOME/output/*.tmp $HOME/output/.orbax-checkpoint-tmp-*'
```

Also check for Orbax temporary metadata:
```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=0 \
    --command='ls -la $HOME/output/'
```

### 3. Clean stale JAX distributed lock files

If `jax.distributed.initialize()` crashes, it may leave a lock or port in use:

```bash
# Check if the coordinator port (default 8476) is still occupied
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=0 \
    --command='ss -tlnp | grep 8476'
```

If the port is occupied by a stale process, kill it (step 1 above should handle this).

### 4. Restart order for multi-host

After cleanup, re-launch training in a single `--worker=all` command. Do NOT try to start workers one at a time — all workers must begin simultaneously for `jax.distributed.initialize()` to succeed.

```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all --command='
cd ~/PROJECT && python3 -m module.train --output_dir $HOME/output ...
'
```

### 5. Common pitfall: accumulating background processes

If you repeatedly launch training via `--worker=all --command='...'` without killing previous runs first, you get multiple Python processes per worker. Each holds TPU devices, causing:
- `RESOURCE_EXHAUSTED` errors (TPU memory fully occupied by stale process)
- `jax.distributed.initialize()` hangs (port 8476 already bound by previous run)
- New training appears to start but immediately OOMs

**Rule**: Always run cleanup (step 1) before launching a new training run if the previous run didn't exit cleanly.

### 6. SSH failures during cleanup (exit code 255)

If `gcloud compute tpus tpu-vm ssh --worker=all` fails with exit code 255 on some workers:

1. **Try `--tunnel-through-iap`** — direct SSH may be blocked but IAP tunnel still works
2. **Try individual workers** — `--worker=0`, `--worker=1`, etc. One may succeed where `--worker=all` fails
3. **Check TPU state** — the TPU may have been preempted or is being repaired:
   ```bash
   gcloud compute tpus tpu-vm describe VM_NAME --zone=ZONE --format='value(state)'
   ```
4. **As last resort, delete and recreate** — if you cannot SSH in at all, the TPU may be in a bad state:
   ```bash
   gcloud compute tpus tpu-vm delete VM_NAME --zone=ZONE --quiet
   ```
