# TPU VM Lifecycle

## Table of Contents

1. [Runtime Version Selection](#runtime-version-selection)
2. [Creating TPU VMs](#creating-tpu-vms)
3. [Networking & Firewall](#networking--firewall)
4. [Environment Setup](#environment-setup)
5. [Code Upload](#code-upload)
6. [Checking TPU Status](#checking-tpu-status)
7. [Cleanup](#cleanup)

## Runtime Version Selection

**This is the single most critical decision.** Wrong runtime = JAX cannot detect TPU.

| TPU Type | Runtime Version | Notes |
|----------|----------------|-------|
| v6e | `v2-alpha-tpuv6e` | **Required**. No alternative. |
| v5e | `v2-alpha-tpuv5e` | For v5e accelerators |
| v5p | `v2-alpha-tpuv5p` | For v5p accelerators |
| v4 | `tpu-ubuntu2204-base` | Older generation |

### Wrong runtime symptoms

- `/dev/accel*` device files missing (v6e uses `/dev/vfio/0,1,2,3`)
- `jax.devices()` raises `Failed to get global TPU topology`
- JAX falls back to CPU silently

## Creating TPU VMs

### Single-host

```bash
gcloud compute tpus tpu-vm create VM_NAME \
    --zone=ZONE \
    --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e \
    --spot
```

### Accelerator types

| Type | Chips | Hosts | HBM/chip | Total HBM |
|------|-------|-------|----------|-----------|
| v6e-4 | 4 | 1 | 31.25 GB | 125 GB |
| v6e-8 | 8 | 1 | 31.25 GB | 250 GB |
| v6e-16 | 16 | 4 | 31.25 GB | 500 GB |
| v6e-32 | 32 | 8 | 31.25 GB | 1 TB |
| v6e-64 | 64 | 16 | 31.25 GB | 2 TB |
| v6e-128 | 128 | 32 | 31.25 GB | 4 TB |
| v6e-256 | 256 | 64 | 31.25 GB | 8 TB |

### Spot vs on-demand

- `--spot`: cheaper, but can be preempted anytime. Good for experimentation.
- Without `--spot`: on-demand pricing, guaranteed availability. Use for production runs.
- **Always enable checkpointing with spot TPUs** to allow resuming after preemption.

### Available zones (as of 2026-02)

Spot availability varies. If one zone fails with STOCKOUT, try others:

- `us-central1-b` (v6e) — frequently preempted
- `us-east5-b` (v6e)
- `asia-northeast1-b` (v6e) — good for v6e-16 spot
- `europe-west4-b` (v6e)
- `us-central2-b` (v4)

## Networking & Firewall

### Default VPC

No extra config needed. SSH works directly:

```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE
```

### Custom VPC

Specify network and subnetwork at creation:

```bash
gcloud compute tpus tpu-vm create VM_NAME \
    --zone=ZONE \
    --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e \
    --spot \
    --network=VPC_NAME \
    --subnetwork=SUBNET_NAME
```

### SSH methods

```bash
# Method 1: IAP tunnel (recommended for custom VPC without public IP)
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --tunnel-through-iap

# Method 2: Direct SSH (requires firewall rule allowing tcp:22)
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE
```

### Temporary firewall rule (testing only)

```bash
# Create
gcloud compute firewall-rules create allow-tpu-ssh-test \
    --network=VPC_NAME \
    --allow=tcp:22 \
    --source-ranges=0.0.0.0/0

# Delete when done
gcloud compute firewall-rules delete allow-tpu-ssh-test --quiet
```

## Environment Setup

### Install JAX/Flax dependencies

```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all --command='
pip install -r PROJECT_DIR/requirements.txt
'
```

### Verified dependency versions (JAX 0.6.2 on v6e)

```
jax==0.6.2
jaxlib==0.6.2
libtpu==0.0.17
flax==0.10.7
optax==0.2.5+
orbax-checkpoint==0.11.15
safetensors==0.5.x
transformers==4.51.x+
huggingface_hub==0.30.x
```

### orbax-checkpoint version compatibility (JAX 0.6.2)

**Only 0.11.15 works.** All others break:

| Version | Status | Error |
|---------|--------|-------|
| **0.11.15** | Works | — |
| 0.11.32+ | Broken | `jax.sharding.set_mesh` not a context manager |
| 0.10.x | Broken | `jax._src.config.enable_memories` missing |
| 0.9.x | Broken | `jax.lib.xla_extension.XlaRuntimeError` removed |

### Set HuggingFace token (for gated models)

```bash
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all --command='
export HF_TOKEN=hf_xxx
'
```

## Code Upload

```bash
# Pack locally
tar czf /tmp/code.tar.gz PROJECT_DIR/

# Upload to all workers (multi-host) or single host
gcloud compute tpus tpu-vm scp /tmp/code.tar.gz VM_NAME:~ --zone=ZONE --worker=all

# Extract on all workers
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all \
    --command='tar xzf code.tar.gz'
```

For iterative development, upload only changed files:

```bash
gcloud compute tpus tpu-vm scp LOCAL_FILE VM_NAME:~/REMOTE_PATH --zone=ZONE --worker=all
```

## Checking TPU Status

```bash
# List all TPUs in a zone
gcloud compute tpus tpu-vm list --zone=ZONE

# Describe a specific TPU
gcloud compute tpus tpu-vm describe VM_NAME --zone=ZONE

# Check JAX can see TPU
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE \
    --command='python3 -c "import jax; print(jax.devices())"'
```

## Cleanup

Always clean up TPU VMs and any temporary firewall rules after use:

```bash
# Delete TPU VM
gcloud compute tpus tpu-vm delete VM_NAME --zone=ZONE --quiet

# Delete temporary firewall rules
gcloud compute firewall-rules delete RULE_NAME --quiet
```
