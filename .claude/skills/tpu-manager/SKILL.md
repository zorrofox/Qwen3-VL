---
name: tpu-manager
description: |
  Manage Google Cloud TPU VMs for JAX/Flax training workloads. Use when the user needs to:
  (1) Create, configure, or delete TPU VMs (v6e, v5e, etc.)
  (2) Set up TPU training environments (JAX, Flax, orbax, dependencies)
  (3) Upload code and run training on TPU VMs (single-host or multi-host pod slices)
  (4) Troubleshoot TPU issues (runtime errors, OOM, SSH, checkpoint failures)
  (5) Manage TPU networking, firewall rules, and IAP tunnels
  (6) Work with multi-host TPU pod slices (v6e-16, v6e-32, etc.)
  Triggers: "create TPU", "TPU VM", "tpu training", "v6e", "pod slice", "multi-host",
  "gcloud tpus", "TPU environment", "TPU setup", "TPU troubleshooting"
---

# TPU Manager

Manage Google Cloud TPU VMs for JAX/Flax training. See reference files for details:

- **Runtime & lifecycle**: See [tpu-lifecycle.md](references/tpu-lifecycle.md) for creation, networking, environment setup, cleanup
- **Multi-host pod slices**: See [multi-host.md](references/multi-host.md) for v6e-16+ training
- **Troubleshooting**: See [troubleshooting.md](references/troubleshooting.md) for common errors and fixes

## Quick Reference

### Create TPU VM

```bash
# Single-host (v6e-4)
gcloud compute tpus tpu-vm create VM_NAME \
    --zone=ZONE --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e --spot

# Multi-host pod slice (v6e-16 = 4 hosts x 4 chips)
gcloud compute tpus tpu-vm create VM_NAME \
    --zone=ZONE --accelerator-type=v6e-16 \
    --version=v2-alpha-tpuv6e --spot
```

### SSH & Run Commands

```bash
# Single-host SSH (via IAP tunnel)
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --tunnel-through-iap

# Multi-host: run on ALL workers
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all --command='CMD'
```

### Upload Code

```bash
tar czf /tmp/code.tar.gz PROJECT_DIR/
gcloud compute tpus tpu-vm scp /tmp/code.tar.gz VM_NAME:~ --zone=ZONE --worker=all
gcloud compute tpus tpu-vm ssh VM_NAME --zone=ZONE --worker=all \
    --command='tar xzf code.tar.gz'
```

### Delete TPU

```bash
gcloud compute tpus tpu-vm delete VM_NAME --zone=ZONE --quiet
```

## Critical Rules

1. **v6e TPU MUST use `v2-alpha-tpuv6e` runtime** — other runtimes fail silently
2. **Multi-host: all workers must run the script simultaneously** via `--worker=all`
3. **Multi-host: call `jax.distributed.initialize()`** before any JAX operation
4. **Multi-host: checkpoint paths must be absolute** — Orbax rejects relative paths
5. **orbax-checkpoint==0.11.15** is the only version compatible with JAX 0.6.2
6. **Use bfloat16 for 2B+ models on v6e-4** — float32 OOMs (31.25 GB HBM/chip)
7. **Spot TPUs can be preempted anytime** — enable checkpointing for long runs

## Decision Tree

```
User wants to create/manage TPU
├── Create single-host (v6e-4, v6e-8)    → See tpu-lifecycle.md
├── Create multi-host (v6e-16+)           → See multi-host.md
├── Custom VPC / firewall                 → See tpu-lifecycle.md "Networking"
├── Install dependencies                  → See tpu-lifecycle.md "Environment"
└── Cleanup resources                     → See tpu-lifecycle.md "Cleanup"

User has TPU errors
├── "Failed to get global TPU topology"   → Wrong runtime (use v2-alpha-tpuv6e)
├── OOM / RESOURCE_EXHAUSTED              → Use bf16 + gradient checkpointing
├── SSH Connection timed out              → Firewall rules or use --tunnel-through-iap
├── orbax / checkpoint errors             → See troubleshooting.md
├── XLA recompilation every step          → Pad all tensors to fixed shapes
└── Other                                 → See troubleshooting.md
```
