"""SPMD sharding utilities for JAX Qwen3-VL distributed training."""

from __future__ import annotations

from typing import Optional

import jax
import os
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def create_device_mesh(
    dp: int = -1,
    fsdp: int = 1,
    tp: int = 1,
) -> Mesh:
    """Create a device mesh for SPMD parallelism.

    Args:
        dp: data parallel dimension (-1 = auto-fill remaining).
        fsdp: fully-sharded data parallel dimension.
        tp: tensor parallel dimension (reserved for future use).

    Returns:
        A JAX ``Mesh`` with axis names ``('dp', 'fsdp', 'tp')``.
    """
    devices = jax.devices()
    num_devices = len(devices)

    if dp == -1:
        dp = num_devices // (fsdp * tp)
    elif fsdp == -1:
        fsdp = num_devices // (dp * tp)

    assert dp * fsdp * tp == num_devices, (
        f"dp={dp} * fsdp={fsdp} * tp={tp} = {dp * fsdp * tp} "
        f"!= {num_devices} devices"
    )

    device_array = np.array(devices).reshape(dp, fsdp, tp)
    return Mesh(device_array, axis_names=('dp', 'fsdp', 'tp'))


def get_param_sharding_rules(mode: str = 'dp') -> dict:
    """Return parameter sharding rules.

    Args:
        mode: ``'dp'`` for pure data parallelism (all params replicated),
              ``'fsdp'`` for fully-sharded (2-D params sharded on first axis),
              ``'hybrid'`` for mixed DP+FSDP (params sharded same as fsdp).

    Returns:
        Dict with keys ``'kernel'``, ``'embedding'``, ``'default_1d'``,
        ``'default'`` mapping to ``PartitionSpec``.
    """
    if mode == 'dp':
        return {
            'kernel': P(),
            'embedding': P(),
            'default_1d': P(),
            'default': P(),
        }
    elif mode in ('fsdp', 'hybrid'):
        return {
            'kernel': P('fsdp', None),
            'embedding': P('fsdp', None),
            'default_1d': P(),
            'default': P(),
        }
    else:
        raise ValueError(f"Unknown sharding mode: {mode!r} (expected 'dp', 'fsdp', or 'hybrid')")


def shard_params(params: dict, mesh: Mesh, rules: dict) -> dict:
    """Place parameters onto the mesh according to sharding rules.

    Args:
        params: nested parameter dict.
        mesh: the device ``Mesh``.
        rules: output of ``get_param_sharding_rules``.

    Returns:
        Parameter dict with each leaf placed on the mesh via
        ``jax.device_put``.
    """
    def _get_spec(path_tuple, value):
        path = '/'.join(str(p) for p in path_tuple)
        is_2d_plus = value.ndim >= 2
        is_kernel_like = (
            'kernel' in path or 'embedding' in path or 'embed_tokens' in path
        )
        if is_2d_plus and is_kernel_like:
            spec = rules.get('kernel', rules['default'])
            # Check that the sharded dimension is divisible by the mesh axis.
            # If not, fall back to replicated to avoid errors.
            if spec != P():
                for axis_idx, axis_name in enumerate(spec):
                    if axis_name is not None and axis_idx < value.ndim:
                        axis_size = mesh.shape[axis_name]
                        if value.shape[axis_idx] % axis_size != 0:
                            spec = rules['default']
                            break
        elif value.ndim == 1:
            spec = rules.get('default_1d', rules['default'])
        else:
            spec = rules['default']
        return NamedSharding(mesh, spec)

    # Build sharding tree first, then do a single batched device_put.
    # This avoids per-leaf device_put calls that can trigger TPU watchdog
    # timeouts for large models (8B+).
    sharding_tree = jax.tree_util.tree_map_with_path(_get_spec, params)
    return jax.device_put(params, sharding_tree)


_VISION_FIELDS = frozenset({
    'pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw',
    'image_pos_ids_2d', 'image_pos_ids_1d', 'image_cu_seqlens',
    'video_pos_ids_2d', 'video_pos_ids_1d', 'video_cu_seqlens',
})


def shard_batch(batch, mesh: Mesh, mode: str = 'dp'):
    """Shard a Batch's arrays onto the mesh for data parallelism.

    Supports both single-host and multi-host (TPU pod slice) configurations.
    In multi-host mode, uses ``host_local_array_to_global_array`` to assemble
    global arrays from each host's local shard.

    - Text fields (batch dim on axis 0): sharded on data axis.
    - ``position_ids``: shape ``(3, B, L)`` -- sharded on axis 1.
    - Vision fields: replicated across all devices.
    - Scalars and None values are passed through unchanged.

    Args:
        batch: a ``Batch`` NamedTuple from the data pipeline.
              In multi-host mode, each host should provide the FULL batch
              (same data on all hosts, via deterministic shuffling).
        mesh: the device ``Mesh``.
        mode: ``'dp'`` shards batch on the dp axis,
              ``'fsdp'`` shards batch on the fsdp axis,
              ``'hybrid'`` shards batch on both dp and fsdp axes
              (``P(('dp', 'fsdp'))``), total shards = dp × fsdp.

    Returns:
        Sharded batch with global arrays placed on the mesh.
    """
    num_processes = jax.process_count()
    use_multihost = num_processes > 1

    if use_multihost:
        return _shard_batch_multihost(batch, mesh, mode=mode)

    # Single-host path
    if mode == 'hybrid':
        data_axis = ('dp', 'fsdp')
    elif mode == 'fsdp':
        data_axis = 'fsdp'
    else:
        data_axis = 'dp'
    dp_sharding = NamedSharding(mesh, P(data_axis))
    replicated = NamedSharding(mesh, P())
    pos_sharding = NamedSharding(mesh, P(None, data_axis, None))

    def _shard_field(name, x):
        if x is None:
            return None
        if name == 'position_ids':
            return jax.device_put(x, pos_sharding)
        if name in _VISION_FIELDS:
            if os.environ.get('SHARD_VISION_BATCH', '0') == '1':
                if name in ('image_cu_seqlens', 'video_cu_seqlens'):
                    return jax.device_put(x, replicated)
                return jax.device_put(x, dp_sharding)
            return jax.device_put(x, replicated)
        if x.ndim >= 1:
            return jax.device_put(x, dp_sharding)
        return jax.device_put(x, replicated)

    field_names = batch._fields
    sharded_values = tuple(
        _shard_field(name, val) for name, val in zip(field_names, batch)
    )
    return type(batch)(*sharded_values)


def _shard_batch_multihost(batch, mesh: Mesh, mode: str = 'dp'):
    """Multi-host version of shard_batch.

    Each host provides the FULL batch. This function slices out each host's
    local shard for data-parallel fields, and passes vision fields in full
    (replicated).

    Uses ``jax.make_array_from_callback`` to create global arrays spanning
    all hosts.
    """
    from jax.experimental.multihost_utils import host_local_array_to_global_array

    process_index = jax.process_index()
    num_processes = jax.process_count()
    local_device_count = jax.local_device_count()

    if mode == 'hybrid':
        data_axis = ('dp', 'fsdp')
    elif mode == 'fsdp':
        data_axis = 'fsdp'
    else:
        data_axis = 'dp'
    dp_pspec = P(data_axis)
    pos_pspec = P(None, data_axis, None)
    replicated_pspec = P()

    def _shard_field(name, x):
        if x is None:
            return None

        if name == 'position_ids':
            # Shape (3, B, L) — shard on axis 1 (batch dim)
            B = x.shape[1]
            local_B = B // num_processes
            local_x = x[:, process_index * local_B : (process_index + 1) * local_B, :]
            return host_local_array_to_global_array(local_x, mesh, pos_pspec)

        if name in _VISION_FIELDS:
            if os.environ.get('SHARD_VISION_BATCH', '0') == '1':
                if name in ('image_cu_seqlens', 'video_cu_seqlens'):
                    global_devices = jax.device_count()
                    local_max = x[-1] // global_devices
                    dummy = np.array([0, local_max], dtype=np.int32)
                    return host_local_array_to_global_array(dummy, mesh, replicated_pspec)
                # Shard on data axis
                B = x.shape[0]
                local_B = B // num_processes
                local_x = x[process_index * local_B : (process_index + 1) * local_B]
                return host_local_array_to_global_array(local_x, mesh, dp_pspec)
            # Replicated: each host provides the full array
            return host_local_array_to_global_array(x, mesh, replicated_pspec)

        if x.ndim >= 1:
            # DP-sharded on axis 0 (batch dim)
            B = x.shape[0]
            local_B = B // num_processes
            local_x = x[process_index * local_B : (process_index + 1) * local_B]
            return host_local_array_to_global_array(local_x, mesh, dp_pspec)

        # Scalar
        return host_local_array_to_global_array(x, mesh, replicated_pspec)

    field_names = batch._fields
    sharded_values = tuple(
        _shard_field(name, val) for name, val in zip(field_names, batch)
    )
    return type(batch)(*sharded_values)
