"""SPMD sharding utilities for JAX Qwen3-VL distributed training."""

from __future__ import annotations

from typing import Optional

import jax
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
              ``'fsdp'`` for fully-sharded (2-D params sharded on first axis).

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
    elif mode == 'fsdp':
        return {
            'kernel': P('fsdp', None),
            'embedding': P('fsdp', None),
            'default_1d': P(),
            'default': P(),
        }
    else:
        raise ValueError(f"Unknown sharding mode: {mode!r} (expected 'dp' or 'fsdp')")


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
    def _get_sharding(path_tuple, value):
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
        return jax.device_put(value, NamedSharding(mesh, spec))

    return jax.tree_util.tree_map_with_path(_get_sharding, params)


def shard_batch(batch, mesh: Mesh):
    """Shard a Batch's arrays onto the mesh for data parallelism.

    - Most 2-D+ arrays: batch dim (axis 0) is sharded along ``'dp'``.
    - ``position_ids``: shape ``(3, B, L)`` -- batch dim is axis 1.
    - Scalars and None values are passed through unchanged.

    Args:
        batch: a ``Batch`` NamedTuple from the data pipeline.
        mesh: the device ``Mesh``.

    Returns:
        Sharded batch.
    """
    dp_sharding = NamedSharding(mesh, P('dp'))
    replicated = NamedSharding(mesh, P())
    # position_ids has shape (3, B, L): shard axis 1
    pos_sharding = NamedSharding(mesh, P(None, 'dp', None))

    def _shard_field(name, x):
        if x is None:
            return None
        if name == 'position_ids':
            return jax.device_put(x, pos_sharding)
        if x.ndim >= 2:
            return jax.device_put(x, dp_sharding)
        if x.ndim == 1:
            return jax.device_put(x, dp_sharding)
        return jax.device_put(x, replicated)

    # Batch is a NamedTuple -- iterate over field names and values
    field_names = batch._fields
    sharded_values = tuple(
        _shard_field(name, val) for name, val in zip(field_names, batch)
    )
    return type(batch)(*sharded_values)
