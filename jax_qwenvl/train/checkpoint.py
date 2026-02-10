"""Checkpoint manager for JAX Qwen3-VL training.

Orbax 0.11.15's multi-host coordination (both CheckpointManager and
StandardCheckpointer) deadlocks with JAX 0.6.2.  This module works around
the issue:

- **Single-host**: uses ``ocp.CheckpointManager`` normally (sync mode).
- **Multi-host DP mode**: saves from process 0 only using
  ``jax.device_get`` + ``flax.serialization`` (no Orbax at all).
  The next ``train_step`` collective provides implicit synchronization.
"""

from __future__ import annotations

import logging
import os
import shutil
import time as _time
from typing import Optional

import jax
import orbax.checkpoint as ocp

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Checkpoint wrapper that works reliably in both single-host and
    multi-host modes."""

    def __init__(
        self,
        output_dir: str,
        max_to_keep: int = 3,
        save_interval_steps: int = 500,
        gcs_dir: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.save_interval_steps = save_interval_steps
        self._gcs_dir = gcs_dir
        self._max_to_keep = max_to_keep
        self._is_multihost = jax.process_count() > 1

        if self._is_multihost:
            # Multi-host: bypass Orbax entirely.
            # Process 0 saves using flax.serialization (pure file I/O).
            # No Orbax coordination.  The next train_step collective
            # provides implicit synchronization.
            if jax.process_index() == 0:
                os.makedirs(output_dir, exist_ok=True)
            logger.info(
                "Multi-host checkpoint: process-0-only save "
                "(bypassing Orbax — no coordination needed)"
            )
        else:
            # Single-host: Orbax CheckpointManager works fine in sync mode.
            options = ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                save_interval_steps=1,
                enable_async_checkpointing=False,
            )
            self.manager = ocp.CheckpointManager(output_dir, options=options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, step: int, state, force: bool = False):
        """Save training state at the given step."""
        if self._is_multihost:
            self._save_multihost(step, state)
        else:
            self._save_singlehost(step, state, force)

    def wait_for_completion(self):
        """Wait for any pending work (no-op — all saves are sync)."""
        pass

    def restore(self, step: Optional[int] = None, state_template=None):
        """Restore training state."""
        if self._is_multihost:
            return self._restore_multihost(step, state_template)
        if step is None:
            step = self.manager.latest_step()
        if step is None:
            return None
        restored = self.manager.restore(
            step, args=ocp.args.StandardRestore(state_template),
        )
        logger.info("Checkpoint restored from step %d", step)
        return restored

    def latest_step(self) -> Optional[int]:
        """Return the latest checkpoint step, or None."""
        if self._is_multihost:
            return self._latest_step_multihost()
        return self.manager.latest_step()

    def should_save(self, step: int) -> bool:
        """Check if a checkpoint should be saved at this step."""
        return step % self.save_interval_steps == 0

    # ------------------------------------------------------------------
    # Single-host implementation
    # ------------------------------------------------------------------

    def _save_singlehost(self, step: int, state, force: bool):
        saved = self.manager.save(
            step, args=ocp.args.StandardSave(state), force=force,
        )
        if saved:
            self.manager.wait_until_finished()
            logger.info("Checkpoint saved at step %d", step)
        else:
            logger.warning("Checkpoint at step %d was skipped", step)

        if self._gcs_dir and jax.process_index() == 0:
            self._sync_to_gcs(step)

    # ------------------------------------------------------------------
    # Multi-host implementation (process-0-only, no Orbax)
    # ------------------------------------------------------------------

    def _save_multihost(self, step: int, state):
        """Save checkpoint from process 0 only using flax serialization.

        In DP mode every host has the full parameter set.  Only process 0
        needs to save.  No Orbax coordination is used — the next
        ``train_step`` collective provides implicit synchronisation.
        """
        if jax.process_index() != 0:
            return

        import flax.serialization

        t0 = _time.time()
        path = os.path.join(self.output_dir, str(step))
        tmp_path = path + ".tmp"
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        os.makedirs(tmp_path)

        # Convert JAX arrays to numpy (local operation in DP mode)
        state_host = jax.device_get(state)

        # Serialize using flax msgpack
        state_bytes = flax.serialization.to_bytes(state_host)
        filepath = os.path.join(tmp_path, "state.msgpack")
        with open(filepath, "wb") as f:
            f.write(state_bytes)

        # Atomic rename (remove existing dir if re-saving same step)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.rename(tmp_path, path)

        elapsed = _time.time() - t0
        size_mb = len(state_bytes) / (1024 * 1024)
        logger.info(
            "Checkpoint saved at step %d (%.0f MB, %.1fs)",
            step, size_mb, elapsed,
        )
        self._cleanup_old(step)

        if self._gcs_dir:
            self._sync_to_gcs(step)

    def _restore_multihost(self, step=None, state_template=None):
        """Restore from a process-0-only checkpoint."""
        import flax.serialization

        if step is None:
            step = self._latest_step_multihost()
        if step is None:
            return None

        filepath = os.path.join(self.output_dir, str(step), "state.msgpack")
        if not os.path.isfile(filepath):
            logger.warning("Checkpoint file not found: %s", filepath)
            return None

        with open(filepath, "rb") as f:
            state_bytes = f.read()

        restored = flax.serialization.from_bytes(state_template, state_bytes)
        logger.info("Checkpoint restored from step %d", step)
        return restored

    def _latest_step_multihost(self) -> Optional[int]:
        """Find latest checkpoint step by scanning directory."""
        if not os.path.isdir(self.output_dir):
            return None
        steps = []
        for name in os.listdir(self.output_dir):
            full = os.path.join(self.output_dir, name)
            if os.path.isdir(full) and name.isdigit():
                steps.append(int(name))
        return max(steps) if steps else None

    def _cleanup_old(self, current_step: int):
        """Remove old checkpoints beyond max_to_keep."""
        steps = []
        for name in os.listdir(self.output_dir):
            full = os.path.join(self.output_dir, name)
            if os.path.isdir(full) and name.isdigit():
                steps.append(int(name))
        steps.sort()
        while len(steps) > self._max_to_keep:
            old_step = steps.pop(0)
            old_path = os.path.join(self.output_dir, str(old_step))
            shutil.rmtree(old_path, ignore_errors=True)
            logger.info("Removed old checkpoint step %d", old_step)

    # ------------------------------------------------------------------
    # GCS sync
    # ------------------------------------------------------------------

    def _sync_to_gcs(self, step: int):
        """Upload a checkpoint step directory to GCS."""
        import subprocess

        src = os.path.join(self.output_dir, str(step))
        dst = self._gcs_dir.rstrip("/") + "/" + str(step) + "/"
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", "-r", src, dst],
                check=True, capture_output=True, text=True,
            )
            logger.info("Checkpoint step %d synced to %s", step, dst)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("Failed to sync checkpoint to GCS: %s", e)
