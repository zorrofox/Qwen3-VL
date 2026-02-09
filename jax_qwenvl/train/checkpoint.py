"""Orbax checkpoint manager for JAX Qwen3-VL training."""

from __future__ import annotations

import logging
import os
from typing import Optional

import jax
import orbax.checkpoint as ocp

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Wrapper around ``orbax.checkpoint.CheckpointManager``.

    Handles saving/restoring ``TrainState`` with automatic step tracking
    and checkpoint rotation.
    """

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

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
            enable_async_checkpointing=False,
        )
        self.manager = ocp.CheckpointManager(
            output_dir,
            options=options,
        )

    def save(self, step: int, state, force: bool = False):
        """Save training state at the given step.

        Args:
            step: current global training step.
            state: ``TrainState`` pytree to save.
            force: if True, save regardless of interval.
        """
        if not force and step % self.save_interval_steps != 0:
            return
        self.manager.save(
            step,
            args=ocp.args.StandardSave(state),
        )
        self.manager.wait_until_finished()
        logger.info("Checkpoint saved at step %d", step)
        if self._gcs_dir and jax.process_index() == 0:
            self._sync_to_gcs(step)

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

    def restore(self, step: Optional[int] = None, state_template=None):
        """Restore training state.

        Args:
            step: step to restore from. ``None`` = latest.
            state_template: pytree with correct shape/dtype/sharding
                for restoration.

        Returns:
            Restored state, or ``None`` if no checkpoint found.
        """
        if step is None:
            step = self.manager.latest_step()
        if step is None:
            return None

        restored = self.manager.restore(
            step,
            args=ocp.args.StandardRestore(state_template),
        )
        logger.info("Checkpoint restored from step %d", step)
        return restored

    def latest_step(self) -> Optional[int]:
        """Return the latest checkpoint step, or None."""
        return self.manager.latest_step()

    def should_save(self, step: int) -> bool:
        """Check if a checkpoint should be saved at this step."""
        return step % self.save_interval_steps == 0
