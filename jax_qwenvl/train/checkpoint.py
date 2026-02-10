"""Checkpoint manager for JAX Qwen3-VL training.

Uses Orbax CheckpointManager with GCS paths for native multi-host support.
"""

from __future__ import annotations

import logging
from typing import Optional

import orbax.checkpoint as ocp

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Checkpoint wrapper using Orbax native multi-host coordination.

    When ``gcs_dir`` is provided, checkpoints are stored under
    ``<gcs_dir>/checkpoints/`` so that all hosts can read/write directly
    via GCS (no ``gcloud`` CLI needed).
    """

    def __init__(
        self,
        output_dir: str,
        max_to_keep: int = 3,
        save_interval_steps: int = 500,
        gcs_dir: Optional[str] = None,
    ):
        self.save_interval_steps = save_interval_steps

        # Use GCS path for checkpoints if available (enables native multi-host);
        # put checkpoints in a subdirectory to avoid conflicting with model
        # export files (safetensors, json, jinja).
        if gcs_dir:
            ckpt_dir = gcs_dir.rstrip("/") + "/checkpoints"
        else:
            ckpt_dir = output_dir

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=1,  # We control save interval externally
            enable_async_checkpointing=False,
        )
        self.manager = ocp.CheckpointManager(ckpt_dir, options=options)
        logger.info("CheckpointManager initialized at %s", ckpt_dir)

    def save(self, step: int, state, force: bool = False):
        """Save training state at the given step."""
        saved = self.manager.save(
            step, args=ocp.args.StandardSave(state), force=force,
        )
        if saved:
            self.manager.wait_until_finished()
            logger.info("Checkpoint saved at step %d", step)

    def restore(self, step: Optional[int] = None, state_template=None):
        """Restore training state.

        Uses ``StandardRestore`` with *state_template* so that Orbax
        restores arrays directly onto the correct devices with the
        template's sharding — no manual re-shard needed.
        """
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
        return self.manager.latest_step()

    def should_save(self, step: int) -> bool:
        """Check if a checkpoint should be saved at this step."""
        return step % self.save_interval_steps == 0

    def wait_for_completion(self):
        """Wait for any pending checkpoint writes."""
        self.manager.wait_until_finished()
