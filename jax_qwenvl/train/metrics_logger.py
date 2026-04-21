"""Unified metrics logging for wandb and/or tensorboard."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Unified metrics logging for wandb and/or tensorboard.

    When report_to="none", all methods are no-ops.

    For GCS logging_dir (gs://...), tensorboard events are written to a local
    temp directory first, then synced to GCS on ``finish()``.  This avoids the
    GCS append-mode limitation that would cause event data loss.
    """

    def __init__(
        self,
        output_dir: str,
        report_to: str = "none",  # "wandb", "tensorboard", "wandb,tensorboard", "none"
        run_name: str = "",
        config: Optional[dict] = None,
        logging_dir: Optional[str] = None,
        sync_interval: int = 100,
    ):
        self._writers = []
        self._summary_writer = None
        self._local_tb_dir: Optional[str] = None
        self._gcs_tb_dir: Optional[str] = None
        self._sync_interval = sync_interval

        if "wandb" in report_to:
            try:
                import wandb
                wandb.init(project="qwen3-vl", name=run_name or None, config=config)
                self._writers.append("wandb")
                logger.info("wandb logging initialized")
            except ImportError:
                logger.warning("wandb not installed, skipping wandb logging")

        if "tensorboard" in report_to:
            tb_dir = logging_dir or output_dir
            # GCS paths don't support append mode — write locally then sync.
            if tb_dir.startswith("gs://"):
                self._gcs_tb_dir = tb_dir
                self._local_tb_dir = tempfile.mkdtemp(prefix="tb_logs_")
                tb_dir = self._local_tb_dir
                logger.info(
                    "GCS tensorboard: writing locally to %s, will sync to %s on finish",
                    self._local_tb_dir, self._gcs_tb_dir,
                )
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._summary_writer = SummaryWriter(log_dir=tb_dir)
                self._writers.append("tensorboard")
                logger.info("tensorboard logging initialized at %s", tb_dir)
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter
                    self._summary_writer = SummaryWriter(log_dir=tb_dir)
                    self._writers.append("tensorboard")
                    logger.info("tensorboard logging initialized (tensorboardX) at %s", tb_dir)
                except ImportError:
                    logger.warning(
                        "tensorboard not available (tried torch.utils.tensorboard and tensorboardX), skipping"
                    )

    def log(self, metrics: dict, step: int) -> None:
        """Log a dict of metrics at the given step."""
        if "wandb" in self._writers:
            import wandb
            wandb.log(metrics, step=step)
        if "tensorboard" in self._writers and self._summary_writer is not None:
            for k, v in metrics.items():
                self._summary_writer.add_scalar(k, float(v), step)
            # Periodically flush and sync to GCS to guard against spot preemption.
            if (
                self._gcs_tb_dir
                and self._local_tb_dir
                and step > 0
                and step % self._sync_interval == 0
            ):
                self._summary_writer.flush()
                self._sync_to_gcs(cleanup=False)

    def finish(self) -> None:
        """Clean up logging resources.

        If tensorboard was writing to a local temp dir for GCS, sync the event
        files to the GCS destination using ``gsutil``.
        """
        if "wandb" in self._writers:
            import wandb
            wandb.finish()
        if "tensorboard" in self._writers and self._summary_writer is not None:
            self._summary_writer.flush()
            self._summary_writer.close()

            # Sync local tensorboard logs to GCS if needed.
            if self._gcs_tb_dir and self._local_tb_dir:
                self._sync_to_gcs()

    def _sync_to_gcs(self, cleanup: bool = True) -> None:
        """Copy local tensorboard event files to GCS.

        Args:
            cleanup: If True, remove the local temp directory after a
                successful sync (used at ``finish()``).  If False, keep local
                files so that subsequent events can still be appended (used for
                periodic mid-training syncs).
        """
        from jax_qwenvl.utils.gcs import sync_dir_to_gcs

        logger.info("Syncing tensorboard logs: %s -> %s", self._local_tb_dir, self._gcs_tb_dir)
        try:
            uploaded = sync_dir_to_gcs(self._local_tb_dir, self._gcs_tb_dir)
            logger.info("Tensorboard logs synced to %s (%d files)", self._gcs_tb_dir, uploaded)
            if cleanup:
                shutil.rmtree(self._local_tb_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(
                "Failed to sync tensorboard logs to GCS: %s. "
                "Local logs preserved at %s",
                e, self._local_tb_dir,
            )
