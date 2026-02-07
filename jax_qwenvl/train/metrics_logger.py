"""Unified metrics logging for wandb and/or tensorboard."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Unified metrics logging for wandb and/or tensorboard.

    When report_to="none", all methods are no-ops.
    """

    def __init__(
        self,
        output_dir: str,
        report_to: str = "none",  # "wandb", "tensorboard", "wandb,tensorboard", "none"
        run_name: str = "",
        config: Optional[dict] = None,
    ):
        self._writers = []
        self._summary_writer = None

        if "wandb" in report_to:
            try:
                import wandb
                wandb.init(project="qwen3-vl", name=run_name or None, config=config)
                self._writers.append("wandb")
                logger.info("wandb logging initialized")
            except ImportError:
                logger.warning("wandb not installed, skipping wandb logging")

        if "tensorboard" in report_to:
            try:
                from flax.metrics import tensorboard
                self._summary_writer = tensorboard.SummaryWriter(output_dir)
                self._writers.append("tensorboard")
                logger.info("tensorboard logging initialized at %s", output_dir)
            except ImportError:
                logger.warning("flax.metrics.tensorboard not available, skipping tensorboard logging")

    def log(self, metrics: dict, step: int) -> None:
        """Log a dict of metrics at the given step."""
        if "wandb" in self._writers:
            import wandb
            wandb.log(metrics, step=step)
        if "tensorboard" in self._writers and self._summary_writer is not None:
            for k, v in metrics.items():
                self._summary_writer.scalar(k, float(v), step)

    def finish(self) -> None:
        """Clean up logging resources."""
        if "wandb" in self._writers:
            import wandb
            wandb.finish()
        if "tensorboard" in self._writers and self._summary_writer is not None:
            self._summary_writer.flush()
