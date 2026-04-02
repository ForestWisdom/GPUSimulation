"""Training placeholders."""

from __future__ import annotations

from predictor.types import KernelDataset, TrainerState


class PlaceholderTrainer:
    """Record dataset statistics instead of fitting a real model."""

    def fit(self, dataset: KernelDataset) -> TrainerState:
        """Fit a placeholder trainer and return its state."""

        return TrainerState(
            model_name="phase1-placeholder-residual",
            sample_count=len(dataset.samples),
        )
