"""Evaluation placeholders."""

from __future__ import annotations

from predictor.types import KernelDataset, TrainerState


class PlaceholderEvaluator:
    """Return deterministic placeholder evaluation metrics."""

    def evaluate(self, dataset: KernelDataset, trainer_state: TrainerState) -> dict[str, float]:
        """Evaluate a placeholder model state on a dataset."""

        del trainer_state
        return {"mae": 0.0, "sample_count": float(len(dataset.samples))}
