"""Residual-model evaluation helpers."""

from __future__ import annotations

import math

from predictor.types import KernelDataset, TrainerState
from predictor.types import ResidualTrainerState, ResidualTrainingDataset


class PlaceholderEvaluator:
    """Return deterministic placeholder evaluation metrics."""

    def evaluate(self, dataset: KernelDataset, trainer_state: TrainerState) -> dict[str, float]:
        """Evaluate a placeholder model state on a dataset."""

        del trainer_state
        return {"mae": 0.0, "sample_count": float(len(dataset.samples))}


class ResidualEvaluator:
    """Evaluate baseline-only and baseline-plus-residual accuracy."""

    def evaluate(
        self,
        dataset: ResidualTrainingDataset,
        trainer_state: ResidualTrainerState,
        train_size: int,
    ) -> dict[str, float]:
        """Evaluate the fitted residual model on a held-out dataset."""

        residual_targets = [sample.residual_target_ms for sample in dataset.samples]
        predicted_residuals = trainer_state.model.predict_batch(
            [sample.features for sample in dataset.samples]
        )
        measured_latencies = [sample.measured_latency_ms for sample in dataset.samples]
        baseline_latencies = [sample.analytical_baseline_ms for sample in dataset.samples]
        combined_latencies = [
            max(1e-6, baseline + residual)
            for baseline, residual in zip(baseline_latencies, predicted_residuals)
        ]
        return {
            "train_size": float(train_size),
            "test_size": float(len(dataset.samples)),
            "residual_mae": _mae(residual_targets, predicted_residuals),
            "residual_rmse": _rmse(residual_targets, predicted_residuals),
            "baseline_only_latency_mae": _mae(measured_latencies, baseline_latencies),
            "baseline_plus_residual_latency_mae": _mae(
                measured_latencies,
                combined_latencies,
            ),
            "baseline_only_latency_mape": _mape(
                measured_latencies,
                baseline_latencies,
            ),
            "baseline_plus_residual_latency_mape": _mape(
                measured_latencies,
                combined_latencies,
            ),
        }


def _mae(actuals: list[float], predictions: list[float]) -> float:
    """Compute mean absolute error."""

    if not actuals:
        return 0.0
    return sum(abs(actual - prediction) for actual, prediction in zip(actuals, predictions)) / len(actuals)


def _rmse(actuals: list[float], predictions: list[float]) -> float:
    """Compute root mean squared error."""

    if not actuals:
        return 0.0
    mse = sum((actual - prediction) ** 2 for actual, prediction in zip(actuals, predictions)) / len(actuals)
    return math.sqrt(mse)


def _mape(actuals: list[float], predictions: list[float]) -> float:
    """Compute mean absolute percentage error."""

    if not actuals:
        return 0.0
    return (
        sum(
            abs((actual - prediction) / max(1e-6, actual))
            for actual, prediction in zip(actuals, predictions)
        )
        / len(actuals)
    )
