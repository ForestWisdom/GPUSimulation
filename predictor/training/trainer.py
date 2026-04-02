"""Residual-model training helpers."""

from __future__ import annotations

import random

from predictor.models import ResidualRidgeModel
from predictor.types import (
    KernelDataset,
    ResidualTrainerState,
    ResidualTrainingDataset,
    ResidualTrainingSample,
    TrainerState,
)


class PlaceholderTrainer:
    """Record dataset statistics instead of fitting a real model."""

    def fit(self, dataset: KernelDataset) -> TrainerState:
        """Fit a placeholder trainer and return its state."""

        return TrainerState(
            model_name="phase1-placeholder-residual",
            sample_count=len(dataset.samples),
        )


class ResidualTrainer:
    """Fit the Phase 3 GEMM/BMM residual model."""

    def fit(self, dataset: ResidualTrainingDataset) -> ResidualTrainerState:
        """Fit a Ridge-based residual model on the provided dataset."""

        model = ResidualRidgeModel().fit(
            [sample.features for sample in dataset.samples],
            [sample.residual_target_ms for sample in dataset.samples],
        )
        return ResidualTrainerState(
            model_name="ridge_additive_residual",
            sample_count=len(dataset.samples),
            model=model,
            feature_names=model.feature_names,
        )


def split_dataset(
    dataset: ResidualTrainingDataset,
    test_fraction: float = 0.2,
    random_seed: int = 7,
) -> tuple[ResidualTrainingDataset, ResidualTrainingDataset]:
    """Split a dataset into deterministic train/test partitions."""

    samples = list(dataset.samples)
    if len(samples) < 2:
        return dataset, ResidualTrainingDataset(samples=())

    rng = random.Random(random_seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    test_size = max(1, int(round(len(samples) * test_fraction)))
    test_indices = set(indices[:test_size])
    train_samples: list[ResidualTrainingSample] = []
    test_samples: list[ResidualTrainingSample] = []
    for index, sample in enumerate(samples):
        if index in test_indices:
            test_samples.append(sample)
        else:
            train_samples.append(sample)
    return (
        ResidualTrainingDataset(samples=tuple(train_samples)),
        ResidualTrainingDataset(samples=tuple(test_samples)),
    )
