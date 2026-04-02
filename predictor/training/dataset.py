"""Training dataset builders."""

from __future__ import annotations

from collections.abc import Iterable

from predictor.types import FeatureVector, KernelDataset, KernelMetadata, TrainingSample


class KernelDatasetBuilder:
    """Build training datasets from normalized rows."""

    def build(
        self,
        rows: Iterable[tuple[KernelMetadata, FeatureVector, float]],
    ) -> KernelDataset:
        """Build a placeholder dataset for residual model training."""

        samples = tuple(
            TrainingSample(
                metadata=metadata,
                features=features,
                target_latency_ms=target_latency_ms,
            )
            for metadata, features, target_latency_ms in rows
        )
        return KernelDataset(samples=samples)
