"""Uncertainty model interfaces and placeholders."""

from __future__ import annotations

from typing import Protocol

from predictor.types import FeatureVector


class UncertaintyModel(Protocol):
    """Interface for uncertainty or p90 prediction."""

    def predict_p90(self, features: FeatureVector, mean_latency_ms: float) -> float:
        """Predict a p90 latency value for a kernel."""


class PlaceholderUncertaintyModel:
    """Return a deterministic p90 uplift for the scaffold."""

    def predict_p90(self, features: FeatureVector, mean_latency_ms: float) -> float:
        """Predict a placeholder p90 latency."""

        del features
        return round(mean_latency_ms * 1.1, 4)
