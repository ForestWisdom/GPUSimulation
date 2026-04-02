"""Residual model interfaces and placeholders."""

from __future__ import annotations

from typing import Protocol

from predictor.types import FeatureVector


class ResidualModel(Protocol):
    """Interface for learned residual correction models."""

    def predict(self, features: FeatureVector, baseline_latency_ms: float) -> float:
        """Predict an additive residual latency correction."""


class PlaceholderResidualModel:
    """Return a zero residual for the scaffold."""

    def predict(self, features: FeatureVector, baseline_latency_ms: float) -> float:
        """Predict a placeholder residual correction."""

        del features, baseline_latency_ms
        return 0.0
