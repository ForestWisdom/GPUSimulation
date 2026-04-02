"""Residual model implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from joblib import dump, load
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from predictor.types import FeatureVector


class ResidualModel(Protocol):
    """Interface for learned residual correction models."""

    def predict(self, features: FeatureVector) -> float:
        """Predict an additive residual latency correction."""


class ResidualRidgeModel:
    """Small trainable Ridge-regression residual model."""

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        feature_names: tuple[str, ...] | None = None,
        is_fitted: bool = False,
    ) -> None:
        """Initialize the residual model."""

        self.pipeline = pipeline or Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        self.feature_names = feature_names or ()
        self.is_fitted = is_fitted

    def fit(
        self,
        features: list[FeatureVector],
        targets: list[float],
    ) -> "ResidualRidgeModel":
        """Fit the residual model and return itself."""

        self.feature_names = self._derive_feature_names(features)
        design_matrix = self._build_matrix(features)
        self.pipeline.fit(design_matrix, np.asarray(targets, dtype=float))
        self.is_fitted = True
        return self

    def predict(self, features: FeatureVector) -> float:
        """Predict the additive residual in milliseconds."""

        if not self.is_fitted:
            return 0.0
        prediction = self.pipeline.predict(self._build_matrix([features]))
        return float(prediction[0])

    def predict_batch(self, features: list[FeatureVector]) -> list[float]:
        """Predict additive residuals for a batch of features."""

        if not self.is_fitted:
            return [0.0 for _ in features]
        predictions = self.pipeline.predict(self._build_matrix(features))
        return [float(value) for value in predictions]

    def save(self, path: str | Path) -> None:
        """Persist the sklearn pipeline and feature schema."""

        dump(
            {
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ResidualRidgeModel":
        """Load a persisted residual model."""

        payload = load(path)
        return cls(
            pipeline=payload["pipeline"],
            feature_names=tuple(payload["feature_names"]),
            is_fitted=bool(payload.get("is_fitted", True)),
        )

    def _derive_feature_names(self, features: list[FeatureVector]) -> tuple[str, ...]:
        """Derive a stable feature order from training data."""

        feature_names: set[str] = set()
        for feature_vector in features:
            feature_names.update(feature_vector.values.keys())
        return tuple(sorted(feature_names))

    def _build_matrix(self, features: list[FeatureVector]) -> np.ndarray:
        """Convert feature vectors into a numeric design matrix."""

        if not self.feature_names:
            self.feature_names = self._derive_feature_names(features)
        return np.asarray(
            [
                [float(feature.values.get(name, 0.0)) for name in self.feature_names]
                for feature in features
            ],
            dtype=float,
        )


class PlaceholderResidualModel(ResidualRidgeModel):
    """Backward-compatible alias for the trainable residual model."""
