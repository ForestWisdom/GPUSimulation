"""Baseline latency interfaces and placeholders."""

from __future__ import annotations

from typing import Protocol

from predictor.types import BaselineEstimate, FeatureVector, KernelMetadata, ScheduleEstimate, TaskPlan


class BaselineLatencyEstimator(Protocol):
    """Interface for analytical baseline latency estimation."""

    def estimate(
        self,
        metadata: KernelMetadata,
        plan: TaskPlan,
        schedule: ScheduleEstimate,
        features: FeatureVector,
    ) -> BaselineEstimate:
        """Estimate a baseline latency before learned corrections."""


class PlaceholderBaselineLatencyEstimator:
    """Return a fixed analytical latency estimate for Phase 1."""

    def estimate(
        self,
        metadata: KernelMetadata,
        plan: TaskPlan,
        schedule: ScheduleEstimate,
        features: FeatureVector,
    ) -> BaselineEstimate:
        """Estimate a placeholder baseline latency."""

        del metadata, plan, schedule, features
        return BaselineEstimate(
            latency_ms=1.0,
            notes="Phase 1 placeholder analytical baseline.",
        )
