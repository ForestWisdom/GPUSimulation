"""Serving orchestration for kernel latency prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from predictor.analytical import (
    AnalyticalBaselineLatencyEstimator,
    AnalyticalSchedulingSimulator,
    AnalyticalTaskDecomposer,
    PlaceholderPipelineFeatureAnalyzer,
)
from predictor.extractor import PlaceholderMetadataExtractor
from predictor.models import PlaceholderResidualModel, PlaceholderUncertaintyModel
from predictor.recognizer import HeuristicKernelRecognizer
from predictor.types import KernelMetadata, LatencyPrediction


@dataclass
class KernelLatencyPredictor:
    """Orchestrate the single-kernel prediction pipeline."""

    extractor: PlaceholderMetadataExtractor
    recognizer: HeuristicKernelRecognizer
    decomposer: AnalyticalTaskDecomposer
    scheduler: AnalyticalSchedulingSimulator
    feature_analyzer: PlaceholderPipelineFeatureAnalyzer
    baseline_estimator: AnalyticalBaselineLatencyEstimator
    residual_model: PlaceholderResidualModel
    uncertainty_model: PlaceholderUncertaintyModel

    @classmethod
    def default(cls) -> "KernelLatencyPredictor":
        """Build the default predictor stack."""

        return cls(
            extractor=PlaceholderMetadataExtractor(),
            recognizer=HeuristicKernelRecognizer(),
            decomposer=AnalyticalTaskDecomposer(),
            scheduler=AnalyticalSchedulingSimulator(),
            feature_analyzer=PlaceholderPipelineFeatureAnalyzer(),
            baseline_estimator=AnalyticalBaselineLatencyEstimator(),
            residual_model=PlaceholderResidualModel(),
            uncertainty_model=PlaceholderUncertaintyModel(),
        )

    def predict(self, metadata: KernelMetadata) -> LatencyPrediction:
        """Predict latency for a normalized kernel metadata object."""

        recognition = self.recognizer.recognize(metadata)
        plan = self.decomposer.decompose(metadata)
        schedule = self.scheduler.simulate(plan, metadata)
        features = self.feature_analyzer.analyze(metadata, schedule)
        baseline = self.baseline_estimator.estimate(metadata, plan, schedule, features)
        residual = self.residual_model.predict(features, baseline.latency_ms)
        mean_latency_ms = round(baseline.latency_ms + residual, 4)
        p90_latency_ms = self.uncertainty_model.predict_p90(features, mean_latency_ms)
        return LatencyPrediction(
            kernel_name=metadata.name,
            mean_latency_ms=mean_latency_ms,
            p90_latency_ms=p90_latency_ms,
            baseline_latency_ms=baseline.latency_ms,
            implementation_bucket=recognition.implementation_bucket,
        )

    def predict_from_raw_metadata(self, raw_metadata: Mapping[str, Any]) -> LatencyPrediction:
        """Predict latency for raw metadata provided as a dictionary."""

        metadata = self.extractor.parse_kernel_metadata(raw_metadata)
        return self.predict(metadata)
