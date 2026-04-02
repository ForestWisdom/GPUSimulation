"""Serving-layer aggregation helpers."""

from __future__ import annotations

from collections.abc import Iterable

from predictor.types import AggregationSummary, LatencyPrediction


class OperatorLatencyAggregator:
    """Aggregate per-kernel predictions into an operator-level summary."""

    def aggregate(self, predictions: Iterable[LatencyPrediction]) -> AggregationSummary:
        """Aggregate kernel predictions into total mean and p90 latency."""

        prediction_list = list(predictions)
        return AggregationSummary(
            kernel_count=len(prediction_list),
            total_mean_latency_ms=round(
                sum(prediction.mean_latency_ms for prediction in prediction_list), 4
            ),
            total_p90_latency_ms=round(
                sum(prediction.p90_latency_ms for prediction in prediction_list), 4
            ),
        )
