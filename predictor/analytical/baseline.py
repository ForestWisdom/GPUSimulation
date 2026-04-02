"""Baseline latency interfaces and placeholders."""

from __future__ import annotations

import math
from typing import Protocol

from predictor.types import (
    BaselineEstimate,
    FeatureVector,
    KernelMetadata,
    ScheduleEstimate,
    TaskPlan,
    is_gemm_bmm_kernel,
)


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


class AnalyticalBaselineLatencyEstimator:
    """Estimate GEMM/BMM latency with a lightweight analytical model."""

    def __init__(self) -> None:
        """Initialize the estimator with a placeholder fallback path."""

        self._fallback = PlaceholderBaselineLatencyEstimator()

    def estimate(
        self,
        metadata: KernelMetadata,
        plan: TaskPlan,
        schedule: ScheduleEstimate,
        features: FeatureVector,
    ) -> BaselineEstimate:
        """Estimate a baseline latency before learned corrections."""

        del plan, features
        if not is_gemm_bmm_kernel(metadata):
            return self._fallback.estimate(metadata, TaskPlan(metadata.name, ()), schedule, FeatureVector({}))

        batch = max(1, int(metadata.dimensions.get("batch", 1)))
        m = max(1, int(metadata.dimensions.get("m", 1)))
        n = max(1, int(metadata.dimensions.get("n", 1)))
        k = max(1, int(metadata.dimensions.get("k", 1)))
        dtype_size_bytes = _dtype_size_bytes(metadata.dtype)
        total_flops = 2.0 * batch * m * n * k
        total_bytes = dtype_size_bytes * batch * ((m * k) + (k * n) + (m * n))

        tensor_core = _is_tensor_core_path(metadata)
        compute_throughput = 312e12 if tensor_core else 19.5e12
        memory_bandwidth = 1.555e12
        compute_ms = (total_flops / compute_throughput) * 1e3
        memory_ms = (total_bytes / memory_bandwidth) * 1e3
        wave_penalty_ms = schedule.estimated_waves * (0.003 if tensor_core else 0.006)
        launch_overhead_ms = 0.010 if tensor_core else 0.014

        latency_ms = max(compute_ms, memory_ms) + wave_penalty_ms + launch_overhead_ms
        return BaselineEstimate(
            latency_ms=round(latency_ms, 4),
            notes=(
                "gemm_bmm analytical baseline; "
                f"compute_ms={compute_ms:.6f}; "
                f"memory_ms={memory_ms:.6f}; "
                f"waves={schedule.estimated_waves}; "
                f"tile_count={schedule.tile_count}"
            ),
        )


def _dtype_size_bytes(dtype: str) -> int:
    """Return the byte width for a kernel datatype."""

    dtype_key = dtype.lower()
    if dtype_key in {"fp16", "bf16", "half"}:
        return 2
    if dtype_key in {"fp32", "float32"}:
        return 4
    return 4


def _is_tensor_core_path(metadata: KernelMetadata) -> bool:
    """Return whether the analytical model should use tensor-core assumptions."""

    dtype = metadata.dtype.lower()
    if dtype not in {"fp16", "bf16", "half"}:
        return False
    m = int(metadata.dimensions.get("m", 0))
    n = int(metadata.dimensions.get("n", 0))
    k = int(metadata.dimensions.get("k", 0))
    return min(m, n, k) > 0 and all(dimension % 16 == 0 for dimension in (m, n, k))
