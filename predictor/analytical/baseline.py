"""Baseline latency interfaces and placeholders."""

from __future__ import annotations

from typing import Protocol

from predictor.types import (
    BaselineEstimate,
    DEFAULT_DEVICE_PROFILE,
    DeviceProfile,
    FeatureVector,
    KernelMetadata,
    ScheduleEstimate,
    TaskPlan,
    is_gemm_bmm_kernel,
    uses_tensor_cores,
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

    def __init__(
        self,
        device_profile: DeviceProfile = DEFAULT_DEVICE_PROFILE,
    ) -> None:
        """Initialize the estimator with a placeholder fallback path."""

        self._fallback = PlaceholderBaselineLatencyEstimator()
        self.device_profile = device_profile

    def estimate(
        self,
        metadata: KernelMetadata,
        plan: TaskPlan,
        schedule: ScheduleEstimate,
        features: FeatureVector,
    ) -> BaselineEstimate:
        """Estimate a baseline latency before learned corrections."""

        del plan
        if not is_gemm_bmm_kernel(metadata):
            return self._fallback.estimate(
                metadata,
                TaskPlan(metadata.name, ()),
                schedule,
                FeatureVector({}),
            )

        use_tensor_cores = bool(
            features.values.get("uses_tensor_cores", float(uses_tensor_cores(metadata))),
        )
        total_flops = features.values.get("flops", 0.0)
        total_bytes = features.values.get("bytes", 0.0)
        compute_throughput = features.values.get(
            "peak_compute_flops",
            self.device_profile.peak_flops_for(use_tensor_cores),
        )
        memory_bandwidth = features.values.get(
            "memory_bandwidth_bytes_per_s",
            self.device_profile.memory_bandwidth_bytes_per_s,
        )
        arithmetic_intensity = features.values.get("arithmetic_intensity", 0.0)

        compute_ms = (total_flops / compute_throughput) * 1e3 if compute_throughput else 0.0
        memory_ms = (total_bytes / memory_bandwidth) * 1e3
        wave_penalty_ms = (
            schedule.estimated_waves
            * self.device_profile.wave_penalty_ms_for(use_tensor_cores)
        )
        launch_overhead_ms = self.device_profile.launch_overhead_ms_for(
            use_tensor_cores,
        )

        latency_ms = max(compute_ms, memory_ms) + wave_penalty_ms + launch_overhead_ms
        path_name = "tensor_core" if use_tensor_cores else "simt"
        return BaselineEstimate(
            latency_ms=round(latency_ms, 4),
            notes=(
                f"device={self.device_profile.name}; "
                f"path={path_name}; "
                f"compute_ms={compute_ms:.6f}; "
                f"memory_ms={memory_ms:.6f}; "
                f"arithmetic_intensity={arithmetic_intensity:.6f}; "
                f"waves={schedule.estimated_waves}; "
                f"tile_count={schedule.tile_count}"
            ),
        )
