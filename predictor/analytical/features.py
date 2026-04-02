"""Pipeline-aware feature analysis interfaces and analytical GEMM/BMM features."""

from __future__ import annotations

from typing import Protocol

from predictor.types import (
    DEFAULT_DEVICE_PROFILE,
    DeviceProfile,
    FeatureVector,
    KernelMetadata,
    ScheduleEstimate,
    dtype_size_bytes,
    is_gemm_bmm_kernel,
    uses_tensor_cores,
)


class PipelineFeatureAnalyzer(Protocol):
    """Interface for deriving pipeline-aware features."""

    def analyze(self, metadata: KernelMetadata, schedule: ScheduleEstimate) -> FeatureVector:
        """Generate analytical and pipeline-aware features for a kernel."""


class AnalyticalPipelineFeatureAnalyzer:
    """Generate analytical GEMM/BMM features plus a generic fallback."""

    def __init__(
        self,
        device_profile: DeviceProfile = DEFAULT_DEVICE_PROFILE,
    ) -> None:
        """Initialize the analyzer with the active device profile."""

        self.device_profile = device_profile

    def analyze(self, metadata: KernelMetadata, schedule: ScheduleEstimate) -> FeatureVector:
        """Generate analytical and pipeline-aware features for a kernel."""

        if not is_gemm_bmm_kernel(metadata):
            return self._analyze_generic(metadata, schedule)

        use_tensor_cores = uses_tensor_cores(metadata)
        batch = float(metadata.dimensions.get("batch", 1))
        m = float(metadata.dimensions.get("m", 0))
        n = float(metadata.dimensions.get("n", 0))
        k = float(metadata.dimensions.get("k", 0))
        total_flops = 2.0 * batch * m * n * k
        total_bytes = float(dtype_size_bytes(metadata.dtype)) * batch * (
            (m * k) + (k * n) + (m * n)
        )
        arithmetic_intensity = total_flops / total_bytes if total_bytes else 0.0
        tile_m, tile_n = self.device_profile.tile_shape_for(use_tensor_cores)
        peak_compute_flops = self.device_profile.peak_flops_for(use_tensor_cores)
        theoretical_wave_slots = max(
            1,
            schedule.estimated_waves * max(1, schedule.tiles_per_wave),
        )
        tile_occupancy = schedule.tile_count / theoretical_wave_slots

        return FeatureVector(
            values={
                "dimension_count": float(len(metadata.dimensions)),
                "estimated_waves": float(schedule.estimated_waves),
                "sm_utilization": schedule.sm_utilization,
                "dtype_is_fp16_family": 1.0 if use_tensor_cores else 0.0,
                "tag_count": float(len(metadata.tags)),
                "tile_count": float(schedule.tile_count),
                "tiles_per_wave": float(schedule.tiles_per_wave),
                "batch": batch,
                "m": m,
                "n": n,
                "k": k,
                "flops": total_flops,
                "bytes": total_bytes,
                "arithmetic_intensity": arithmetic_intensity,
                "is_gemm_bmm": 1.0,
                "uses_tensor_cores": 1.0 if use_tensor_cores else 0.0,
                "device_sm_count": float(self.device_profile.sm_count),
                "peak_compute_flops": peak_compute_flops,
                "memory_bandwidth_bytes_per_s": self.device_profile.memory_bandwidth_bytes_per_s,
                "tile_m": float(tile_m),
                "tile_n": float(tile_n),
                "tile_occupancy": tile_occupancy,
                "is_bmm": 1.0 if batch > 1.0 else 0.0,
            }
        )

    def _analyze_generic(
        self,
        metadata: KernelMetadata,
        schedule: ScheduleEstimate,
    ) -> FeatureVector:
        """Generate generic fallback features for non-GEMM/BMM kernels."""

        dtype_flag = 1.0 if metadata.dtype.lower() in {"fp16", "bf16", "half"} else 0.0
        return FeatureVector(
            values={
                "dimension_count": float(len(metadata.dimensions)),
                "estimated_waves": float(schedule.estimated_waves),
                "sm_utilization": schedule.sm_utilization,
                "dtype_is_fp16_family": dtype_flag,
                "tag_count": float(len(metadata.tags)),
                "tile_count": float(schedule.tile_count),
                "tiles_per_wave": float(schedule.tiles_per_wave),
                "is_gemm_bmm": 0.0,
                "device_sm_count": float(self.device_profile.sm_count),
            }
        )


class PlaceholderPipelineFeatureAnalyzer(AnalyticalPipelineFeatureAnalyzer):
    """Backward-compatible alias for the analytical feature analyzer."""
