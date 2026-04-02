"""Pipeline-aware feature analysis interfaces and placeholders."""

from __future__ import annotations

from typing import Protocol

from predictor.types import FeatureVector, KernelMetadata, ScheduleEstimate, is_gemm_bmm_kernel


class PipelineFeatureAnalyzer(Protocol):
    """Interface for deriving pipeline-aware features."""

    def analyze(self, metadata: KernelMetadata, schedule: ScheduleEstimate) -> FeatureVector:
        """Generate analytical and pipeline-aware features for a kernel."""


class PlaceholderPipelineFeatureAnalyzer:
    """Emit a compact, deterministic feature vector for Phase 1."""

    def analyze(self, metadata: KernelMetadata, schedule: ScheduleEstimate) -> FeatureVector:
        """Generate placeholder features from metadata and schedule estimates."""

        dtype_flag = 1.0 if metadata.dtype.lower() in {"fp16", "bf16", "half"} else 0.0
        batch = float(metadata.dimensions.get("batch", 1))
        m = float(metadata.dimensions.get("m", 0))
        n = float(metadata.dimensions.get("n", 0))
        k = float(metadata.dimensions.get("k", 0))
        total_flops = 2.0 * batch * m * n * k
        dtype_size = 2.0 if metadata.dtype.lower() in {"fp16", "bf16", "half"} else 4.0
        total_bytes = dtype_size * batch * ((m * k) + (k * n) + (m * n))
        arithmetic_intensity = total_flops / total_bytes if total_bytes else 0.0
        return FeatureVector(
            values={
                "dimension_count": float(len(metadata.dimensions)),
                "estimated_waves": float(schedule.estimated_waves),
                "sm_utilization": schedule.sm_utilization,
                "dtype_is_fp16_family": dtype_flag,
                "tag_count": float(len(metadata.tags)),
                "tile_count": float(schedule.tile_count),
                "batch": batch,
                "m": m,
                "n": n,
                "k": k,
                "flops": total_flops,
                "bytes": total_bytes,
                "arithmetic_intensity": arithmetic_intensity,
                "is_gemm_bmm": 1.0 if is_gemm_bmm_kernel(metadata) else 0.0,
            }
        )
