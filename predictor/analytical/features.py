"""Pipeline-aware feature analysis interfaces and placeholders."""

from __future__ import annotations

from typing import Protocol

from predictor.types import FeatureVector, KernelMetadata, ScheduleEstimate


class PipelineFeatureAnalyzer(Protocol):
    """Interface for deriving pipeline-aware features."""

    def analyze(self, metadata: KernelMetadata, schedule: ScheduleEstimate) -> FeatureVector:
        """Generate analytical and pipeline-aware features for a kernel."""


class PlaceholderPipelineFeatureAnalyzer:
    """Emit a compact, deterministic feature vector for Phase 1."""

    def analyze(self, metadata: KernelMetadata, schedule: ScheduleEstimate) -> FeatureVector:
        """Generate placeholder features from metadata and schedule estimates."""

        dtype_flag = 1.0 if metadata.dtype.lower() in {"fp16", "bf16", "half"} else 0.0
        return FeatureVector(
            values={
                "dimension_count": float(len(metadata.dimensions)),
                "estimated_waves": float(schedule.estimated_waves),
                "sm_utilization": schedule.sm_utilization,
                "dtype_is_fp16_family": dtype_flag,
                "tag_count": float(len(metadata.tags)),
            }
        )
