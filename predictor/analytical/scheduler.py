"""Scheduling simulator interfaces and placeholders."""

from __future__ import annotations

import math
from typing import Protocol

from predictor.types import KernelMetadata, ScheduleEstimate, TaskPlan, is_gemm_bmm_kernel


class SchedulingSimulator(Protocol):
    """Interface for scheduling simulation."""

    def simulate(self, plan: TaskPlan, metadata: KernelMetadata) -> ScheduleEstimate:
        """Simulate a kernel task plan and return schedule estimates."""


class PlaceholderSchedulingSimulator:
    """Return fixed scheduling estimates for the scaffold."""

    def simulate(self, plan: TaskPlan, metadata: KernelMetadata) -> ScheduleEstimate:
        """Simulate a placeholder schedule for a task plan."""

        del metadata
        estimated_waves = max(1, len(plan.tasks))
        return ScheduleEstimate(
            estimated_waves=estimated_waves,
            sm_utilization=0.5,
            tile_count=len(plan.tasks),
            tiles_per_wave=len(plan.tasks),
        )


class AnalyticalSchedulingSimulator:
    """Estimate GEMM/BMM wave count from threadblock tiling."""

    def __init__(self) -> None:
        """Initialize the simulator with a placeholder fallback path."""

        self._fallback = PlaceholderSchedulingSimulator()

    def simulate(self, plan: TaskPlan, metadata: KernelMetadata) -> ScheduleEstimate:
        """Simulate a kernel task plan and return schedule estimates."""

        if not is_gemm_bmm_kernel(metadata):
            return self._fallback.simulate(plan, metadata)

        tile_m, tile_n = _select_output_tile_shape(metadata)
        batch = int(metadata.dimensions.get("batch", 1))
        m = max(1, int(metadata.dimensions.get("m", 1)))
        n = max(1, int(metadata.dimensions.get("n", 1)))
        sm_count = max(1, int(metadata.extra.get("sm_count", 108)))

        tile_count = batch * math.ceil(m / tile_m) * math.ceil(n / tile_n)
        estimated_waves = max(1, math.ceil(tile_count / sm_count))
        sm_utilization = min(1.0, tile_count / sm_count)
        return ScheduleEstimate(
            estimated_waves=estimated_waves,
            sm_utilization=round(sm_utilization, 4),
            tile_count=tile_count,
            tiles_per_wave=sm_count,
        )


def _select_output_tile_shape(metadata: KernelMetadata) -> tuple[int, int]:
    """Select an output tile shape for analytical wave estimation."""

    dtype = metadata.dtype.lower()
    m = int(metadata.dimensions.get("m", 0))
    n = int(metadata.dimensions.get("n", 0))
    k = int(metadata.dimensions.get("k", 0))
    if dtype in {"fp16", "bf16", "half"} and (m % 16 == 0) and (n % 16 == 0) and (k % 16 == 0):
        return 128, 128
    return 64, 64
