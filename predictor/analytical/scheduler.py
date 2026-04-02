"""Scheduling simulator interfaces and placeholders."""

from __future__ import annotations

import math
from typing import Protocol

from predictor.types import (
    DEFAULT_DEVICE_PROFILE,
    DeviceProfile,
    KernelMetadata,
    ScheduleEstimate,
    TaskPlan,
    is_gemm_bmm_kernel,
    uses_tensor_cores,
)


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

    def __init__(
        self,
        device_profile: DeviceProfile = DEFAULT_DEVICE_PROFILE,
    ) -> None:
        """Initialize the simulator with a placeholder fallback path."""

        self._fallback = PlaceholderSchedulingSimulator()
        self.device_profile = device_profile

    def simulate(self, plan: TaskPlan, metadata: KernelMetadata) -> ScheduleEstimate:
        """Simulate a kernel task plan and return schedule estimates."""

        if not is_gemm_bmm_kernel(metadata):
            return self._fallback.simulate(plan, metadata)

        del plan
        tile_m, tile_n = self.device_profile.tile_shape_for(
            uses_tensor_cores(metadata),
        )
        batch = int(metadata.dimensions.get("batch", 1))
        m = max(1, int(metadata.dimensions.get("m", 1)))
        n = max(1, int(metadata.dimensions.get("n", 1)))
        sm_count = max(1, self.device_profile.sm_count)

        tile_count = batch * math.ceil(m / tile_m) * math.ceil(n / tile_n)
        estimated_waves = max(1, math.ceil(tile_count / sm_count))
        sm_utilization = min(1.0, tile_count / sm_count)
        return ScheduleEstimate(
            estimated_waves=estimated_waves,
            sm_utilization=round(sm_utilization, 4),
            tile_count=tile_count,
            tiles_per_wave=sm_count,
        )
