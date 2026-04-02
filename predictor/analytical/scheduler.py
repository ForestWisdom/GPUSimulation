"""Scheduling simulator interfaces and placeholders."""

from __future__ import annotations

from typing import Protocol

from predictor.types import KernelMetadata, ScheduleEstimate, TaskPlan


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
        return ScheduleEstimate(estimated_waves=estimated_waves, sm_utilization=0.5)
