"""Kernel task decomposition interfaces and placeholders."""

from __future__ import annotations

from typing import Protocol

from predictor.types import KernelMetadata, KernelTask, TaskPlan


class TaskDecomposer(Protocol):
    """Interface for decomposing kernels into analytical tasks."""

    def decompose(self, metadata: KernelMetadata) -> TaskPlan:
        """Create an analytical task plan for a kernel."""


class PlaceholderTaskDecomposer:
    """Return a minimal single-task decomposition for Phase 1."""

    def decompose(self, metadata: KernelMetadata) -> TaskPlan:
        """Decompose a kernel into a single placeholder task."""

        return TaskPlan(
            kernel_name=metadata.name,
            tasks=(
                KernelTask(
                    name="single_kernel",
                    description="Phase 1 placeholder decomposition for a single kernel.",
                ),
            ),
        )
