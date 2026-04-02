"""Kernel task decomposition interfaces and placeholders."""

from __future__ import annotations

from typing import Protocol

from predictor.types import KernelMetadata, KernelTask, TaskPlan, is_gemm_bmm_kernel


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


class AnalyticalTaskDecomposer:
    """Create analytical task plans for GEMM/BMM kernels."""

    def __init__(self) -> None:
        """Initialize the decomposer with a placeholder fallback path."""

        self._fallback = PlaceholderTaskDecomposer()

    def decompose(self, metadata: KernelMetadata) -> TaskPlan:
        """Create an analytical task plan for a kernel."""

        if not is_gemm_bmm_kernel(metadata):
            return self._fallback.decompose(metadata)

        return TaskPlan(
            kernel_name=metadata.name,
            tasks=(
                KernelTask(name="load_lhs_tiles", description="Load tiled matrix A operands."),
                KernelTask(name="load_rhs_tiles", description="Load tiled matrix B operands."),
                KernelTask(
                    name="matrix_multiply_accumulate",
                    description="Run the main GEMM/BMM MMA loop.",
                ),
                KernelTask(
                    name="store_output_tiles",
                    description="Write the output tiles back to global memory.",
                ),
            ),
        )
