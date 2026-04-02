from predictor.analytical import AnalyticalSchedulingSimulator
from predictor.types import KernelMetadata, KernelTask, TaskPlan


def test_scheduler_estimates_multiple_waves_for_large_bmm() -> None:
    simulator = AnalyticalSchedulingSimulator()
    plan = TaskPlan(
        kernel_name="large_bmm",
        tasks=(
            KernelTask(name="load_lhs_tiles", description="load A"),
            KernelTask(name="load_rhs_tiles", description="load B"),
            KernelTask(name="matrix_multiply_accumulate", description="mma"),
            KernelTask(name="store_output_tiles", description="store C"),
        ),
    )
    metadata = KernelMetadata(
        name="large_bmm",
        family_hint=None,
        dimensions={"batch": 16, "m": 4096, "n": 4096, "k": 8192},
        dtype="fp16",
        backend="cuda",
        extra={"sm_count": 108},
    )

    schedule = simulator.simulate(plan, metadata)

    assert schedule.estimated_waves > 1
    assert schedule.sm_utilization == 1.0
