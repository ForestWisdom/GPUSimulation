from predictor.analytical import PlaceholderSchedulingSimulator
from predictor.types import KernelMetadata, KernelTask, TaskPlan


def test_placeholder_scheduler_returns_single_wave_estimate() -> None:
    simulator = PlaceholderSchedulingSimulator()
    plan = TaskPlan(
        kernel_name="vector_kernel",
        tasks=(KernelTask(name="single_kernel", description="phase1 placeholder"),),
    )
    metadata = KernelMetadata(name="vector_kernel", dtype="fp16", backend="cuda")

    schedule = simulator.simulate(plan, metadata)

    assert schedule.estimated_waves == 1
    assert schedule.sm_utilization == 0.5
