from dataclasses import replace

from predictor.analytical import AnalyticalSchedulingSimulator
from predictor.types import DEFAULT_DEVICE_PROFILE, KernelMetadata, KernelTask, TaskPlan


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


def test_scheduler_wave_estimation_changes_with_sm_count() -> None:
    smaller_device = replace(DEFAULT_DEVICE_PROFILE, name="small_sm_gpu", sm_count=40)
    larger_device = replace(DEFAULT_DEVICE_PROFILE, name="large_sm_gpu", sm_count=160)
    small_scheduler = AnalyticalSchedulingSimulator(device_profile=smaller_device)
    large_scheduler = AnalyticalSchedulingSimulator(device_profile=larger_device)
    plan = TaskPlan(
        kernel_name="wave_sensitive_gemm",
        tasks=(
            KernelTask(name="load_lhs_tiles", description="load A"),
            KernelTask(name="load_rhs_tiles", description="load B"),
            KernelTask(name="matrix_multiply_accumulate", description="mma"),
            KernelTask(name="store_output_tiles", description="store C"),
        ),
    )
    metadata = KernelMetadata(
        name="wave_sensitive_gemm",
        dimensions={"m": 4096, "n": 4096, "k": 4096},
        dtype="fp16",
        backend="cuda",
    )

    small_schedule = small_scheduler.simulate(plan, metadata)
    large_schedule = large_scheduler.simulate(plan, metadata)

    assert small_schedule.estimated_waves > large_schedule.estimated_waves
    assert small_schedule.tiles_per_wave == 40
    assert large_schedule.tiles_per_wave == 160
