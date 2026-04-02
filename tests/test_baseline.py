from dataclasses import replace

from predictor.analytical import (
    AnalyticalBaselineLatencyEstimator,
    AnalyticalPipelineFeatureAnalyzer,
    AnalyticalSchedulingSimulator,
    AnalyticalTaskDecomposer,
)
from predictor.types import DEFAULT_DEVICE_PROFILE, KernelFamily, KernelMetadata


def test_baseline_estimator_scales_with_gemm_problem_size() -> None:
    estimator = AnalyticalBaselineLatencyEstimator()
    decomposer = AnalyticalTaskDecomposer()
    scheduler = AnalyticalSchedulingSimulator()
    analyzer = AnalyticalPipelineFeatureAnalyzer()

    small_metadata = KernelMetadata(
        name="small_gemm",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 256, "n": 256, "k": 256},
        dtype="fp16",
        backend="cuda",
    )
    large_metadata = KernelMetadata(
        name="large_gemm",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 4096, "n": 4096, "k": 4096},
        dtype="fp16",
        backend="cuda",
    )

    small_plan = decomposer.decompose(small_metadata)
    large_plan = decomposer.decompose(large_metadata)
    small_schedule = scheduler.simulate(small_plan, small_metadata)
    large_schedule = scheduler.simulate(large_plan, large_metadata)
    small_features = analyzer.analyze(small_metadata, small_schedule)
    large_features = analyzer.analyze(large_metadata, large_schedule)

    small_estimate = estimator.estimate(
        small_metadata, small_plan, small_schedule, small_features
    )
    large_estimate = estimator.estimate(
        large_metadata, large_plan, large_schedule, large_features
    )

    assert small_estimate.latency_ms > 0.0
    assert large_estimate.latency_ms > small_estimate.latency_ms
    assert "compute_ms" in large_estimate.notes


def test_baseline_estimator_prefers_tensor_core_path_for_eligible_gemm() -> None:
    estimator = AnalyticalBaselineLatencyEstimator()
    decomposer = AnalyticalTaskDecomposer()
    scheduler = AnalyticalSchedulingSimulator()
    analyzer = AnalyticalPipelineFeatureAnalyzer()

    tensor_core_metadata = KernelMetadata(
        name="tensor_core_gemm",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 2048, "n": 2048, "k": 4096},
        dtype="fp16",
        backend="cuda",
    )
    simt_metadata = KernelMetadata(
        name="simt_gemm",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 2048, "n": 2048, "k": 4096},
        dtype="fp32",
        backend="cuda",
    )

    tensor_core_plan = decomposer.decompose(tensor_core_metadata)
    simt_plan = decomposer.decompose(simt_metadata)
    tensor_core_schedule = scheduler.simulate(tensor_core_plan, tensor_core_metadata)
    simt_schedule = scheduler.simulate(simt_plan, simt_metadata)
    tensor_core_features = analyzer.analyze(tensor_core_metadata, tensor_core_schedule)
    simt_features = analyzer.analyze(simt_metadata, simt_schedule)

    tensor_core_estimate = estimator.estimate(
        tensor_core_metadata,
        tensor_core_plan,
        tensor_core_schedule,
        tensor_core_features,
    )
    simt_estimate = estimator.estimate(
        simt_metadata,
        simt_plan,
        simt_schedule,
        simt_features,
    )

    assert tensor_core_estimate.latency_ms < simt_estimate.latency_ms
    assert "tensor_core" in tensor_core_estimate.notes
    assert "simt" in simt_estimate.notes


def test_baseline_estimator_depends_on_device_profile() -> None:
    faster_device = replace(
        DEFAULT_DEVICE_PROFILE,
        name="fast_gpu",
        sm_count=132,
        memory_bandwidth_bytes_per_s=2.4e12,
        tensor_core_flops_per_s=420e12,
        simt_flops_per_s=30e12,
    )
    slower_device = replace(
        DEFAULT_DEVICE_PROFILE,
        name="slow_gpu",
        sm_count=80,
        memory_bandwidth_bytes_per_s=0.9e12,
        tensor_core_flops_per_s=160e12,
        simt_flops_per_s=12e12,
    )
    metadata = KernelMetadata(
        name="profile_sensitive_bmm",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"batch": 4, "m": 2048, "n": 2048, "k": 4096},
        dtype="fp16",
        backend="cuda",
    )
    plan = AnalyticalTaskDecomposer().decompose(metadata)

    fast_scheduler = AnalyticalSchedulingSimulator(device_profile=faster_device)
    slow_scheduler = AnalyticalSchedulingSimulator(device_profile=slower_device)
    fast_analyzer = AnalyticalPipelineFeatureAnalyzer(device_profile=faster_device)
    slow_analyzer = AnalyticalPipelineFeatureAnalyzer(device_profile=slower_device)
    fast_estimator = AnalyticalBaselineLatencyEstimator(device_profile=faster_device)
    slow_estimator = AnalyticalBaselineLatencyEstimator(device_profile=slower_device)

    fast_schedule = fast_scheduler.simulate(plan, metadata)
    slow_schedule = slow_scheduler.simulate(plan, metadata)
    fast_features = fast_analyzer.analyze(metadata, fast_schedule)
    slow_features = slow_analyzer.analyze(metadata, slow_schedule)

    fast_estimate = fast_estimator.estimate(metadata, plan, fast_schedule, fast_features)
    slow_estimate = slow_estimator.estimate(metadata, plan, slow_schedule, slow_features)

    assert fast_estimate.latency_ms < slow_estimate.latency_ms
    assert "fast_gpu" in fast_estimate.notes
    assert "slow_gpu" in slow_estimate.notes
