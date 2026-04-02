from predictor.analytical import (
    AnalyticalBaselineLatencyEstimator,
    AnalyticalSchedulingSimulator,
    AnalyticalTaskDecomposer,
    PlaceholderPipelineFeatureAnalyzer,
)
from predictor.types import KernelFamily, KernelMetadata


def test_baseline_estimator_scales_with_gemm_problem_size() -> None:
    estimator = AnalyticalBaselineLatencyEstimator()
    decomposer = AnalyticalTaskDecomposer()
    scheduler = AnalyticalSchedulingSimulator()
    analyzer = PlaceholderPipelineFeatureAnalyzer()

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
