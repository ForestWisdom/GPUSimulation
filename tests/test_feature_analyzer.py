from predictor.analytical import PlaceholderPipelineFeatureAnalyzer
from predictor.types import KernelMetadata, ScheduleEstimate


def test_placeholder_feature_analyzer_emits_feature_vector() -> None:
    analyzer = PlaceholderPipelineFeatureAnalyzer()
    metadata = KernelMetadata(
        name="norm_kernel",
        dimensions={"tokens": 256, "hidden": 4096},
        dtype="bf16",
        backend="cuda",
    )
    schedule = ScheduleEstimate(estimated_waves=2, sm_utilization=0.5)

    features = analyzer.analyze(metadata, schedule)

    assert features.values["dimension_count"] == 2.0
    assert features.values["estimated_waves"] == 2.0
    assert features.values["dtype_is_fp16_family"] == 1.0
