from predictor.analytical import PlaceholderBaselineLatencyEstimator
from predictor.models import PlaceholderResidualModel, PlaceholderUncertaintyModel
from predictor.types import FeatureVector, KernelMetadata, ScheduleEstimate, TaskPlan


def test_placeholder_models_produce_mean_and_p90_outputs() -> None:
    estimator = PlaceholderBaselineLatencyEstimator()
    residual_model = PlaceholderResidualModel()
    uncertainty_model = PlaceholderUncertaintyModel()
    metadata = KernelMetadata(name="moe_kernel", dtype="fp16", backend="cuda")
    plan = TaskPlan(kernel_name="moe_kernel", tasks=())
    schedule = ScheduleEstimate(estimated_waves=1, sm_utilization=0.5)
    features = FeatureVector(values={"estimated_waves": 1.0, "task_count": 0.0})

    baseline = estimator.estimate(metadata, plan, schedule, features)
    residual = residual_model.predict(features, baseline.latency_ms)
    p90 = uncertainty_model.predict_p90(features, baseline.latency_ms + residual)

    assert baseline.latency_ms == 1.0
    assert residual == 0.0
    assert p90 == 1.1
