from predictor.training import (
    KernelDatasetBuilder,
    PlaceholderEvaluator,
    PlaceholderTrainer,
)
from predictor.types import FeatureVector, KernelMetadata, LatencyPrediction


def test_training_placeholders_build_and_evaluate_dataset() -> None:
    builder = KernelDatasetBuilder()
    trainer = PlaceholderTrainer()
    evaluator = PlaceholderEvaluator()
    metadata = KernelMetadata(name="gemm_kernel", dtype="fp16", backend="cuda")
    features = FeatureVector(values={"estimated_waves": 1.0})
    prediction = LatencyPrediction(
        kernel_name="gemm_kernel",
        mean_latency_ms=1.0,
        p90_latency_ms=1.1,
        baseline_latency_ms=1.0,
        implementation_bucket="gemm_bmm.placeholder",
    )

    dataset = builder.build([(metadata, features, prediction.mean_latency_ms)])
    trainer_state = trainer.fit(dataset)
    metrics = evaluator.evaluate(dataset, trainer_state)

    assert len(dataset.samples) == 1
    assert trainer_state.sample_count == 1
    assert metrics["mae"] == 0.0
