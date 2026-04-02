from __future__ import annotations

from dataclasses import replace

from predictor.serving import KernelLatencyPredictor
from predictor.types import DEFAULT_DEVICE_PROFILE, FeatureVector


class MockResidualModel:
    """Test double for the serving residual path."""

    def __init__(self, residual_ms: float) -> None:
        self.residual_ms = residual_ms

    def predict(self, features: FeatureVector) -> float:
        assert "flops" in features.values
        return self.residual_ms


def test_serving_pipeline_uses_mocked_residual_model() -> None:
    predictor = KernelLatencyPredictor.default(
        device_profile=replace(DEFAULT_DEVICE_PROFILE, name="mock_gpu"),
    )
    predictor.residual_model = MockResidualModel(residual_ms=0.05)

    prediction = predictor.predict_from_raw_metadata(
        {
            "name": "mocked_residual_gemm",
            "family_hint": "gemm_bmm",
            "dtype": "fp16",
            "dimensions": {"m": 1024, "n": 1024, "k": 2048},
            "backend": "cuda",
        }
    )

    assert prediction.mean_latency_ms == round(prediction.baseline_latency_ms + 0.05, 4)


def test_serving_pipeline_clamps_final_latency_to_positive_minimum() -> None:
    predictor = KernelLatencyPredictor.default()
    predictor.residual_model = MockResidualModel(residual_ms=-10.0)

    prediction = predictor.predict_from_raw_metadata(
        {
            "name": "clamped_gemm",
            "family_hint": "gemm_bmm",
            "dtype": "fp16",
            "dimensions": {"m": 1024, "n": 1024, "k": 2048},
            "backend": "cuda",
        }
    )

    assert prediction.mean_latency_ms == 1e-06
