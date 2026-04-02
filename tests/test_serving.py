from dataclasses import replace

from predictor.serving import KernelLatencyPredictor, OperatorLatencyAggregator
from predictor.types import DEFAULT_DEVICE_PROFILE


def test_serving_pipeline_predicts_gemm_analytical_latency_and_aggregates() -> None:
    predictor = KernelLatencyPredictor.default()

    prediction = predictor.predict_from_raw_metadata(
        {
            "name": "gemm_demo_kernel",
            "family_hint": "gemm_bmm",
            "dtype": "fp16",
            "dimensions": {"m": 2048, "n": 2048, "k": 4096},
            "backend": "cuda",
        }
    )
    summary = OperatorLatencyAggregator().aggregate([prediction, prediction])

    assert prediction.kernel_name == "gemm_demo_kernel"
    assert prediction.implementation_bucket == "gemm.tensor_core"
    assert prediction.mean_latency_ms > 0.0
    assert summary.kernel_count == 2
    assert summary.total_mean_latency_ms == round(prediction.mean_latency_ms * 2, 4)


def test_serving_pipeline_latency_changes_with_device_profile() -> None:
    fast_predictor = KernelLatencyPredictor.default(
        device_profile=replace(
            DEFAULT_DEVICE_PROFILE,
            name="fast_serving_gpu",
            sm_count=132,
            memory_bandwidth_bytes_per_s=2.4e12,
            tensor_core_flops_per_s=420e12,
        )
    )
    slow_predictor = KernelLatencyPredictor.default(
        device_profile=replace(
            DEFAULT_DEVICE_PROFILE,
            name="slow_serving_gpu",
            sm_count=72,
            memory_bandwidth_bytes_per_s=0.9e12,
            tensor_core_flops_per_s=160e12,
        )
    )
    raw_metadata = {
        "name": "profile_sensitive_gemm",
        "family_hint": "gemm_bmm",
        "dtype": "fp16",
        "dimensions": {"m": 2048, "n": 2048, "k": 4096},
        "backend": "cuda",
    }

    fast_prediction = fast_predictor.predict_from_raw_metadata(raw_metadata)
    slow_prediction = slow_predictor.predict_from_raw_metadata(raw_metadata)

    assert fast_prediction.mean_latency_ms < slow_prediction.mean_latency_ms
