from predictor.serving import KernelLatencyPredictor, OperatorLatencyAggregator


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
