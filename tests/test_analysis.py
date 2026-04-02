from __future__ import annotations

from predictor.models import ResidualRidgeModel
from predictor.types import FeatureVector, KernelFamily, KernelMetadata


def test_size_bucket_derivation_is_explicit() -> None:
    from predictor.training.analysis import SIZE_BUCKET_RULE, derive_size_bucket

    bucket, rule = derive_size_bucket({"m": 1024, "n": 1024, "k": 2048})

    assert bucket == "small"
    assert rule == SIZE_BUCKET_RULE

    medium_bucket, _ = derive_size_bucket({"m": 1024, "n": 1024, "k": 8192})
    large_bucket, _ = derive_size_bucket({"m": 4096, "n": 4096, "k": 16384})

    assert medium_bucket == "medium"
    assert large_bucket == "large"


def test_alignment_group_has_three_buckets() -> None:
    from predictor.training.analysis import classify_alignment_group

    tc_metadata = KernelMetadata(
        name="tc",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 256, "n": 256, "k": 256},
        dtype="fp16",
        backend="cuda",
    )
    aligned_but_not_tc_metadata = KernelMetadata(
        name="aligned",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 256, "n": 256, "k": 256},
        dtype="fp32",
        backend="cuda",
    )
    non_aligned_metadata = KernelMetadata(
        name="unaligned",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 250, "n": 256, "k": 256},
        dtype="fp16",
        backend="cuda",
    )

    assert classify_alignment_group(tc_metadata) == "tensor_core_friendly"
    assert (
        classify_alignment_group(aligned_but_not_tc_metadata)
        == "aligned_but_not_tc_friendly"
    )
    assert classify_alignment_group(non_aligned_metadata) == "non_aligned"


def test_slice_summary_reports_mae_rmse_and_deltas() -> None:
    from predictor.training.analysis import summarize_slice_metrics

    rows = [
        {
            "gpu_name": "NVIDIA A40",
            "measured_latency_ms": 10.0,
            "baseline_latency_ms": 8.0,
            "predicted_latency_ms": 9.0,
        },
        {
            "gpu_name": "NVIDIA A40",
            "measured_latency_ms": 20.0,
            "baseline_latency_ms": 18.0,
            "predicted_latency_ms": 16.0,
        },
        {
            "gpu_name": "NVIDIA L40S",
            "measured_latency_ms": 30.0,
            "baseline_latency_ms": 33.0,
            "predicted_latency_ms": 31.0,
        },
    ]

    summary_rows = summarize_slice_metrics(
        rows=rows,
        slice_field="gpu_name",
        experiment_name="device_holdout",
    )

    a40_row = next(row for row in summary_rows if row["slice_value"] == "NVIDIA A40")

    assert a40_row["sample_count"] == 2
    assert "baseline_only_latency_mae" in a40_row
    assert "baseline_plus_residual_latency_mae" in a40_row
    assert "baseline_only_latency_rmse" in a40_row
    assert "baseline_plus_residual_latency_rmse" in a40_row
    assert "mae_delta" in a40_row
    assert "rmse_delta" in a40_row


def test_feature_coefficient_export_marks_standardized_space() -> None:
    from predictor.training.analysis import (
        STANDARDIZED_COEFFICIENT_NOTE,
        extract_top_feature_coefficients,
    )

    model = ResidualRidgeModel().fit(
        features=[
            FeatureVector(values={"flops": 1.0, "bytes": 1.0}),
            FeatureVector(values={"flops": 2.0, "bytes": 1.5}),
            FeatureVector(values={"flops": 4.0, "bytes": 2.0}),
        ],
        targets=[0.1, 0.2, 0.6],
    )

    coefficients = extract_top_feature_coefficients(model, top_k=2)

    assert len(coefficients) == 2
    assert coefficients[0]["coefficient_note"] == STANDARDIZED_COEFFICIENT_NOTE
