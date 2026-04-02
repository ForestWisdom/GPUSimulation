"""Training package exports."""

from predictor.training.analysis import (
    SLICE_FIELDS,
    SIZE_BUCKET_RULE,
    STANDARDIZED_COEFFICIENT_NOTE,
    build_prediction_rows,
    classify_alignment_group,
    derive_size_bucket,
    extract_top_feature_coefficients,
    load_analysis_records,
    summarize_experiment,
    summarize_residual_diagnostics,
    summarize_slice_metrics,
    write_csv_rows,
    write_json_payload,
)
from predictor.training.dataset import GemmBmmDatasetBuilder, KernelDatasetBuilder
from predictor.training.evaluator import PlaceholderEvaluator, ResidualEvaluator
from predictor.training.profiling import (
    GemmBmmShapeSpec,
    build_gemm_bmm_sampling_plan,
    collect_gemm_bmm_profile_records,
    write_profile_records_csv,
    write_profile_records_jsonl,
)
from predictor.training.trainer import PlaceholderTrainer, ResidualTrainer, split_dataset

__all__ = [
    "SLICE_FIELDS",
    "SIZE_BUCKET_RULE",
    "STANDARDIZED_COEFFICIENT_NOTE",
    "build_prediction_rows",
    "classify_alignment_group",
    "derive_size_bucket",
    "extract_top_feature_coefficients",
    "GemmBmmDatasetBuilder",
    "GemmBmmShapeSpec",
    "KernelDatasetBuilder",
    "load_analysis_records",
    "PlaceholderEvaluator",
    "PlaceholderTrainer",
    "ResidualEvaluator",
    "ResidualTrainer",
    "build_gemm_bmm_sampling_plan",
    "collect_gemm_bmm_profile_records",
    "split_dataset",
    "summarize_experiment",
    "summarize_residual_diagnostics",
    "summarize_slice_metrics",
    "write_csv_rows",
    "write_json_payload",
    "write_profile_records_csv",
    "write_profile_records_jsonl",
]
