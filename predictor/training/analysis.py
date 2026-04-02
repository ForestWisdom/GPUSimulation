"""Error-analysis helpers for Phase 3.2 GEMM/BMM evaluation."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from predictor.types import KernelMetadata, PathLike, uses_tensor_cores

SIZE_BUCKET_RULE = "size_bucket derived from max(m, n, k): <=2048 small; <=8192 medium; >8192 large"
STANDARDIZED_COEFFICIENT_NOTE = (
    "Coefficients are from the standardized feature space because the model is StandardScaler + Ridge."
)
SLICE_FIELDS = ("gpu_name", "family", "dtype", "size_bucket", "alignment_group")


def load_analysis_records(path: PathLike) -> list[dict[str, Any]]:
    """Load raw profiling records for analysis from JSONL or CSV."""

    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if input_path.suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as file_obj:
            return [dict(row) for row in csv.DictReader(file_obj)]
    raise ValueError(f"Unsupported analysis dataset format: {input_path.suffix}")


def derive_size_bucket(dimensions: dict[str, int | float]) -> tuple[str, str]:
    """Derive the GEMM/BMM size bucket from kernel dimensions."""

    max_dimension = max(
        int(dimensions.get("m", 0)),
        int(dimensions.get("n", 0)),
        int(dimensions.get("k", 0)),
    )
    if max_dimension <= 2048:
        return "small", SIZE_BUCKET_RULE
    if max_dimension <= 8192:
        return "medium", SIZE_BUCKET_RULE
    return "large", SIZE_BUCKET_RULE


def classify_alignment_group(metadata: KernelMetadata) -> str:
    """Classify one GEMM/BMM sample into an alignment group."""

    m = int(metadata.dimensions.get("m", 0))
    n = int(metadata.dimensions.get("n", 0))
    k = int(metadata.dimensions.get("k", 0))
    all_aligned = all(dimension > 0 and dimension % 16 == 0 for dimension in (m, n, k))
    if uses_tensor_cores(metadata):
        return "tensor_core_friendly"
    if all_aligned:
        return "aligned_but_not_tc_friendly"
    return "non_aligned"


def build_prediction_rows(
    records: list[dict[str, Any]],
    dataset: Any,
    model: Any,
) -> list[dict[str, Any]]:
    """Build analysis rows with baseline and residual predictions."""

    predicted_residuals = model.predict_batch(
        [sample.features for sample in dataset.samples]
    )
    rows: list[dict[str, Any]] = []
    for record, sample, predicted_residual_ms in zip(
        records,
        dataset.samples,
        predicted_residuals,
    ):
        family = _extract_family(record, sample.metadata)
        size_bucket, size_bucket_rule = derive_size_bucket(sample.metadata.dimensions)
        rows.append(
            {
                "gpu_name": sample.device_profile.name,
                "family": family,
                "dtype": sample.metadata.dtype,
                "size_bucket": size_bucket,
                "size_bucket_rule": size_bucket_rule,
                "alignment_group": classify_alignment_group(sample.metadata),
                "measured_latency_ms": sample.measured_latency_ms,
                "baseline_latency_ms": sample.analytical_baseline_ms,
                "predicted_latency_ms": max(
                    1e-6,
                    sample.analytical_baseline_ms + float(predicted_residual_ms),
                ),
                "residual_target_ms": sample.residual_target_ms,
                "predicted_residual_ms": float(predicted_residual_ms),
            }
        )
    return rows


def summarize_experiment(
    rows: list[dict[str, Any]],
    experiment_name: str,
    train_size: int,
    holdout_device: str | None = None,
) -> dict[str, Any]:
    """Summarize one random or holdout experiment."""

    baseline_mae = _mae_from_rows(rows, "baseline_latency_ms")
    predicted_mae = _mae_from_rows(rows, "predicted_latency_ms")
    baseline_rmse = _rmse_from_rows(rows, "baseline_latency_ms")
    predicted_rmse = _rmse_from_rows(rows, "predicted_latency_ms")
    return {
        "experiment_name": experiment_name,
        "holdout_device": holdout_device or "",
        "train_size": train_size,
        "test_size": len(rows),
        "baseline_only_latency_mae": baseline_mae,
        "baseline_plus_residual_latency_mae": predicted_mae,
        "baseline_only_latency_rmse": baseline_rmse,
        "baseline_plus_residual_latency_rmse": predicted_rmse,
        "mae_delta": predicted_mae - baseline_mae,
        "rmse_delta": predicted_rmse - baseline_rmse,
    }


def summarize_slice_metrics(
    rows: list[dict[str, Any]],
    slice_field: str,
    experiment_name: str,
) -> list[dict[str, Any]]:
    """Summarize baseline and residual errors for one slice key."""

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row[slice_field])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for slice_value, slice_rows in sorted(grouped_rows.items()):
        baseline_mae = _mae_from_rows(slice_rows, "baseline_latency_ms")
        predicted_mae = _mae_from_rows(slice_rows, "predicted_latency_ms")
        baseline_rmse = _rmse_from_rows(slice_rows, "baseline_latency_ms")
        predicted_rmse = _rmse_from_rows(slice_rows, "predicted_latency_ms")
        summary_rows.append(
            {
                "experiment_name": experiment_name,
                "slice_field": slice_field,
                "slice_value": slice_value,
                "sample_count": len(slice_rows),
                "size_bucket_rule": SIZE_BUCKET_RULE,
                "baseline_only_latency_mae": baseline_mae,
                "baseline_plus_residual_latency_mae": predicted_mae,
                "baseline_only_latency_rmse": baseline_rmse,
                "baseline_plus_residual_latency_rmse": predicted_rmse,
                "mae_delta": predicted_mae - baseline_mae,
                "rmse_delta": predicted_rmse - baseline_rmse,
            }
        )
    return summary_rows


def summarize_residual_diagnostics(
    rows: list[dict[str, Any]],
    experiment_name: str,
) -> list[dict[str, Any]]:
    """Summarize residual-target and predicted-residual distributions by device."""

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["gpu_name"])].append(row)

    diagnostics: list[dict[str, Any]] = []
    for device_name, device_rows in sorted(grouped_rows.items()):
        residual_targets = [float(row["residual_target_ms"]) for row in device_rows]
        predicted_residuals = [float(row["predicted_residual_ms"]) for row in device_rows]
        diagnostics.append(
            {
                "experiment_name": experiment_name,
                "gpu_name": device_name,
                "sample_count": len(device_rows),
                "residual_target_mean": statistics.mean(residual_targets),
                "residual_target_std": _safe_pstdev(residual_targets),
                "predicted_residual_mean": statistics.mean(predicted_residuals),
                "predicted_residual_std": _safe_pstdev(predicted_residuals),
            }
        )
    return diagnostics


def extract_top_feature_coefficients(model: Any, top_k: int = 10) -> list[dict[str, Any]]:
    """Extract the largest-magnitude Ridge coefficients from one fitted model."""

    ridge = model.pipeline.named_steps["ridge"]
    coefficient_rows = [
        {
            "feature_name": feature_name,
            "coefficient": float(coefficient),
            "abs_coefficient": abs(float(coefficient)),
            "coefficient_note": STANDARDIZED_COEFFICIENT_NOTE,
        }
        for feature_name, coefficient in zip(model.feature_names, ridge.coef_)
    ]
    return sorted(
        coefficient_rows,
        key=lambda row: row["abs_coefficient"],
        reverse=True,
    )[:top_k]


def write_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write analysis rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json_payload(payload: dict[str, Any], path: Path) -> None:
    """Write one JSON payload to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_family(record: dict[str, Any], metadata: KernelMetadata) -> str:
    """Recover the GEMM/BMM family label from raw records or metadata."""

    run_tags = dict(record.get("run_tags", {}))
    if run_tags.get("family"):
        return str(run_tags["family"])
    if record.get("tag_family"):
        return str(record["tag_family"])
    if int(metadata.dimensions.get("batch", 1)) > 1:
        return "bmm"
    if metadata.name.startswith("bmm"):
        return "bmm"
    return "gemm"


def _mae_from_rows(rows: list[dict[str, Any]], prediction_key: str) -> float:
    """Compute MAE for one prediction column."""

    if not rows:
        return 0.0
    return sum(
        abs(float(row["measured_latency_ms"]) - float(row[prediction_key]))
        for row in rows
    ) / len(rows)


def _rmse_from_rows(rows: list[dict[str, Any]], prediction_key: str) -> float:
    """Compute RMSE for one prediction column."""

    if not rows:
        return 0.0
    mse = sum(
        (float(row["measured_latency_ms"]) - float(row[prediction_key])) ** 2
        for row in rows
    ) / len(rows)
    return math.sqrt(mse)


def _safe_pstdev(values: list[float]) -> float:
    """Return population standard deviation or zero for singleton lists."""

    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)
