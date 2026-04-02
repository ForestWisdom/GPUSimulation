from __future__ import annotations

import json
from pathlib import Path

from predictor.training import GemmBmmDatasetBuilder
from predictor.training.profiling import (
    build_gemm_bmm_sampling_plan,
    collect_gemm_bmm_profile_records,
    write_profile_records_jsonl,
)


def test_sampling_plan_covers_required_shape_families() -> None:
    plan = build_gemm_bmm_sampling_plan(
        families=("gemm", "bmm"),
        dtypes=("fp16", "fp32"),
        size_buckets=("small",),
    )

    assert {"square-ish", "tall-skinny", "wide", "k-heavy"} == {
        spec.shape_pattern for spec in plan
    }
    assert {True, False} == {spec.aligned for spec in plan}
    assert {1, 4, 16, 64}.issubset(
        {spec.batch for spec in plan if spec.family == "bmm"}
    )


def test_mock_profile_records_include_measurement_metadata_and_are_dataset_compatible(
    tmp_path: Path,
) -> None:
    plan = build_gemm_bmm_sampling_plan(
        families=("gemm",),
        dtypes=("fp16",),
        size_buckets=("small",),
    )[:2]
    records = collect_gemm_bmm_profile_records(
        plan=plan,
        mode="mock",
        num_warmup=5,
        num_repeats=7,
        seed=11,
        gpu_names=("mock_gpu_a", "mock_gpu_b"),
    )
    output_path = tmp_path / "profile_records.jsonl"

    write_profile_records_jsonl(records, output_path)
    dataset = GemmBmmDatasetBuilder().from_jsonl(output_path)
    first_record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert "measured_latency_ms" in first_record
    assert "latency_std_ms" in first_record
    assert first_record["num_warmup"] == 5
    assert first_record["num_repeats"] == 7
    assert first_record["measurement_backend"] == "mock"
    assert first_record["gpu_name"] in {"mock_gpu_a", "mock_gpu_b"}
    assert "torch_version" in first_record
    assert "cuda_version" in first_record
    assert first_record["seed"] == 11
    assert len(dataset.samples) == len(records)
