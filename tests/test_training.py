from __future__ import annotations

import csv
import json

from predictor.training import (
    GemmBmmDatasetBuilder,
    ResidualEvaluator,
    ResidualTrainer,
    split_dataset,
)
from predictor.types import ResidualTrainingDataset


def test_dataset_builder_loads_jsonl_and_csv_records(tmp_path) -> None:
    jsonl_path = tmp_path / "runs.jsonl"
    csv_path = tmp_path / "runs.csv"
    records = [
        {
            "name": "gemm_sample_a",
            "family_hint": "gemm_bmm",
            "dtype": "fp16",
            "backend": "cuda",
            "dimensions": {"m": 1024, "n": 1024, "k": 2048},
            "device_profile": {
                "name": "test_gpu",
                "sm_count": 80,
                "memory_bandwidth_bytes_per_s": 1.2e12,
                "simt_flops_per_s": 16e12,
                "tensor_core_flops_per_s": 220e12,
            },
            "measured_latency_ms": 0.42,
        },
        {
            "name": "bmm_sample_b",
            "family_hint": "gemm_bmm",
            "dtype": "fp32",
            "backend": "cuda",
            "dimensions": {"batch": 4, "m": 512, "n": 512, "k": 1024},
            "device_profile": {
                "name": "test_gpu",
                "sm_count": 80,
                "memory_bandwidth_bytes_per_s": 1.2e12,
                "simt_flops_per_s": 16e12,
                "tensor_core_flops_per_s": 220e12,
            },
            "measured_latency_ms": 0.77,
        },
    ]

    jsonl_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "name",
                "family_hint",
                "dtype",
                "backend",
                "m",
                "n",
                "k",
                "batch",
                "device_name",
                "sm_count",
                "memory_bandwidth_bytes_per_s",
                "simt_flops_per_s",
                "tensor_core_flops_per_s",
                "measured_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "gemm_sample_c",
                "family_hint": "gemm_bmm",
                "dtype": "fp16",
                "backend": "cuda",
                "m": 2048,
                "n": 1024,
                "k": 4096,
                "batch": 1,
                "device_name": "csv_gpu",
                "sm_count": 96,
                "memory_bandwidth_bytes_per_s": 1.5e12,
                "simt_flops_per_s": 18e12,
                "tensor_core_flops_per_s": 260e12,
                "measured_latency_ms": 0.55,
            }
        )

    builder = GemmBmmDatasetBuilder()
    json_dataset = builder.from_jsonl(jsonl_path)
    csv_dataset = builder.from_csv(csv_path)

    assert isinstance(json_dataset, ResidualTrainingDataset)
    assert len(json_dataset.samples) == 2
    assert json_dataset.samples[0].residual_target_ms == round(
        json_dataset.samples[0].measured_latency_ms
        - json_dataset.samples[0].analytical_baseline_ms,
        6,
    )
    assert csv_dataset.samples[0].device_profile.name == "csv_gpu"
    assert csv_dataset.samples[0].metadata.dimensions["k"] == 4096


def test_trainer_and_evaluator_fit_and_score_residual_dataset(tmp_path) -> None:
    builder = GemmBmmDatasetBuilder()
    trainer = ResidualTrainer()
    evaluator = ResidualEvaluator()
    records = [
        {
            "name": f"gemm_train_{index}",
            "family_hint": "gemm_bmm",
            "dtype": "fp16" if index % 2 == 0 else "fp32",
            "backend": "cuda",
            "dimensions": {
                "batch": 1 + (index % 3),
                "m": 512 + index * 64,
                "n": 256 + index * 32,
                "k": 1024 + index * 64,
            },
            "device_profile": {
                "name": "train_gpu",
                "sm_count": 80,
                "memory_bandwidth_bytes_per_s": 1.2e12,
                "simt_flops_per_s": 16e12,
                "tensor_core_flops_per_s": 220e12,
            },
            "measured_latency_ms": 0.20 + index * 0.03,
        }
        for index in range(10)
    ]
    dataset = builder.build_from_records(records)
    train_dataset, test_dataset = split_dataset(dataset)
    state = trainer.fit(train_dataset)
    metrics = evaluator.evaluate(test_dataset, state, train_size=len(train_dataset.samples))
    save_path = tmp_path / "ridge_pipeline.joblib"

    state.model.save(save_path)
    restored_model = state.model.load(save_path)

    assert len(train_dataset.samples) == 8
    assert len(test_dataset.samples) == 2
    assert state.sample_count == 8
    assert metrics["train_size"] == 8.0
    assert metrics["test_size"] == 2.0
    assert "baseline_plus_residual_latency_mae" in metrics
    assert restored_model.predict(test_dataset.samples[0].features) == state.model.predict(
        test_dataset.samples[0].features
    )
