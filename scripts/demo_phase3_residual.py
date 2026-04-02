"""Phase 3 demo for training and using a GEMM/BMM residual model."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictor.serving import KernelLatencyPredictor
from predictor.training import ResidualEvaluator, ResidualTrainer, split_dataset
from predictor.training.io import load_gemm_bmm_dataset


def main() -> None:
    """Train a tiny residual model and run one prediction through serving."""

    records = [
        _make_record("demo_gemm_a", 512, 512, 1024, 0.11, "fp16"),
        _make_record("demo_gemm_b", 768, 512, 1024, 0.16, "fp16"),
        _make_record("demo_gemm_c", 1024, 1024, 2048, 0.29, "fp16"),
        _make_record("demo_gemm_d", 1536, 1024, 2048, 0.41, "fp32"),
        _make_record("demo_gemm_e", 2048, 1024, 4096, 0.58, "fp16"),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = Path(temp_dir) / "phase3_demo.jsonl"
        dataset_path.write_text(
            "\n".join(json.dumps(record) for record in records),
            encoding="utf-8",
        )
        dataset = load_gemm_bmm_dataset(dataset_path)
        train_dataset, test_dataset = split_dataset(dataset)
        trainer_state = ResidualTrainer().fit(train_dataset)
        metrics = ResidualEvaluator().evaluate(
            test_dataset,
            trainer_state,
            train_size=len(train_dataset.samples),
        )

        predictor = KernelLatencyPredictor.default()
        predictor.residual_model = trainer_state.model
        prediction = predictor.predict_from_raw_metadata(
            {
                "name": "demo_phase3_inference",
                "family_hint": "gemm_bmm",
                "dtype": "fp16",
                "dimensions": {"m": 1280, "n": 1024, "k": 2048},
                "backend": "cuda",
            }
        )

    print(f"train_size={int(metrics['train_size'])}")
    print(f"test_size={int(metrics['test_size'])}")
    print(
        "baseline_plus_residual_latency_mae="
        f"{metrics['baseline_plus_residual_latency_mae']:.6f}"
    )
    print(prediction)


def _make_record(
    name: str,
    m: int,
    n: int,
    k: int,
    measured_latency_ms: float,
    dtype: str,
) -> dict[str, object]:
    """Create a small deterministic GEMM training record."""

    return {
        "name": name,
        "family_hint": "gemm_bmm",
        "dtype": dtype,
        "backend": "cuda",
        "dimensions": {"m": m, "n": n, "k": k},
        "device_profile": {
            "name": "demo_gpu",
            "sm_count": 108,
            "memory_bandwidth_bytes_per_s": 1.555e12,
            "simt_flops_per_s": 19.5e12,
            "tensor_core_flops_per_s": 312e12,
        },
        "measured_latency_ms": measured_latency_ms,
    }


if __name__ == "__main__":
    main()
