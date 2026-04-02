"""Phase 2 runnable demo for GEMM/BMM analytical baseline prediction."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictor.serving import KernelLatencyPredictor, OperatorLatencyAggregator


def main() -> None:
    """Run a GEMM/BMM analytical baseline demo."""

    predictor = KernelLatencyPredictor.default()
    kernels = [
        {
            "name": "demo_gemm_simt",
            "family_hint": "gemm_bmm",
            "dtype": "fp32",
            "backend": "cuda",
            "dimensions": {"m": 1024, "n": 768, "k": 512},
            "extra": {"sm_count": 108},
        },
        {
            "name": "demo_bmm_tensor_core",
            "family_hint": "gemm_bmm",
            "dtype": "fp16",
            "backend": "cuda",
            "dimensions": {"batch": 8, "m": 2048, "n": 1024, "k": 4096},
            "extra": {"sm_count": 108},
        },
    ]

    predictions = [predictor.predict_from_raw_metadata(kernel) for kernel in kernels]
    summary = OperatorLatencyAggregator().aggregate(predictions)

    for prediction in predictions:
        print(prediction)
    print(summary)


if __name__ == "__main__":
    main()
