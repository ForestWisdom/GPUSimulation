"""Phase 1 runnable demo for the GPU operator predictor scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictor.serving import KernelLatencyPredictor, OperatorLatencyAggregator


def main() -> None:
    """Run a small demo with two placeholder kernel predictions."""

    predictor = KernelLatencyPredictor.default()
    kernels = [
        {
            "name": "demo_gemm_kernel",
            "family_hint": "gemm_bmm",
            "dtype": "fp16",
            "backend": "cuda",
            "dimensions": {"m": 128, "n": 128, "k": 64},
        },
        {
            "name": "demo_attention_kernel",
            "family_hint": "attention",
            "dtype": "bf16",
            "backend": "cuda",
            "dimensions": {"batch": 2, "seq_len": 128, "heads": 16},
        },
    ]
    predictions = [predictor.predict_from_raw_metadata(kernel) for kernel in kernels]
    summary = OperatorLatencyAggregator().aggregate(predictions)

    for prediction in predictions:
        print(prediction)
    print(summary)


if __name__ == "__main__":
    main()
