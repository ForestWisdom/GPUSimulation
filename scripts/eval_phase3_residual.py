"""Evaluate the Phase 3 GEMM/BMM residual model with configurable splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictor.training import ResidualEvaluator, ResidualTrainer, split_dataset
from predictor.training.io import load_gemm_bmm_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation-script CLI parser."""

    parser = argparse.ArgumentParser(description="Evaluate a GEMM/BMM residual model.")
    parser.add_argument("--data", required=True, help="Path to a JSONL or CSV dataset.")
    parser.add_argument(
        "--split-mode",
        choices=("random", "device-holdout"),
        default="random",
        help="Evaluation split strategy.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic random-split seed.")
    parser.add_argument(
        "--holdout-device",
        default=None,
        help="Optional device name to hold out when --split-mode=device-holdout.",
    )
    return parser


def main() -> None:
    """Train on the selected train split and evaluate on the test split."""

    args = build_parser().parse_args()
    dataset = load_gemm_bmm_dataset(args.data)
    train_dataset, test_dataset = split_dataset(
        dataset,
        mode=args.split_mode,
        random_seed=args.seed,
        holdout_device_name=args.holdout_device,
    )
    state = ResidualTrainer().fit(train_dataset)
    metrics = ResidualEvaluator().evaluate(
        test_dataset,
        state,
        train_size=len(train_dataset.samples),
    )
    print(f"split_mode={args.split_mode}")
    print(f"train_size={int(metrics['train_size'])}")
    print(f"test_size={int(metrics['test_size'])}")
    for key, value in metrics.items():
        if key in {"train_size", "test_size"}:
            continue
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
