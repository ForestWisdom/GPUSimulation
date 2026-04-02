"""Train the Phase 3 GEMM/BMM residual model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictor.training import ResidualTrainer
from predictor.training.io import load_gemm_bmm_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build the training-script CLI parser."""

    parser = argparse.ArgumentParser(description="Train a GEMM/BMM residual model.")
    parser.add_argument("--data", required=True, help="Path to a JSONL or CSV dataset.")
    parser.add_argument("--output", required=True, help="Path to write the model artifact.")
    return parser


def main() -> None:
    """Train and save a residual model."""

    args = build_parser().parse_args()
    dataset = load_gemm_bmm_dataset(args.data)
    state = ResidualTrainer().fit(dataset)
    state.model.save(args.output)
    print(f"trained_samples={state.sample_count}")
    print(f"saved_model={args.output}")


if __name__ == "__main__":
    main()
