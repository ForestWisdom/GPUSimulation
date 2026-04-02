"""Run the Phase 3.1 GEMM/BMM validation workflow."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictor.training import (
    ResidualEvaluator,
    ResidualTrainer,
    build_gemm_bmm_sampling_plan,
    collect_gemm_bmm_profile_records,
    split_dataset,
    write_profile_records_csv,
    write_profile_records_jsonl,
)
from predictor.training.io import load_gemm_bmm_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 3 validation CLI parser."""

    parser = argparse.ArgumentParser(
        description="Collect/load GEMM/BMM data, train the Ridge residual model, and evaluate it.",
    )
    parser.add_argument("--data", default=None, help="Existing JSONL/CSV dataset path.")
    parser.add_argument("--mode", choices=("mock", "torch"), default=None)
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--families", default="gemm,bmm")
    parser.add_argument("--dtypes", default="fp16,bf16,fp32")
    parser.add_argument("--sizes", default="small,medium,large")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-names", default="mock_gpu_a,mock_gpu_b")
    parser.add_argument(
        "--split-mode",
        choices=("random", "device-holdout"),
        default="random",
    )
    return parser


def main() -> None:
    """Execute the Phase 3 validation workflow."""

    args = build_parser().parse_args()
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = _resolve_dataset_path(args, Path(temp_dir))
        dataset = load_gemm_bmm_dataset(dataset_path)
        train_dataset, test_dataset = split_dataset(
            dataset,
            mode=args.split_mode,
            random_seed=args.seed,
        )
        trainer_state = ResidualTrainer().fit(train_dataset)
        metrics = ResidualEvaluator().evaluate(
            test_dataset,
            trainer_state,
            train_size=len(train_dataset.samples),
        )

    print(f"split_mode={args.split_mode}")
    print(f"train_size={int(metrics['train_size'])}")
    print(f"test_size={int(metrics['test_size'])}")
    print(f"residual_mae={metrics['residual_mae']:.6f}")
    print(f"residual_rmse={metrics['residual_rmse']:.6f}")
    print(f"baseline_only_latency_mae={metrics['baseline_only_latency_mae']:.6f}")
    print(
        "baseline_plus_residual_latency_mae="
        f"{metrics['baseline_plus_residual_latency_mae']:.6f}"
    )
    print(
        f"baseline_only_latency_mape={metrics['baseline_only_latency_mape']:.6f}"
    )
    print(
        "baseline_plus_residual_latency_mape="
        f"{metrics['baseline_plus_residual_latency_mape']:.6f}"
    )


def _resolve_dataset_path(args: argparse.Namespace, temp_dir: Path) -> Path:
    """Resolve the dataset path by loading existing data or collecting it."""

    if args.data:
        return Path(args.data)
    if not args.mode:
        raise ValueError("Either --data or --mode must be provided.")

    plan = build_gemm_bmm_sampling_plan(
        families=_split_csv_arg(args.families),
        dtypes=_split_csv_arg(args.dtypes),
        size_buckets=_split_csv_arg(args.sizes),
    )
    records = collect_gemm_bmm_profile_records(
        plan=plan,
        mode=args.mode,
        num_warmup=args.warmup,
        num_repeats=args.repeats,
        seed=args.seed,
        gpu_names=_split_csv_arg(args.gpu_names),
    )
    dataset_path = temp_dir / f"phase3_validation.{args.format}"
    if args.format == "jsonl":
        write_profile_records_jsonl(records, dataset_path)
    else:
        write_profile_records_csv(records, dataset_path)
    return dataset_path


def _split_csv_arg(value: str) -> tuple[str, ...]:
    """Split a comma-separated CLI argument into a tuple."""

    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
