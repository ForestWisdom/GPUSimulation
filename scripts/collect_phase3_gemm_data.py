"""Collect GEMM/BMM profiling data for the Phase 3.1 workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictor.training import (
    build_gemm_bmm_sampling_plan,
    collect_gemm_bmm_profile_records,
    write_profile_records_csv,
    write_profile_records_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 3 data-collection CLI parser."""

    parser = argparse.ArgumentParser(description="Collect GEMM/BMM profiling records.")
    parser.add_argument("--output", required=True, help="Path to write JSONL/CSV records.")
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--mode", choices=("mock", "torch"), default="mock")
    parser.add_argument("--dtypes", default="fp16,bf16,fp32")
    parser.add_argument("--families", default="gemm,bmm")
    parser.add_argument("--sizes", default="small,medium,large")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-names", default="mock_gpu_a,mock_gpu_b")
    parser.add_argument("--rounds", type=int, default=1)
    return parser


def main() -> None:
    """Collect profiling records and persist them in the requested format."""

    args = build_parser().parse_args()
    plan = build_gemm_bmm_sampling_plan(
        families=_split_csv_arg(args.families),
        dtypes=_split_csv_arg(args.dtypes),
        size_buckets=_split_csv_arg(args.sizes),
    )
    output_path = Path(args.output)
    total_records = 0
    for round_id in range(args.rounds):
        records = collect_gemm_bmm_profile_records(
            plan=plan,
            mode=args.mode,
            num_warmup=args.warmup,
            num_repeats=args.repeats,
            seed=args.seed,
            gpu_names=_split_csv_arg(args.gpu_names),
        )
        if args.rounds > 1:
            records = [dict(record, round_id=round_id) for record in records]
        append = round_id > 0
        if args.format == "jsonl":
            write_profile_records_jsonl(records, output_path, append=append)
        else:
            write_profile_records_csv(records, output_path, append=append)
        total_records += len(records)

    print(f"records_collected={total_records}")
    print(f"output_path={output_path}")
    print(f"measurement_backend={args.mode}")


def _split_csv_arg(value: str) -> tuple[str, ...]:
    """Split a comma-separated CLI argument into a tuple."""

    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
