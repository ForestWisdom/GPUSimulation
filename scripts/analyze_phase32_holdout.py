"""Run Phase 3.2 error analysis on the merged real GEMM/BMM dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictor.training import GemmBmmDatasetBuilder, ResidualTrainer, split_dataset
from predictor.training.analysis import (
    SLICE_FIELDS,
    build_prediction_rows,
    extract_top_feature_coefficients,
    load_analysis_records,
    summarize_experiment,
    summarize_residual_diagnostics,
    summarize_slice_metrics,
    write_csv_rows,
    write_json_payload,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 3.2 analysis CLI parser."""

    parser = argparse.ArgumentParser(
        description="Analyze Phase 3.1 holdout errors on the merged real GEMM/BMM dataset.",
    )
    parser.add_argument(
        "--data",
        default="artifacts/phase3_real_multi_gpu.jsonl",
        help="Path to the merged real dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to write Phase 3.2 CSV/JSON outputs.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic random split seed.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many Ridge coefficients to export.",
    )
    return parser


def main() -> None:
    """Run Phase 3.2 error analysis and print main findings."""

    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    records = load_analysis_records(args.data)
    dataset = GemmBmmDatasetBuilder().build_from_records(records)
    record_by_sample_id = {
        id(sample): record
        for sample, record in zip(dataset.samples, records)
    }

    full_state = ResidualTrainer().fit(dataset)
    coefficient_rows = extract_top_feature_coefficients(full_state.model, top_k=args.top_k)
    write_csv_rows(coefficient_rows, output_dir / "phase32_top_coefficients.csv")

    experiment_summaries = []
    holdout_summaries = []
    residual_diagnostics = []
    findings_payload: dict[str, list[dict[str, object]]] = {"slices": []}

    random_rows, random_summary = _run_one_experiment(
        dataset=dataset,
        record_by_sample_id=record_by_sample_id,
        split_mode="random",
        seed=args.seed,
    )
    experiment_summaries.append(random_summary)
    _write_slice_files(
        output_dir=output_dir,
        experiment_slug="random",
        experiment_name="random",
        rows=random_rows,
        findings_payload=findings_payload,
    )

    device_names = sorted({sample.device_profile.name for sample in dataset.samples})
    holdout_slice_rows: list[dict[str, object]] = []
    for device_name in device_names:
        holdout_rows, holdout_summary = _run_one_experiment(
            dataset=dataset,
            record_by_sample_id=record_by_sample_id,
            split_mode="device-holdout",
            holdout_device=device_name,
            seed=args.seed,
        )
        experiment_summaries.append(holdout_summary)
        holdout_summaries.append(holdout_summary)
        residual_diagnostics.extend(
            summarize_residual_diagnostics(
                holdout_rows,
                experiment_name=f"device_holdout:{device_name}",
            )
        )
        holdout_slice_rows.extend(
            _write_slice_files(
                output_dir=output_dir,
                experiment_slug=f"holdout_{_slugify(device_name)}",
                experiment_name=f"device_holdout:{device_name}",
                rows=holdout_rows,
                findings_payload=findings_payload,
            )
        )

    write_csv_rows(experiment_summaries, output_dir / "phase32_experiment_summary.csv")
    write_csv_rows(holdout_summaries, output_dir / "phase32_holdout_summary.csv")
    write_csv_rows(residual_diagnostics, output_dir / "phase32_residual_diagnostics.csv")

    findings = _build_findings(
        holdout_summaries=holdout_summaries,
        residual_diagnostics=residual_diagnostics,
        coefficient_rows=coefficient_rows,
        holdout_slice_rows=holdout_slice_rows,
    )
    write_json_payload(
        {
            "experiment_summaries": experiment_summaries,
            "holdout_summaries": holdout_summaries,
            "residual_diagnostics": residual_diagnostics,
            "top_coefficients": coefficient_rows,
            "findings": findings,
        },
        output_dir / "phase32_analysis_summary.json",
    )

    print("Main findings")
    for finding in findings:
        print(f"- {finding}")
    print(f"output_dir={output_dir}")


def _run_one_experiment(
    dataset,
    record_by_sample_id: dict[int, dict[str, object]],
    split_mode: str,
    seed: int,
    holdout_device: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run one random or device-holdout experiment."""

    train_dataset, test_dataset = split_dataset(
        dataset,
        mode=split_mode,
        random_seed=seed,
        holdout_device_name=holdout_device,
    )
    state = ResidualTrainer().fit(train_dataset)
    test_records = [record_by_sample_id[id(sample)] for sample in test_dataset.samples]
    test_rows = build_prediction_rows(test_records, test_dataset, state.model)
    summary = summarize_experiment(
        rows=test_rows,
        experiment_name=split_mode if holdout_device is None else f"device_holdout:{holdout_device}",
        train_size=len(train_dataset.samples),
        holdout_device=holdout_device,
    )
    return test_rows, summary


def _write_slice_files(
    output_dir: Path,
    experiment_slug: str,
    experiment_name: str,
    rows: list[dict[str, object]],
    findings_payload: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Write per-slice CSVs for one experiment."""

    all_slice_rows: list[dict[str, object]] = []
    for slice_field in SLICE_FIELDS:
        slice_rows = summarize_slice_metrics(
            rows=rows,
            slice_field=slice_field,
            experiment_name=experiment_name,
        )
        all_slice_rows.extend(slice_rows)
        findings_payload["slices"].extend(slice_rows)
        write_csv_rows(
            slice_rows,
            output_dir / f"phase32_{experiment_slug}_slice_{slice_field}.csv",
        )
    return all_slice_rows


def _build_findings(
    holdout_summaries: list[dict[str, object]],
    residual_diagnostics: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
    holdout_slice_rows: list[dict[str, object]],
) -> list[str]:
    """Build a short human-readable findings summary."""

    a40_summary = next(
        row for row in holdout_summaries if row["holdout_device"] == "NVIDIA A40"
    )
    worst_holdout = max(holdout_summaries, key=lambda row: row["mae_delta"])
    a40_diagnostics = next(
        row for row in residual_diagnostics if row["gpu_name"] == "NVIDIA A40"
    )
    a40_worst_slice = max(
        (
            row
            for row in holdout_slice_rows
            if row["experiment_name"] == "device_holdout:NVIDIA A40"
            and row["slice_field"] != "gpu_name"
        ),
        key=lambda row: row["mae_delta"],
    )
    top_features = ", ".join(
        row["feature_name"] for row in coefficient_rows[:3]
    )
    return [
        (
            f"Worst device holdout by MAE delta is {worst_holdout['holdout_device']} "
            f"with mae_delta={worst_holdout['mae_delta']:.4f} ms."
        ),
        (
            f"A40 holdout baseline_plus_residual MAE is "
            f"{a40_summary['baseline_plus_residual_latency_mae']:.4f} ms versus "
            f"{a40_summary['baseline_only_latency_mae']:.4f} ms baseline-only."
        ),
        (
            f"A40 residual target mean/std is "
            f"{a40_diagnostics['residual_target_mean']:.4f}/{a40_diagnostics['residual_target_std']:.4f} ms, "
            f"while predicted residual mean/std is "
            f"{a40_diagnostics['predicted_residual_mean']:.4f}/"
            f"{a40_diagnostics['predicted_residual_std']:.4f} ms."
        ),
        (
            f"Largest positive A40 slice delta is {a40_worst_slice['slice_field']}="
            f"{a40_worst_slice['slice_value']} with mae_delta={a40_worst_slice['mae_delta']:.4f} ms."
        ),
        f"Top standardized-space Ridge features by absolute coefficient are: {top_features}.",
    ]


def _slugify(value: str) -> str:
    """Convert one device name into a filesystem-safe slug."""

    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


if __name__ == "__main__":
    main()
