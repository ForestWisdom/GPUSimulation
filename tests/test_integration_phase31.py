from __future__ import annotations

import subprocess
from pathlib import Path


def test_phase3_collection_entrypoint_runs_in_mock_mode(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "collected.jsonl"
    result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "sglang",
            "python",
            str(repo_root / "scripts" / "collect_phase3_gemm_data.py"),
            "--output",
            str(output_path),
            "--format",
            "jsonl",
            "--mode",
            "mock",
            "--families",
            "gemm,bmm",
            "--dtypes",
            "fp16",
            "--sizes",
            "small",
            "--warmup",
            "3",
            "--repeats",
            "5",
            "--seed",
            "17",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    assert "records_collected=" in result.stdout


def test_phase3_validation_workflow_runs_end_to_end_in_mock_mode(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "collected.jsonl"
    collect_result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "sglang",
            "python",
            str(repo_root / "scripts" / "collect_phase3_gemm_data.py"),
            "--output",
            str(output_path),
            "--format",
            "jsonl",
            "--mode",
            "mock",
            "--families",
            "gemm,bmm",
            "--dtypes",
            "fp16",
            "--sizes",
            "small",
            "--warmup",
            "3",
            "--repeats",
            "5",
            "--seed",
            "17",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert collect_result.returncode == 0, collect_result.stderr

    result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "sglang",
            "python",
            str(repo_root / "scripts" / "run_phase3_validation.py"),
            "--data",
            str(output_path),
            "--split-mode",
            "device-holdout",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "split_mode=device-holdout" in result.stdout
    assert "train_size=" in result.stdout
    assert "test_size=" in result.stdout
    assert "residual_mae=" in result.stdout
    assert "residual_rmse=" in result.stdout
    assert "baseline_only_latency_mae=" in result.stdout
    assert "baseline_plus_residual_latency_mae=" in result.stdout
    assert "baseline_only_latency_mape=" in result.stdout
    assert "baseline_plus_residual_latency_mape=" in result.stdout
