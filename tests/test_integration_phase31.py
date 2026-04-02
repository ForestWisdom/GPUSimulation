from __future__ import annotations

import json
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


def test_phase3_collection_entrypoint_supports_multi_round_append(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "collected_rounds.jsonl"
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
            "gemm",
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
            "--rounds",
            "2",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = output_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 16
    assert {0, 1} == {record["round_id"] for record in records}
    assert "records_collected=16" in result.stdout


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


def test_phase3_train_and_eval_entrypoints_run_without_pythonpath(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = tmp_path / "collected.jsonl"
    model_path = tmp_path / "residual.joblib"
    collect_result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "sglang",
            "python",
            str(repo_root / "scripts" / "collect_phase3_gemm_data.py"),
            "--output",
            str(dataset_path),
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

    train_result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "sglang",
            "python",
            str(repo_root / "scripts" / "train_phase3_residual.py"),
            "--data",
            str(dataset_path),
            "--output",
            str(model_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert train_result.returncode == 0, train_result.stderr
    assert model_path.exists()
    assert "trained_samples=40" in train_result.stdout

    eval_result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "sglang",
            "python",
            str(repo_root / "scripts" / "eval_phase3_residual.py"),
            "--data",
            str(dataset_path),
            "--split-mode",
            "device-holdout",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert eval_result.returncode == 0, eval_result.stderr
    assert "split_mode=device-holdout" in eval_result.stdout
    assert "train_size=20" in eval_result.stdout
    assert "test_size=20" in eval_result.stdout
