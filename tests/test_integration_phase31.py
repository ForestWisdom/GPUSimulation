from __future__ import annotations

import subprocess
from pathlib import Path


def test_phase31_validation_workflow_runs_end_to_end_in_mock_mode() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "sglang",
            "python",
            str(repo_root / "scripts" / "run_phase31_validation.py"),
            "--collect-mode",
            "mock",
            "--split-mode",
            "device-holdout",
            "--families",
            "gemm,bmm",
            "--dtypes",
            "fp16",
            "--sizes",
            "small",
            "--output-format",
            "jsonl",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "split_mode=device-holdout" in result.stdout
    assert "baseline_plus_residual_latency_mae=" in result.stdout
