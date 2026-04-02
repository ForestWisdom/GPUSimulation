from __future__ import annotations

import subprocess
from pathlib import Path


def test_phase3_demo_script_runs_directly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "sglang",
            "python",
            str(repo_root / "scripts" / "demo_phase3_residual.py"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "demo_phase3_inference" in result.stdout
