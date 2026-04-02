from __future__ import annotations

import subprocess
from pathlib import Path

from predictor.serving import KernelLatencyPredictor


def test_phase2_integration_predicts_bmm_tensor_core_path() -> None:
    predictor = KernelLatencyPredictor.default()

    prediction = predictor.predict_from_raw_metadata(
        {
            "name": "phase2_bmm_kernel",
            "family_hint": "gemm_bmm",
            "dtype": "bf16",
            "dimensions": {"batch": 8, "m": 1024, "n": 1024, "k": 2048},
            "backend": "cuda",
            "extra": {"sm_count": 108},
        }
    )

    assert prediction.implementation_bucket == "bmm.tensor_core"
    assert prediction.baseline_latency_ms > 0.0
    assert prediction.p90_latency_ms >= prediction.mean_latency_ms


def test_phase2_demo_script_runs_directly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "sglang",
            "python",
            str(repo_root / "scripts" / "demo_phase2_gemm.py"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "demo_bmm_tensor_core" in result.stdout
