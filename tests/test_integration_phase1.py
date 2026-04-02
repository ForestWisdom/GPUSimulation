from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from predictor.serving import KernelLatencyPredictor


def test_phase1_integration_demo_path() -> None:
    predictor = KernelLatencyPredictor.default()

    prediction = predictor.predict_from_raw_metadata(
        {
            "name": "phase1_attention_kernel",
            "family_hint": "attention",
            "dtype": "bf16",
            "dimensions": {"batch": 2, "seq_len": 128, "heads": 16},
            "backend": "cuda",
            "tags": ["demo"],
        }
    )

    assert prediction.kernel_name == "phase1_attention_kernel"
    assert prediction.implementation_bucket == "attention.placeholder"
    assert prediction.p90_latency_ms >= prediction.mean_latency_ms


def test_phase1_demo_script_runs_directly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "demo_phase1.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "demo_gemm_kernel" in result.stdout
