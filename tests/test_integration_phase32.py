from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_phase32_analysis_cli_runs_and_writes_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = tmp_path / "phase32_real_multi_gpu.jsonl"
    output_dir = tmp_path / "analysis_outputs"
    records = []
    device_names = (
        "NVIDIA GeForce RTX 4090",
        "NVIDIA A40",
        "NVIDIA L40S",
    )
    for device_index, gpu_name in enumerate(device_names):
        for sample_index in range(4):
            family = "gemm" if sample_index % 2 == 0 else "bmm"
            dtype = "fp16" if sample_index % 3 == 0 else ("bf16" if sample_index % 3 == 1 else "fp32")
            aligned_dims = (
                {"m": 256, "n": 256, "k": 256}
                if sample_index % 2 == 0
                else {"batch": 4, "m": 250, "n": 256, "k": 256}
            )
            records.append(
                {
                    "name": f"{family}_{device_index}_{sample_index}",
                    "family_hint": "gemm_bmm",
                    "dtype": dtype,
                    "backend": "cuda",
                    "dimensions": aligned_dims,
                    "device_profile": {
                        "name": gpu_name,
                        "sm_count": 80 + device_index,
                        "memory_bandwidth_bytes_per_s": 1.2e12,
                        "simt_flops_per_s": 16e12,
                        "tensor_core_flops_per_s": 220e12,
                    },
                    "measured_latency_ms": 0.2 + device_index * 0.03 + sample_index * 0.02,
                    "latency_std_ms": 0.001,
                    "num_warmup": 5,
                    "num_repeats": 10,
                    "measurement_backend": "mock",
                    "gpu_name": gpu_name,
                    "torch_version": "mock",
                    "cuda_version": "mock",
                    "seed": 7,
                    "run_tags": {
                        "gpu_name": gpu_name,
                        "dtype": dtype,
                        "family": family,
                        "batch": aligned_dims.get("batch", 1),
                    },
                }
            )
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "sglang",
            "python",
            str(repo_root / "scripts" / "analyze_phase32_holdout.py"),
            "--data",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Main findings" in result.stdout
    assert (output_dir / "phase32_holdout_summary.csv").exists()
    assert (output_dir / "phase32_residual_diagnostics.csv").exists()
    assert (output_dir / "phase32_top_coefficients.csv").exists()
    assert (output_dir / "phase32_random_slice_gpu_name.csv").exists()
