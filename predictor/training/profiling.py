"""Profiling helpers for GEMM/BMM data collection."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from predictor.analytical import (
    AnalyticalBaselineLatencyEstimator,
    AnalyticalPipelineFeatureAnalyzer,
    AnalyticalSchedulingSimulator,
    AnalyticalTaskDecomposer,
)
from predictor.types import DEFAULT_DEVICE_PROFILE, DeviceProfile, KernelFamily, KernelMetadata


@dataclass(frozen=True)
class GemmBmmShapeSpec:
    """One reproducible GEMM/BMM sampling-plan entry."""

    family: str
    dtype: str
    size_bucket: str
    shape_pattern: str
    aligned: bool
    batch: int
    m: int
    n: int
    k: int


def build_gemm_bmm_sampling_plan(
    families: tuple[str, ...] = ("gemm", "bmm"),
    dtypes: tuple[str, ...] = ("fp16", "bf16", "fp32"),
    size_buckets: tuple[str, ...] = ("small", "medium", "large"),
) -> list[GemmBmmShapeSpec]:
    """Build a reproducible GEMM/BMM sampling plan."""

    shape_table = {
        "small": {
            "square-ish": {"aligned": (256, 256, 256), "non_aligned": (250, 250, 250)},
            "tall-skinny": {"aligned": (1024, 128, 256), "non_aligned": (1000, 120, 250)},
            "wide": {"aligned": (128, 1024, 256), "non_aligned": (120, 1000, 250)},
            "k-heavy": {"aligned": (256, 256, 2048), "non_aligned": (250, 250, 2050)},
        },
        "medium": {
            "square-ish": {"aligned": (1024, 1024, 1024), "non_aligned": (1000, 1000, 1000)},
            "tall-skinny": {"aligned": (4096, 512, 1024), "non_aligned": (4032, 500, 990)},
            "wide": {"aligned": (512, 4096, 1024), "non_aligned": (500, 4032, 990)},
            "k-heavy": {"aligned": (1024, 1024, 8192), "non_aligned": (1000, 1000, 8178)},
        },
        "large": {
            "square-ish": {"aligned": (4096, 4096, 4096), "non_aligned": (4080, 4080, 4080)},
            "tall-skinny": {"aligned": (8192, 1024, 4096), "non_aligned": (8176, 1008, 4082)},
            "wide": {"aligned": (1024, 8192, 4096), "non_aligned": (1008, 8176, 4082)},
            "k-heavy": {"aligned": (4096, 4096, 16384), "non_aligned": (4080, 4080, 16370)},
        },
    }
    bmm_batches = (1, 4, 16, 64)
    specs: list[GemmBmmShapeSpec] = []
    for family in families:
        for dtype in dtypes:
            for size_bucket in size_buckets:
                for shape_pattern, variants in shape_table[size_bucket].items():
                    for aligned, key in ((True, "aligned"), (False, "non_aligned")):
                        m, n, k = variants[key]
                        batches = (1,) if family == "gemm" else bmm_batches
                        for batch in batches:
                            specs.append(
                                GemmBmmShapeSpec(
                                    family=family,
                                    dtype=dtype,
                                    size_bucket=size_bucket,
                                    shape_pattern=shape_pattern,
                                    aligned=aligned,
                                    batch=batch,
                                    m=m,
                                    n=n,
                                    k=k,
                                )
                            )
    return specs


def collect_gemm_bmm_profile_records(
    plan: Iterable[GemmBmmShapeSpec],
    mode: str,
    num_warmup: int,
    num_repeats: int,
    seed: int,
    gpu_names: tuple[str, ...] = ("mock_gpu_a", "mock_gpu_b"),
) -> list[dict[str, object]]:
    """Collect GEMM/BMM profiling records in mock or torch mode."""

    specs = list(plan)
    if mode == "mock":
        return [
            _mock_profile_record(
                spec=spec,
                num_warmup=num_warmup,
                num_repeats=num_repeats,
                seed=seed,
                gpu_name=gpu_names[index % len(gpu_names)],
            )
            for index, spec in enumerate(specs)
        ]
    if mode == "torch":
        return [
            _torch_profile_record(
                spec=spec,
                num_warmup=num_warmup,
                num_repeats=num_repeats,
                seed=seed,
            )
            for spec in specs
        ]
    raise ValueError(f"Unsupported profiling mode: {mode}")


def write_profile_records_jsonl(records: Iterable[dict[str, object]], path: Path) -> None:
    """Write profiling records to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record) + "\n")


def write_profile_records_csv(records: Iterable[dict[str, object]], path: Path) -> None:
    """Write profiling records to CSV."""

    record_list = [_flatten_record_for_csv(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not record_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for record in record_list for key in record})
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record_list)


def _mock_profile_record(
    spec: GemmBmmShapeSpec,
    num_warmup: int,
    num_repeats: int,
    seed: int,
    gpu_name: str,
) -> dict[str, object]:
    """Generate a deterministic mock profiling record."""

    device_profile = _profile_for_gpu_name(gpu_name)
    metadata = KernelMetadata(
        name=_kernel_name(spec),
        family_hint=KernelFamily.GEMM_BMM,
        dimensions=_dimensions_for_spec(spec),
        dtype=spec.dtype,
        backend="cuda",
    )
    analytical = _analytical_latency_ms(metadata, device_profile)
    noise_key = hash((gpu_name, spec.family, spec.dtype, spec.batch, spec.m, spec.n, spec.k, seed))
    delta = ((noise_key % 21) - 10) / 1000.0
    measured_latency_ms = round(max(1e-6, analytical + delta), 6)
    latency_std_ms = round(abs(delta) / 3.0 + 0.001, 6)
    return _build_record(
        spec=spec,
        device_profile=device_profile,
        measured_latency_ms=measured_latency_ms,
        latency_std_ms=latency_std_ms,
        num_warmup=num_warmup,
        num_repeats=num_repeats,
        measurement_backend="mock",
        gpu_name=gpu_name,
        torch_version="mock",
        cuda_version="mock",
        seed=seed,
    )


def _torch_profile_record(
    spec: GemmBmmShapeSpec,
    num_warmup: int,
    num_repeats: int,
    seed: int,
) -> dict[str, object]:
    """Collect one profiling record using strict torch CUDA timing."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch mode requires CUDA.")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    props = torch.cuda.get_device_properties(device)
    device_profile = DeviceProfile(
        name=gpu_name,
        sm_count=int(props.multi_processor_count),
        memory_bandwidth_bytes_per_s=DEFAULT_DEVICE_PROFILE.memory_bandwidth_bytes_per_s,
        simt_flops_per_s=DEFAULT_DEVICE_PROFILE.simt_flops_per_s,
        tensor_core_flops_per_s=DEFAULT_DEVICE_PROFILE.tensor_core_flops_per_s,
    )
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[spec.dtype]
    dims = _dimensions_for_spec(spec)
    left_shape = (dims["m"], dims["k"]) if spec.family == "gemm" else (dims["batch"], dims["m"], dims["k"])
    right_shape = (dims["k"], dims["n"]) if spec.family == "gemm" else (dims["batch"], dims["k"], dims["n"])
    with torch.no_grad():
        left = torch.randn(left_shape, device=device, dtype=torch_dtype)
        right = torch.randn(right_shape, device=device, dtype=torch_dtype)
        op = torch.matmul if spec.family == "gemm" else torch.bmm
        for _ in range(num_warmup):
            _ = op(left, right)
        torch.cuda.synchronize(device)
        latencies: list[float] = []
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start.record()
            _ = op(left, right)
            end.record()
            torch.cuda.synchronize(device)
            latencies.append(float(start.elapsed_time(end)))
    return _build_record(
        spec=spec,
        device_profile=device_profile,
        measured_latency_ms=round(statistics.mean(latencies), 6),
        latency_std_ms=round(statistics.pstdev(latencies), 6),
        num_warmup=num_warmup,
        num_repeats=num_repeats,
        measurement_backend="torch_cuda_event",
        gpu_name=gpu_name,
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        seed=seed,
    )


def _build_record(
    spec: GemmBmmShapeSpec,
    device_profile: DeviceProfile,
    measured_latency_ms: float,
    latency_std_ms: float,
    num_warmup: int,
    num_repeats: int,
    measurement_backend: str,
    gpu_name: str,
    torch_version: str,
    cuda_version: str,
    seed: int,
) -> dict[str, object]:
    """Construct a profiling record compatible with the dataset builder."""

    return {
        "name": _kernel_name(spec),
        "family_hint": "gemm_bmm",
        "dtype": spec.dtype,
        "backend": "cuda",
        "dimensions": _dimensions_for_spec(spec),
        "device_profile": asdict(device_profile),
        "measured_latency_ms": measured_latency_ms,
        "latency_std_ms": latency_std_ms,
        "num_warmup": num_warmup,
        "num_repeats": num_repeats,
        "measurement_backend": measurement_backend,
        "gpu_name": gpu_name,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "seed": seed,
        "run_tags": {
            "gpu_name": gpu_name,
            "dtype": spec.dtype,
            "family": spec.family,
            "batch": spec.batch,
        },
    }


def _kernel_name(spec: GemmBmmShapeSpec) -> str:
    """Build a descriptive kernel name from a shape spec."""

    align_tag = "aligned" if spec.aligned else "unaligned"
    return f"{spec.family}_{spec.size_bucket}_{spec.shape_pattern}_{align_tag}_{spec.dtype}_b{spec.batch}"


def _dimensions_for_spec(spec: GemmBmmShapeSpec) -> dict[str, int]:
    """Convert a shape spec to metadata dimensions."""

    dimensions = {"m": spec.m, "n": spec.n, "k": spec.k}
    if spec.family == "bmm":
        dimensions["batch"] = spec.batch
    return dimensions


def _profile_for_gpu_name(gpu_name: str) -> DeviceProfile:
    """Create a deterministic mock device profile from a GPU name."""

    return DeviceProfile(
        name=gpu_name,
        sm_count=DEFAULT_DEVICE_PROFILE.sm_count if gpu_name.endswith("a") else 80,
        memory_bandwidth_bytes_per_s=DEFAULT_DEVICE_PROFILE.memory_bandwidth_bytes_per_s,
        simt_flops_per_s=DEFAULT_DEVICE_PROFILE.simt_flops_per_s,
        tensor_core_flops_per_s=DEFAULT_DEVICE_PROFILE.tensor_core_flops_per_s,
    )


def _analytical_latency_ms(metadata: KernelMetadata, device_profile: DeviceProfile) -> float:
    """Compute one analytical baseline latency for mock data generation."""

    decomposer = AnalyticalTaskDecomposer()
    scheduler = AnalyticalSchedulingSimulator(device_profile=device_profile)
    feature_analyzer = AnalyticalPipelineFeatureAnalyzer(device_profile=device_profile)
    baseline_estimator = AnalyticalBaselineLatencyEstimator(device_profile=device_profile)
    plan = decomposer.decompose(metadata)
    schedule = scheduler.simulate(plan, metadata)
    features = feature_analyzer.analyze(metadata, schedule)
    baseline = baseline_estimator.estimate(metadata, plan, schedule, features)
    return baseline.latency_ms


def _flatten_record_for_csv(record: dict[str, object]) -> dict[str, object]:
    """Flatten a profiling record for CSV output."""

    dimensions = dict(record.get("dimensions", {}))
    device_profile = dict(record.get("device_profile", {}))
    run_tags = dict(record.get("run_tags", {}))
    return {
        "name": record["name"],
        "family_hint": record["family_hint"],
        "dtype": record["dtype"],
        "backend": record["backend"],
        "batch": dimensions.get("batch", 1),
        "m": dimensions.get("m"),
        "n": dimensions.get("n"),
        "k": dimensions.get("k"),
        "gpu_name": record["gpu_name"],
        "device_name": device_profile.get("name"),
        "sm_count": device_profile.get("sm_count"),
        "memory_bandwidth_bytes_per_s": device_profile.get("memory_bandwidth_bytes_per_s"),
        "simt_flops_per_s": device_profile.get("simt_flops_per_s"),
        "tensor_core_flops_per_s": device_profile.get("tensor_core_flops_per_s"),
        "measured_latency_ms": record["measured_latency_ms"],
        "latency_std_ms": record["latency_std_ms"],
        "num_warmup": record["num_warmup"],
        "num_repeats": record["num_repeats"],
        "measurement_backend": record["measurement_backend"],
        "torch_version": record["torch_version"],
        "cuda_version": record["cuda_version"],
        "seed": record["seed"],
        "tag_gpu_name": run_tags.get("gpu_name"),
        "tag_dtype": run_tags.get("dtype"),
        "tag_family": run_tags.get("family"),
        "tag_batch": run_tags.get("batch"),
    }
