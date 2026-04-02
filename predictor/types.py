"""Shared types for the GPU operator predictor scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class KernelFamily(str, Enum):
    """Supported kernel families for the prototype."""

    GEMM_BMM = "gemm_bmm"
    ATTENTION = "attention"
    VECTOR = "vector_fused"
    NORMALIZATION = "normalization"
    FUSED_MOE = "fused_moe"
    UNKNOWN = "unknown"


def normalize_kernel_family(value: str | KernelFamily | None) -> KernelFamily | None:
    """Normalize user-provided kernel family values to the enum."""

    if value is None:
        return None
    if isinstance(value, KernelFamily):
        return value

    normalized = value.strip().lower()
    alias_map = {
        "gemm": KernelFamily.GEMM_BMM,
        "bmm": KernelFamily.GEMM_BMM,
        "gemm_bmm": KernelFamily.GEMM_BMM,
        "attention": KernelFamily.ATTENTION,
        "vector": KernelFamily.VECTOR,
        "fused_vector": KernelFamily.VECTOR,
        "vector_fused": KernelFamily.VECTOR,
        "rmsnorm": KernelFamily.NORMALIZATION,
        "layernorm": KernelFamily.NORMALIZATION,
        "normalization": KernelFamily.NORMALIZATION,
        "fused_moe": KernelFamily.FUSED_MOE,
        "moe": KernelFamily.FUSED_MOE,
    }
    return alias_map.get(normalized, KernelFamily.UNKNOWN)


@dataclass(frozen=True)
class KernelMetadata:
    """Parsed metadata for a single kernel."""

    name: str
    family_hint: KernelFamily | None = None
    dimensions: dict[str, int | float] = field(default_factory=dict)
    dtype: str = "unknown"
    backend: str = "unknown"
    tags: tuple[str, ...] = ()
    extra: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceProfile:
    """Hardware profile used by analytical schedulers and estimators."""

    name: str
    sm_count: int
    memory_bandwidth_bytes_per_s: float
    simt_flops_per_s: float
    tensor_core_flops_per_s: float
    simt_tile_m: int = 64
    simt_tile_n: int = 64
    tensor_core_tile_m: int = 128
    tensor_core_tile_n: int = 128
    simt_launch_overhead_ms: float = 0.014
    tensor_core_launch_overhead_ms: float = 0.010
    simt_wave_penalty_ms: float = 0.006
    tensor_core_wave_penalty_ms: float = 0.003

    def tile_shape_for(self, use_tensor_cores: bool) -> tuple[int, int]:
        """Return the analytical output tile shape for a kernel path."""

        if use_tensor_cores:
            return self.tensor_core_tile_m, self.tensor_core_tile_n
        return self.simt_tile_m, self.simt_tile_n

    def peak_flops_for(self, use_tensor_cores: bool) -> float:
        """Return the peak compute throughput for a kernel path."""

        if use_tensor_cores:
            return self.tensor_core_flops_per_s
        return self.simt_flops_per_s

    def launch_overhead_ms_for(self, use_tensor_cores: bool) -> float:
        """Return the launch overhead for a kernel path."""

        if use_tensor_cores:
            return self.tensor_core_launch_overhead_ms
        return self.simt_launch_overhead_ms

    def wave_penalty_ms_for(self, use_tensor_cores: bool) -> float:
        """Return the per-wave penalty for a kernel path."""

        if use_tensor_cores:
            return self.tensor_core_wave_penalty_ms
        return self.simt_wave_penalty_ms


DEFAULT_DEVICE_PROFILE = DeviceProfile(
    name="nvidia_h100_sxm",
    sm_count=108,
    memory_bandwidth_bytes_per_s=1.555e12,
    simt_flops_per_s=19.5e12,
    tensor_core_flops_per_s=312e12,
)


def is_gemm_bmm_kernel(metadata: KernelMetadata) -> bool:
    """Return whether metadata describes a GEMM/BMM-style kernel."""

    if metadata.family_hint is KernelFamily.GEMM_BMM:
        return True
    return {"m", "n", "k"}.issubset(metadata.dimensions)


def uses_tensor_cores(metadata: KernelMetadata) -> bool:
    """Return whether GEMM/BMM metadata matches the tensor-core path."""

    dtype = metadata.dtype.lower()
    if dtype not in {"fp16", "bf16", "half"}:
        return False

    m = int(metadata.dimensions.get("m", 0))
    n = int(metadata.dimensions.get("n", 0))
    k = int(metadata.dimensions.get("k", 0))
    if min(m, n, k) <= 0:
        return False

    return all(dimension % 16 == 0 for dimension in (m, n, k))


def dtype_size_bytes(dtype: str) -> int:
    """Return the byte width for a kernel datatype."""

    dtype_key = dtype.lower()
    if dtype_key in {"fp16", "bf16", "half"}:
        return 2
    if dtype_key in {"fp32", "float32"}:
        return 4
    return 4


@dataclass(frozen=True)
class RecognitionResult:
    """Recognizer output for a kernel."""

    family: KernelFamily
    implementation_bucket: str
    confidence: float
    notes: str = ""


@dataclass(frozen=True)
class KernelTask:
    """One analytical sub-task within a kernel execution."""

    name: str
    description: str


@dataclass(frozen=True)
class TaskPlan:
    """Analytical decomposition result for a kernel."""

    kernel_name: str
    tasks: tuple[KernelTask, ...]


@dataclass(frozen=True)
class ScheduleEstimate:
    """Scheduling estimate for a kernel task plan."""

    estimated_waves: int
    sm_utilization: float
    tile_count: int = 0
    tiles_per_wave: int = 0


@dataclass(frozen=True)
class FeatureVector:
    """Pipeline-aware features for the learned correction models."""

    values: dict[str, float]


@dataclass(frozen=True)
class BaselineEstimate:
    """Analytical latency estimate before learned corrections."""

    latency_ms: float
    notes: str = ""


@dataclass(frozen=True)
class LatencyPrediction:
    """Serving-layer latency prediction for a single kernel."""

    kernel_name: str
    mean_latency_ms: float
    p90_latency_ms: float
    baseline_latency_ms: float
    implementation_bucket: str


@dataclass(frozen=True)
class AggregationSummary:
    """Aggregated latency summary across multiple kernels."""

    kernel_count: int
    total_mean_latency_ms: float
    total_p90_latency_ms: float


@dataclass(frozen=True)
class TrainingSample:
    """One row in a training dataset."""

    metadata: KernelMetadata
    features: FeatureVector
    target_latency_ms: float


@dataclass(frozen=True)
class KernelDataset:
    """A collection of training samples."""

    samples: tuple[TrainingSample, ...]


@dataclass(frozen=True)
class TrainerState:
    """Result of fitting a placeholder training component."""

    model_name: str
    sample_count: int
