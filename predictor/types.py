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
    """Placeholder scheduling estimate for a kernel task plan."""

    estimated_waves: int
    sm_utilization: float


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
