"""Training dataset builders."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping

from predictor.analytical import (
    AnalyticalBaselineLatencyEstimator,
    AnalyticalPipelineFeatureAnalyzer,
    AnalyticalSchedulingSimulator,
    AnalyticalTaskDecomposer,
)
from predictor.recognizer import HeuristicKernelRecognizer
from predictor.types import FeatureVector, KernelDataset, KernelMetadata, TrainingSample
from predictor.types import (
    DEFAULT_DEVICE_PROFILE,
    DeviceProfile,
    KernelFamily,
    PathLike,
    ResidualTrainingDataset,
    ResidualTrainingSample,
    is_gemm_bmm_kernel,
    normalize_kernel_family,
)


class KernelDatasetBuilder:
    """Build training datasets from normalized rows."""

    def build(
        self,
        rows: Iterable[tuple[KernelMetadata, FeatureVector, float]],
    ) -> KernelDataset:
        """Build a placeholder dataset for residual model training."""

        samples = tuple(
            TrainingSample(
                metadata=metadata,
                features=features,
                target_latency_ms=target_latency_ms,
            )
            for metadata, features, target_latency_ms in rows
        )
        return KernelDataset(samples=samples)


class GemmBmmDatasetBuilder:
    """Build GEMM/BMM residual-training datasets from records or files."""

    def __init__(self, target_mode: str = "additive") -> None:
        """Initialize the dataset builder."""

        self.target_mode = target_mode
        self._recognizer = HeuristicKernelRecognizer()

    def build_from_records(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> ResidualTrainingDataset:
        """Build a residual-training dataset from normalized records."""

        samples = tuple(self._build_sample(record) for record in records)
        return ResidualTrainingDataset(samples=samples)

    def from_jsonl(self, path: PathLike) -> ResidualTrainingDataset:
        """Load a residual-training dataset from a JSONL file."""

        with Path(path).open("r", encoding="utf-8") as file_obj:
            records = [
                json.loads(line)
                for line in file_obj
                if line.strip()
            ]
        return self.build_from_records(records)

    def from_csv(self, path: PathLike) -> ResidualTrainingDataset:
        """Load a residual-training dataset from a CSV file."""

        with Path(path).open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            records = [self._csv_row_to_record(row) for row in reader]
        return self.build_from_records(records)

    def _build_sample(self, record: Mapping[str, Any]) -> ResidualTrainingSample:
        """Build one residual-training sample from a normalized record."""

        metadata = self._build_metadata(record)
        if not is_gemm_bmm_kernel(metadata):
            raise ValueError("Phase 3 dataset builder only supports GEMM/BMM records.")

        device_profile = self._build_device_profile(record)
        measured_latency_ms = float(record["measured_latency_ms"])
        decomposer = AnalyticalTaskDecomposer()
        scheduler = AnalyticalSchedulingSimulator(device_profile=device_profile)
        feature_analyzer = AnalyticalPipelineFeatureAnalyzer(
            device_profile=device_profile,
        )
        baseline_estimator = AnalyticalBaselineLatencyEstimator(
            device_profile=device_profile,
        )

        plan = decomposer.decompose(metadata)
        schedule = scheduler.simulate(plan, metadata)
        features = feature_analyzer.analyze(metadata, schedule)
        baseline = baseline_estimator.estimate(metadata, plan, schedule, features)
        recognition = self._recognizer.recognize(metadata)
        residual_target_ms = self._derive_target(
            measured_latency_ms=measured_latency_ms,
            analytical_baseline_ms=baseline.latency_ms,
        )
        return ResidualTrainingSample(
            metadata=metadata,
            device_profile=device_profile,
            features=features,
            analytical_baseline_ms=baseline.latency_ms,
            measured_latency_ms=measured_latency_ms,
            residual_target_ms=round(residual_target_ms, 6),
            implementation_bucket=recognition.implementation_bucket,
        )

    def _build_metadata(self, record: Mapping[str, Any]) -> KernelMetadata:
        """Construct kernel metadata from a normalized record."""

        dimensions = {
            key: int(value)
            for key, value in dict(record.get("dimensions", {})).items()
            if value not in (None, "")
        }
        return KernelMetadata(
            name=str(record.get("name", "unknown_kernel")),
            family_hint=normalize_kernel_family(record.get("family_hint"))
            or KernelFamily.GEMM_BMM,
            dimensions=dimensions,
            dtype=str(record.get("dtype", "unknown")),
            backend=str(record.get("backend", "unknown")),
        )

    def _build_device_profile(self, record: Mapping[str, Any]) -> DeviceProfile:
        """Construct a device profile from a normalized record."""

        profile_data = dict(record.get("device_profile", {}))
        if not profile_data:
            return DEFAULT_DEVICE_PROFILE
        return DeviceProfile(
            name=str(profile_data.get("name", DEFAULT_DEVICE_PROFILE.name)),
            sm_count=int(profile_data.get("sm_count", DEFAULT_DEVICE_PROFILE.sm_count)),
            memory_bandwidth_bytes_per_s=float(
                profile_data.get(
                    "memory_bandwidth_bytes_per_s",
                    DEFAULT_DEVICE_PROFILE.memory_bandwidth_bytes_per_s,
                )
            ),
            simt_flops_per_s=float(
                profile_data.get(
                    "simt_flops_per_s",
                    DEFAULT_DEVICE_PROFILE.simt_flops_per_s,
                )
            ),
            tensor_core_flops_per_s=float(
                profile_data.get(
                    "tensor_core_flops_per_s",
                    DEFAULT_DEVICE_PROFILE.tensor_core_flops_per_s,
                )
            ),
            simt_tile_m=int(
                profile_data.get("simt_tile_m", DEFAULT_DEVICE_PROFILE.simt_tile_m)
            ),
            simt_tile_n=int(
                profile_data.get("simt_tile_n", DEFAULT_DEVICE_PROFILE.simt_tile_n)
            ),
            tensor_core_tile_m=int(
                profile_data.get(
                    "tensor_core_tile_m",
                    DEFAULT_DEVICE_PROFILE.tensor_core_tile_m,
                )
            ),
            tensor_core_tile_n=int(
                profile_data.get(
                    "tensor_core_tile_n",
                    DEFAULT_DEVICE_PROFILE.tensor_core_tile_n,
                )
            ),
            simt_launch_overhead_ms=float(
                profile_data.get(
                    "simt_launch_overhead_ms",
                    DEFAULT_DEVICE_PROFILE.simt_launch_overhead_ms,
                )
            ),
            tensor_core_launch_overhead_ms=float(
                profile_data.get(
                    "tensor_core_launch_overhead_ms",
                    DEFAULT_DEVICE_PROFILE.tensor_core_launch_overhead_ms,
                )
            ),
            simt_wave_penalty_ms=float(
                profile_data.get(
                    "simt_wave_penalty_ms",
                    DEFAULT_DEVICE_PROFILE.simt_wave_penalty_ms,
                )
            ),
            tensor_core_wave_penalty_ms=float(
                profile_data.get(
                    "tensor_core_wave_penalty_ms",
                    DEFAULT_DEVICE_PROFILE.tensor_core_wave_penalty_ms,
                )
            ),
        )

    def _derive_target(
        self,
        measured_latency_ms: float,
        analytical_baseline_ms: float,
    ) -> float:
        """Derive the training target according to the configured target mode."""

        if self.target_mode != "additive":
            raise ValueError(
                f"Unsupported target_mode={self.target_mode!r}; "
                "Phase 3 only implements additive residuals.",
            )
        return measured_latency_ms - analytical_baseline_ms

    def _csv_row_to_record(self, row: Mapping[str, str]) -> dict[str, Any]:
        """Normalize a CSV row into the internal record schema."""

        dimensions = {
            key: int(row[key])
            for key in ("batch", "m", "n", "k")
            if row.get(key)
        }
        return {
            "name": row.get("name", "unknown_kernel"),
            "family_hint": row.get("family_hint", "gemm_bmm"),
            "dtype": row.get("dtype", "unknown"),
            "backend": row.get("backend", "unknown"),
            "dimensions": dimensions,
            "device_profile": {
                "name": row.get("device_name", DEFAULT_DEVICE_PROFILE.name),
                "sm_count": int(row.get("sm_count", DEFAULT_DEVICE_PROFILE.sm_count)),
                "memory_bandwidth_bytes_per_s": float(
                    row.get(
                        "memory_bandwidth_bytes_per_s",
                        DEFAULT_DEVICE_PROFILE.memory_bandwidth_bytes_per_s,
                    )
                ),
                "simt_flops_per_s": float(
                    row.get(
                        "simt_flops_per_s",
                        DEFAULT_DEVICE_PROFILE.simt_flops_per_s,
                    )
                ),
                "tensor_core_flops_per_s": float(
                    row.get(
                        "tensor_core_flops_per_s",
                        DEFAULT_DEVICE_PROFILE.tensor_core_flops_per_s,
                    )
                ),
            },
            "measured_latency_ms": float(row["measured_latency_ms"]),
        }
