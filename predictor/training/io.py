"""I/O helpers for residual-training datasets."""

from __future__ import annotations

from pathlib import Path

from predictor.training import GemmBmmDatasetBuilder
from predictor.types import PathLike, ResidualTrainingDataset


def load_gemm_bmm_dataset(path: PathLike) -> ResidualTrainingDataset:
    """Load a GEMM/BMM dataset from JSONL or CSV."""

    dataset_path = Path(path)
    builder = GemmBmmDatasetBuilder()
    if dataset_path.suffix == ".jsonl":
        return builder.from_jsonl(dataset_path)
    if dataset_path.suffix == ".csv":
        return builder.from_csv(dataset_path)
    raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
