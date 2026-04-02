"""Placeholder metadata extraction logic."""

from __future__ import annotations

from typing import Any, Mapping

from predictor.types import KernelMetadata, normalize_kernel_family


class PlaceholderMetadataExtractor:
    """Parse raw dictionaries into the project metadata schema."""

    def parse_kernel_metadata(self, raw_metadata: Mapping[str, Any]) -> KernelMetadata:
        """Parse raw kernel metadata for the Phase 1 scaffold."""

        dimensions = {
            str(key): value
            for key, value in dict(raw_metadata.get("dimensions", {})).items()
            if isinstance(value, (int, float))
        }
        tags = tuple(str(tag) for tag in raw_metadata.get("tags", ()))
        known_keys = {"name", "family_hint", "dimensions", "dtype", "backend", "tags"}
        extra = {
            str(key): value
            for key, value in raw_metadata.items()
            if key not in known_keys and isinstance(value, (str, int, float, bool))
        }
        return KernelMetadata(
            name=str(raw_metadata.get("name", "unknown_kernel")),
            family_hint=normalize_kernel_family(raw_metadata.get("family_hint")),
            dimensions=dimensions,
            dtype=str(raw_metadata.get("dtype", "unknown")),
            backend=str(raw_metadata.get("backend", "unknown")),
            tags=tags,
            extra=extra,
        )
