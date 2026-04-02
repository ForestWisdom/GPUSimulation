"""Extractor interfaces."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from predictor.types import KernelMetadata


class KernelMetadataExtractor(Protocol):
    """Interface for parsing raw kernel metadata."""

    def parse_kernel_metadata(self, raw_metadata: Mapping[str, Any]) -> KernelMetadata:
        """Parse raw metadata into a normalized kernel metadata object."""

