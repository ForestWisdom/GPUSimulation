"""Placeholder recognizer implementation."""

from __future__ import annotations

from predictor.types import KernelFamily, KernelMetadata, RecognitionResult


class PlaceholderKernelRecognizer:
    """Assign placeholder implementation buckets from metadata hints."""

    def recognize(self, metadata: KernelMetadata) -> RecognitionResult:
        """Recognize a kernel family and placeholder implementation bucket."""

        family = metadata.family_hint or self._infer_family_from_name(metadata.name)
        return RecognitionResult(
            family=family,
            implementation_bucket=f"{family.value}.placeholder",
            confidence=0.25,
            notes="Phase 1 heuristic recognizer placeholder.",
        )

    def _infer_family_from_name(self, kernel_name: str) -> KernelFamily:
        """Infer a kernel family from the kernel name."""

        normalized_name = kernel_name.lower()
        if "gemm" in normalized_name or "bmm" in normalized_name:
            return KernelFamily.GEMM_BMM
        if "attention" in normalized_name:
            return KernelFamily.ATTENTION
        if "norm" in normalized_name:
            return KernelFamily.NORMALIZATION
        if "moe" in normalized_name:
            return KernelFamily.FUSED_MOE
        if "vector" in normalized_name or "fused" in normalized_name:
            return KernelFamily.VECTOR
        return KernelFamily.UNKNOWN
