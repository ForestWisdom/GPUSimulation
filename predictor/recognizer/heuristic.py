"""Heuristic recognizer implementation."""

from __future__ import annotations

from predictor.types import KernelFamily, KernelMetadata, RecognitionResult


class HeuristicKernelRecognizer:
    """Assign implementation buckets from kernel metadata hints."""

    def recognize(self, metadata: KernelMetadata) -> RecognitionResult:
        """Recognize a kernel family and implementation bucket."""

        family = metadata.family_hint or self._infer_family_from_name(metadata.name)
        if family is KernelFamily.GEMM_BMM:
            return self._recognize_gemm_bmm(metadata)
        return RecognitionResult(
            family=family,
            implementation_bucket=f"{family.value}.placeholder",
            confidence=0.25,
            notes="Fallback placeholder recognizer path.",
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

    def _recognize_gemm_bmm(self, metadata: KernelMetadata) -> RecognitionResult:
        """Recognize a GEMM/BMM kernel bucket."""

        batch = int(metadata.dimensions.get("batch", 1))
        op_kind = "bmm" if batch > 1 else "gemm"
        implementation = "tensor_core" if _is_tensor_core_eligible(metadata) else "simt"
        return RecognitionResult(
            family=KernelFamily.GEMM_BMM,
            implementation_bucket=f"{op_kind}.{implementation}",
            confidence=0.95,
            notes="Phase 2 GEMM/BMM heuristic recognizer.",
        )


class PlaceholderKernelRecognizer(HeuristicKernelRecognizer):
    """Backward-compatible alias for the default recognizer."""


def _is_tensor_core_eligible(metadata: KernelMetadata) -> bool:
    """Return whether a GEMM/BMM kernel likely uses tensor cores."""

    dtype = metadata.dtype.lower()
    if dtype not in {"fp16", "bf16", "half"}:
        return False
    m = int(metadata.dimensions.get("m", 0))
    n = int(metadata.dimensions.get("n", 0))
    k = int(metadata.dimensions.get("k", 0))
    if min(m, n, k) == 0:
        return False
    return (m % 16 == 0) and (n % 16 == 0) and (k % 16 == 0)
