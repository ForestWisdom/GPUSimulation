"""Recognizer interfaces."""

from __future__ import annotations

from typing import Protocol

from predictor.types import KernelMetadata, RecognitionResult


class KernelRecognizer(Protocol):
    """Interface for kernel implementation recognition."""

    def recognize(self, metadata: KernelMetadata) -> RecognitionResult:
        """Recognize the family and implementation bucket for a kernel."""

