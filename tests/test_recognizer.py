from predictor.recognizer import PlaceholderKernelRecognizer
from predictor.types import KernelFamily, KernelMetadata


def test_placeholder_recognizer_assigns_bucket_from_metadata() -> None:
    recognizer = PlaceholderKernelRecognizer()
    metadata = KernelMetadata(
        name="bmm_cutlass_kernel",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"batch": 4, "m": 64, "n": 64, "k": 128},
        dtype="fp16",
        backend="cuda",
    )

    result = recognizer.recognize(metadata)

    assert result.family is KernelFamily.GEMM_BMM
    assert result.implementation_bucket == "gemm_bmm.placeholder"
    assert result.confidence == 0.25
