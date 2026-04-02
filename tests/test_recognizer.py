from predictor.recognizer import HeuristicKernelRecognizer
from predictor.types import KernelFamily, KernelMetadata


def test_recognizer_assigns_tensor_core_bucket_for_fp16_gemm() -> None:
    recognizer = HeuristicKernelRecognizer()
    metadata = KernelMetadata(
        name="gemm_tensor_core_kernel",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 256, "n": 128, "k": 64},
        dtype="fp16",
        backend="cuda",
    )

    result = recognizer.recognize(metadata)

    assert result.family is KernelFamily.GEMM_BMM
    assert result.implementation_bucket == "gemm.tensor_core"
    assert result.confidence >= 0.9


def test_recognizer_assigns_simt_bucket_for_fp32_bmm() -> None:
    recognizer = HeuristicKernelRecognizer()
    metadata = KernelMetadata(
        name="bmm_simt_kernel",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"batch": 8, "m": 96, "n": 80, "k": 48},
        dtype="fp32",
        backend="cuda",
    )

    result = recognizer.recognize(metadata)

    assert result.family is KernelFamily.GEMM_BMM
    assert result.implementation_bucket == "bmm.simt"
