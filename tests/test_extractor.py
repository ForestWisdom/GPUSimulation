from predictor.extractor import PlaceholderMetadataExtractor
from predictor.types import KernelFamily


def test_placeholder_extractor_parses_kernel_metadata() -> None:
    extractor = PlaceholderMetadataExtractor()

    metadata = extractor.parse_kernel_metadata(
        {
            "name": "gemm_kernel",
            "family_hint": "gemm_bmm",
            "dtype": "fp16",
            "dimensions": {"m": 128, "n": 256, "k": 64},
            "backend": "cuda",
            "tags": ["phase1"],
        }
    )

    assert metadata.name == "gemm_kernel"
    assert metadata.family_hint is KernelFamily.GEMM_BMM
    assert metadata.dimensions["m"] == 128
    assert metadata.tags == ("phase1",)
