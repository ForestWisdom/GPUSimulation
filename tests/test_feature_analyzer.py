from dataclasses import replace

from predictor.analytical import (
    AnalyticalPipelineFeatureAnalyzer,
    AnalyticalSchedulingSimulator,
    AnalyticalTaskDecomposer,
)
from predictor.types import DEFAULT_DEVICE_PROFILE, KernelFamily, KernelMetadata


def test_feature_analyzer_emits_device_aware_gemm_features() -> None:
    device_profile = replace(DEFAULT_DEVICE_PROFILE, name="test_gpu", sm_count=120)
    analyzer = AnalyticalPipelineFeatureAnalyzer(device_profile=device_profile)
    scheduler = AnalyticalSchedulingSimulator(device_profile=device_profile)
    decomposer = AnalyticalTaskDecomposer()
    metadata = KernelMetadata(
        name="gemm_tensor_core_kernel",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 2048, "n": 1024, "k": 4096},
        dtype="fp16",
        backend="cuda",
    )
    plan = decomposer.decompose(metadata)
    schedule = scheduler.simulate(plan, metadata)

    features = analyzer.analyze(metadata, schedule)

    assert features.values["device_sm_count"] == 120.0
    assert features.values["uses_tensor_cores"] == 1.0
    assert features.values["tile_m"] == float(device_profile.tensor_core_tile_m)
    assert features.values["peak_compute_flops"] == device_profile.tensor_core_flops_per_s
    assert features.values["flops"] > features.values["bytes"]


def test_feature_analyzer_distinguishes_tensor_core_and_simt_paths() -> None:
    analyzer = AnalyticalPipelineFeatureAnalyzer()
    scheduler = AnalyticalSchedulingSimulator()
    decomposer = AnalyticalTaskDecomposer()

    tensor_core_metadata = KernelMetadata(
        name="gemm_tensor_core_kernel",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 1024, "n": 1024, "k": 2048},
        dtype="bf16",
        backend="cuda",
    )
    simt_metadata = KernelMetadata(
        name="gemm_simt_kernel",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 1024, "n": 1024, "k": 2048},
        dtype="fp32",
        backend="cuda",
    )

    tensor_core_features = analyzer.analyze(
        tensor_core_metadata,
        scheduler.simulate(decomposer.decompose(tensor_core_metadata), tensor_core_metadata),
    )
    simt_features = analyzer.analyze(
        simt_metadata,
        scheduler.simulate(decomposer.decompose(simt_metadata), simt_metadata),
    )

    assert tensor_core_features.values["uses_tensor_cores"] == 1.0
    assert simt_features.values["uses_tensor_cores"] == 0.0
    assert (
        tensor_core_features.values["peak_compute_flops"]
        > simt_features.values["peak_compute_flops"]
    )
