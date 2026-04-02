from predictor.analytical import AnalyticalTaskDecomposer
from predictor.types import KernelFamily, KernelMetadata


def test_gemm_task_decomposer_returns_gemm_pipeline_tasks() -> None:
    decomposer = AnalyticalTaskDecomposer()
    metadata = KernelMetadata(
        name="gemm_kernel",
        family_hint=KernelFamily.GEMM_BMM,
        dimensions={"m": 256, "n": 128, "k": 64},
        dtype="fp16",
        backend="cuda",
    )

    plan = decomposer.decompose(metadata)

    assert plan.kernel_name == "gemm_kernel"
    assert [task.name for task in plan.tasks] == [
        "load_lhs_tiles",
        "load_rhs_tiles",
        "matrix_multiply_accumulate",
        "store_output_tiles",
    ]
