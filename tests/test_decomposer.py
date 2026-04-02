from predictor.analytical import PlaceholderTaskDecomposer
from predictor.types import KernelFamily, KernelMetadata


def test_placeholder_task_decomposer_returns_single_kernel_task() -> None:
    decomposer = PlaceholderTaskDecomposer()
    metadata = KernelMetadata(
        name="attention_kernel",
        family_hint=KernelFamily.ATTENTION,
        dimensions={"seq_len": 128, "heads": 8},
        dtype="fp16",
        backend="cuda",
    )

    plan = decomposer.decompose(metadata)

    assert plan.kernel_name == "attention_kernel"
    assert len(plan.tasks) == 1
    assert plan.tasks[0].name == "single_kernel"
