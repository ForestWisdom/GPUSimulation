# GPU Operator Predictor

A research prototype for single-kernel GPU latency prediction.

## v0 Scope

Phase 1 only scaffolds the repository and public interfaces.

- Single-kernel prediction only
- No distributed inference
- Placeholder implementations only
- Supported kernel families in v0:
  - GEMM/BMM
  - Attention
  - Vector/Fused vector
  - RMSNorm/LayerNorm
  - Fused MoE

## Package Layout

```text
predictor/
  extractor/   # graph, trace, and metadata parsing interfaces
  recognizer/  # kernel family and implementation bucket recognition
  analytical/  # task decomposition, scheduling, features, baseline estimation
  models/      # residual and uncertainty model interfaces
  training/    # dataset builder, trainer, evaluator scaffolds
  serving/     # single-kernel prediction and aggregation entrypoints
tests/
scripts/
```

The 7 conceptual modules required by `AGENTS.md` are scaffolded as:

1. graph/kernel extractor
2. kernel implementation recognizer
3. task decomposer
4. scheduling simulator
5. pipeline-aware feature analyzer
6. baseline + learned predictor
7. operator/model aggregator

## Quick Start

Run the Phase 1 test suite:

```bash
python -m pytest tests -q
```

Run the Phase 1 demo script:

```bash
python scripts/demo_phase1.py
```

Run the placeholder CLI directly:

```bash
python -m predictor.serving --name demo_kernel --family gemm_bmm --dimension m=128 --dimension n=128 --dimension k=64
```

## Implemented in This Phase

- Scaffolded the repository under the required `predictor/`, `tests/`, and `scripts/` layout.
- Added typed shared dataclasses for metadata, recognition, analytical plans, features, datasets, and predictions.
- Added interface-style package boundaries and placeholder implementations for extraction, recognition, analytical processing, learned-model hooks, training, and serving.
- Added a default single-kernel serving pipeline plus an operator-level aggregation helper.
- Added a runnable demo script and a simple CLI entrypoint.
- Added unit tests for each major module and an integration test that exercises the Phase 1 demo path.

## Remaining Work

- Phase 2: implement a real GEMM/BMM analytical baseline, including decomposition, wave estimation, and latency estimation.
- Phase 3: add a lightweight learned residual head, dataset builder, training flow, and evaluation flow.
- Phase 4: extend real logic to Attention and Vector/Fused vector kernels while reusing the same public interfaces.
- Phase 5: replace placeholder uncertainty and aggregation behavior with real uncertainty or p90 prediction and end-to-end toy operator/model aggregation.

## Current Notes

Phase 1 outputs are deterministic placeholders intended to prove package boundaries, test structure, and serving/demo entrypoints. They are not valid performance models yet.
