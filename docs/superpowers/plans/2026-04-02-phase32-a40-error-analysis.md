# Phase 3.2 A40 Error Analysis Plan

## Scope

Implement a GEMM/BMM-only analysis pass to diagnose why A40 device holdout degrades.

This work adds:

- a reusable analysis module under `predictor/training/analysis.py`
- a CLI at `scripts/analyze_phase32_holdout.py`
- CSV/JSON analysis outputs under `artifacts/`
- README updates for a short Phase 3.2 analysis section

This work does not:

- change the Ridge model family
- add MLP or other models
- start Phase 4
- add Attention, Vector, or MoE logic

## Files To Touch

- `predictor/training/analysis.py`
  Add size-bucket derivation, alignment grouping, slice metrics, residual diagnostics, coefficient export, and holdout experiment helpers.
- `predictor/training/__init__.py`
  Export the new analysis helpers.
- `scripts/analyze_phase32_holdout.py`
  Run random/device-holdout experiments, write analysis outputs, and print a short human-readable findings summary.
- `tests/test_analysis.py`
  Cover size-bucket derivation, alignment grouping, slice metrics, deltas, and coefficient export metadata.
- `tests/test_integration_phase32.py`
  Run the analysis CLI on a small synthetic merged dataset and assert outputs are written.
- `README.md`
  Add a short Phase 3.2 analysis section and the new analysis command.

## TDD Order

1. Write failing unit tests for:
   - explicit `size_bucket` derivation with `size_bucket_rule`
   - three-way alignment grouping
   - slice summaries including MAE/RMSE deltas
   - standardized-space Ridge coefficient export note
2. Write a failing integration test for the analysis CLI on a synthetic multi-device dataset.
3. Implement the smallest analysis module needed to satisfy those tests.
4. Implement the CLI and artifact writers.
5. Update README.
6. Run tests.
7. Run the analysis on `artifacts/phase3_real_multi_gpu.jsonl`.
