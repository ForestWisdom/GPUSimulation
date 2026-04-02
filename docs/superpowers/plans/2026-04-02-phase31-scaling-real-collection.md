# Phase 3.1 Scaling Real Collection Plan

## Scope

Implement the next Phase 3.1 iteration for GEMM/BMM only:

- fix `scripts/train_phase3_residual.py` and `scripts/eval_phase3_residual.py` so they run without `PYTHONPATH=.`
- add multi-round collection to `scripts/collect_phase3_gemm_data.py`
- keep the sampling plan reproducible across rounds
- when `--rounds > 1`, append to one output file and add `round_id`
- expose real `--split-mode random|device-holdout` support in `scripts/eval_phase3_residual.py`
- keep Ridge residual modeling unchanged
- update README with larger real-data collection commands for RTX 4090, A40, and L40S

## Files To Touch

- `scripts/train_phase3_residual.py`
  Add repo-root import bootstrap so direct script execution works.
- `scripts/eval_phase3_residual.py`
  Add repo-root import bootstrap and real split-mode CLI handling.
- `scripts/collect_phase3_gemm_data.py`
  Add `--rounds`, annotate `round_id`, and append records across rounds.
- `scripts/run_phase3_validation.py`
  Optionally mirror `--rounds` for parity with collection-driven validation.
- `predictor/training/profiling.py`
  Extend record writers to support append mode and flatten `round_id` for CSV.
- `tests/test_integration_phase31.py`
  Add failing coverage for direct train/eval entrypoints and multi-round collection.
- `tests/test_profiling.py`
  Add focused coverage for `round_id` and append behavior.
- `README.md`
  Document the new workflow and GPU-specific real-data commands.

## Execution Order

1. Write failing tests for:
   - direct `train_phase3_residual.py` execution without `PYTHONPATH`
   - direct `eval_phase3_residual.py --split-mode device-holdout`
   - `collect_phase3_gemm_data.py --rounds 2` producing appended records with `round_id`
2. Implement the smallest code changes to make those tests pass.
3. Update README commands and notes.
4. Run the full test suite in `sglang`.
5. Run one mocked end-to-end workflow with the new Phase 3.1 entrypoints.
