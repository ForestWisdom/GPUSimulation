# GPU Operator Predictor

A research prototype for single-kernel GPU latency prediction.

## v0 Scope

Current repository status: Phase 3.1 implemented for GEMM/BMM analytical baseline, Ridge residual head, and real-data validation tooling.

- Single-kernel prediction only
- No distributed inference
- Real analytical baseline plus a learned additive residual head for GEMM/BMM
- Uncertainty modeling is still placeholder-only
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

Run the full test suite with the `sglang` conda environment:

```bash
conda run -n sglang python -m pytest tests -q
```

Run the Phase 3 GEMM/BMM residual demo script:

```bash
conda run -n sglang python scripts/demo_phase3_residual.py
```

Run the CLI directly for a GEMM example:

```bash
conda run -n sglang python -m predictor.serving --name demo_kernel --family gemm_bmm --dtype fp16 --dimension m=2048 --dimension n=2048 --dimension k=4096
```

Train a residual model from JSONL or CSV:

```bash
conda run -n sglang python scripts/train_phase3_residual.py --data path/to/runs.jsonl --output artifacts/gemm_residual.joblib
```

Evaluate the residual model workflow with the default deterministic split:

```bash
conda run -n sglang python scripts/eval_phase3_residual.py --data path/to/runs.jsonl
```

Collect GEMM/BMM profiling data:

```bash
conda run -n sglang python scripts/collect_phase31_gemm_data.py --output artifacts/gemm_profiles.jsonl --format jsonl --mode mock
```

Run the Phase 3.1 validation workflow end-to-end:

```bash
conda run -n sglang python scripts/run_phase31_validation.py --collect-mode mock --split-mode device-holdout --output-format jsonl
```

## Implemented in This Phase

- Added heuristic GEMM/BMM kernel bucket recognition with `gemm.tensor_core`, `gemm.simt`, `bmm.tensor_core`, and `bmm.simt` buckets.
- Added GEMM/BMM analytical task decomposition into operand load, MMA mainloop, and output store stages.
- Added a shared `DeviceProfile` dataclass so analytical scheduling and baseline latency estimation use explicit device assumptions rather than hard-coded constants.
- Replaced the old placeholder pipeline feature path with a real analytical GEMM/BMM feature analyzer that emits FLOPs, bytes, arithmetic intensity, tile shape, tensor-core usage, and device-aware throughput features.
- Wired the serving pipeline to use the real analytical feature analyzer, analytical scheduler, and analytical baseline estimator under one shared device profile.
- Hardened tests for problem-size scaling, tensor-core versus SIMT behavior, SM-count-dependent wave estimation, and device-profile-dependent latency changes.
- Added a GEMM/BMM residual-training dataset schema including kernel metadata, device profile, analytical features, analytical baseline latency, measured latency, and additive residual targets.
- Added a JSONL/CSV dataset builder that derives analytical features and additive residual targets from GEMM/BMM run records.
- Added a small trainable residual model implemented as `StandardScaler + Ridge(alpha=1.0)` saved as one sklearn pipeline artifact.
- Added training, evaluation, and demo scripts for the GEMM/BMM residual head.
- Updated the serving pipeline so final latency is computed as `max(1e-6, analytical_baseline_ms + predicted_residual_ms)`.
- Added a GEMM/BMM profiling/data-collection path that records kernel metadata, device profile, measured latency, latency standard deviation, warmup/repeat counts, measurement backend, GPU name, torch/cuda version, and seed.
- Added a reproducible sampling plan spanning GEMM/BMM, fp16/bf16/fp32, small/medium/large buckets, square-ish/tall-skinny/wide/K-heavy shapes, aligned and non-aligned dimensions, and BMM batch values `1/4/16/64`.
- Added a validation workflow script that can collect or load data, train the Ridge residual model, and evaluate random-split or device-holdout baselines.
- Kept non-GEMM/BMM families on placeholder paths so Phase 3 scope stays narrow.

## Remaining Work

- Phase 4: extend real logic to Attention and Vector/Fused vector kernels while reusing the same public interfaces.
- Phase 5: replace placeholder uncertainty and aggregation behavior with real uncertainty or p90 prediction and end-to-end toy operator/model aggregation.

## Current Notes

- The serving stack now produces device-profile-aware analytical baseline latency plus an optional learned additive residual for GEMM/BMM kernels.
- The default residual model is a small Ridge-regression pipeline and predicts additive residual milliseconds from analytical features only.
- The Phase 3.1 validation workflow supports `random` and `device-holdout` split modes and reports residual MAE/RMSE plus baseline-only and baseline+residual latency MAE/MAPE.
- Uncertainty remains a placeholder uplift and is intentionally unchanged in Phase 3.
- Attention, normalization, vector/fused vector, and fused MoE paths still use placeholder analytical behavior.

## Multi-GPU Data Collection

To collect data on multiple GPUs, run the collection script separately on each machine or device, then merge the generated JSONL files.

Example:

```bash
conda run -n sglang python scripts/collect_phase31_gemm_data.py \
  --output artifacts/h100_profiles.jsonl \
  --format jsonl \
  --mode torch \
  --families gemm,bmm \
  --dtypes fp16,bf16,fp32 \
  --sizes small,medium,large
```

Repeat the same command on another GPU and concatenate the JSONL outputs:

```bash
cat artifacts/h100_profiles.jsonl artifacts/a100_profiles.jsonl > artifacts/multi_gpu_profiles.jsonl
```

Then evaluate with a device-holdout split:

```bash
conda run -n sglang python scripts/run_phase31_validation.py --data artifacts/multi_gpu_profiles.jsonl --split-mode device-holdout
```
