# GPU Operator Predictor

A research prototype for single-kernel GPU latency prediction.

## v0 Scope

Current repository status: Phase 3.1 implemented for GEMM/BMM analytical baseline, Ridge residual head, and scalable real-data validation tooling.

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

Evaluate the residual model workflow with a configurable split:

```bash
conda run -n sglang python scripts/eval_phase3_residual.py --data path/to/runs.jsonl --split-mode random
```

Collect GEMM/BMM profiling data:

```bash
conda run -n sglang python scripts/collect_phase3_gemm_data.py --output artifacts/gemm_profiles.jsonl --format jsonl --mode mock --families gemm,bmm --dtypes fp16,bf16,fp32 --sizes small,medium,large --warmup 10 --repeats 20 --seed 7 --rounds 2
```

Run the Phase 3.1 validation workflow end-to-end:

```bash
conda run -n sglang python scripts/run_phase3_validation.py --mode mock --split-mode device-holdout --format jsonl --families gemm,bmm --dtypes fp16,bf16,fp32 --sizes small,medium,large --warmup 10 --repeats 20 --seed 7 --rounds 2
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
- Added multi-round collection support so one command can append repeated runs of the same reproducible shape plan into one JSONL or CSV file, with `round_id` included when `--rounds > 1`.
- Fixed the standalone training and evaluation script entrypoints so they run without requiring `PYTHONPATH=.`
- Added validation workflow support for both `random` and `device-holdout` splits in the dedicated evaluation CLI and the Phase 3.1 end-to-end validation CLI.
- Kept non-GEMM/BMM families on placeholder paths so Phase 3 scope stays narrow.

## Remaining Work

- Phase 4: extend real logic to Attention and Vector/Fused vector kernels while reusing the same public interfaces.
- Phase 5: replace placeholder uncertainty and aggregation behavior with real uncertainty or p90 prediction and end-to-end toy operator/model aggregation.

## Current Notes

- The serving stack now produces device-profile-aware analytical baseline latency plus an optional learned additive residual for GEMM/BMM kernels.
- The default residual model is a small Ridge-regression pipeline and predicts additive residual milliseconds from analytical features only.
- The Phase 3.1 validation workflow supports `random` and `device-holdout` split modes and reports residual MAE/RMSE plus baseline-only and baseline+residual latency MAE/MAPE.
- The `device-holdout` path uses the collected GPU/device tag, which is stored as the device profile name and mirrored by `gpu_name` in the profiling records.
- Uncertainty remains a placeholder uplift and is intentionally unchanged in Phase 3.
- Attention, normalization, vector/fused vector, and fused MoE paths still use placeholder analytical behavior.

## Multi-GPU Data Collection

To collect data on multiple GPUs, run the collection script separately on each machine or device, then merge the generated JSONL files.

Example:

```bash
conda run -n sglang python scripts/collect_phase3_gemm_data.py \
  --output artifacts/h100_profiles.jsonl \
  --format jsonl \
  --mode torch \
  --families gemm,bmm \
  --dtypes fp16,bf16,fp32 \
  --sizes small,medium,large \
  --warmup 20 \
  --repeats 50 \
  --seed 7 \
  --rounds 3
```

Repeat the same command on another GPU and concatenate the JSONL outputs:

```bash
cat artifacts/h100_profiles.jsonl artifacts/a100_profiles.jsonl > artifacts/multi_gpu_profiles.jsonl
```

Then evaluate with a device-holdout split:

```bash
conda run -n sglang python scripts/run_phase3_validation.py --data artifacts/multi_gpu_profiles.jsonl --split-mode device-holdout
```

## Recommended Real Collection Commands

Use `CUDA_VISIBLE_DEVICES` to pin one GPU per run. The commands below collect a broader reproducible GEMM/BMM dataset covering `gemm,bmm`, `fp16,bf16,fp32`, `small,medium,large`, aligned and non-aligned shapes, and BMM batches `1/4/16/64`.

RTX 4090:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n sglang python scripts/collect_phase3_gemm_data.py \
  --output artifacts/phase3_real_RTX_4090.jsonl \
  --format jsonl \
  --mode torch \
  --families gemm,bmm \
  --dtypes fp16,bf16,fp32 \
  --sizes small,medium,large \
  --warmup 10 \
  --repeats 20 \
  --seed 7 \
  --rounds 3
```

A40:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n sglang python scripts/collect_phase3_gemm_data.py \
  --output artifacts/phase3_real_A40.jsonl \
  --format jsonl \
  --mode torch \
  --families gemm,bmm \
  --dtypes fp16,bf16,fp32 \
  --sizes small,medium,large \
  --warmup 15 \
  --repeats 30 \
  --seed 7 \
  --rounds 3
```

L40S:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n sglang python scripts/collect_phase3_gemm_data.py \
  --output artifacts/phase3_real_L40S.jsonl \
  --format jsonl \
  --mode torch \
  --families gemm,bmm \
  --dtypes fp16,bf16,fp32 \
  --sizes small,medium,large \
  --warmup 15 \
  --repeats 30 \
  --seed 7 \
  --rounds 3
```

Train a Ridge residual model from one collected dataset:

```bash
conda run -n sglang python scripts/train_phase3_residual.py \
  --data artifacts/phase3_real_RTX_4090.jsonl \
  --output artifacts/phase3_real_RTX_4090.joblib
```

Evaluate with a random split:

```bash
conda run -n sglang python scripts/eval_phase3_residual.py \
  --data artifacts/phase3_real_RTX_4090.jsonl \
  --split-mode random
```

Evaluate a merged multi-GPU dataset with device holdout:

```bash
cat artifacts/phase3_real_RTX_4090.jsonl artifacts/phase3_real_A40.jsonl artifacts/phase3_real_L40S.jsonl > artifacts/phase3_real_multi_gpu.jsonl

conda run -n sglang python scripts/eval_phase3_residual.py \
  --data artifacts/phase3_real_multi_gpu.jsonl \
  --split-mode device-holdout
```
