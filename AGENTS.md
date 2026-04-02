# AGENTS.md

## Project goal
Build a GPU operator latency prediction system for research prototyping.

The system must be split into exactly 7 modules:
1. graph/kernel extractor
2. kernel implementation recognizer
3. task decomposer
4. scheduling simulator
5. pipeline-aware feature analyzer
6. baseline + learned predictor
7. operator/model aggregator

## v0 scope
Implement only a single-kernel predictor first.
Do NOT implement distributed inference in v0.

Supported kernel families in v0:
- GEMM/BMM
- Attention
- Vector/Fused vector
- RMSNorm/LayerNorm
- Fused MoE

## Architecture rules
Use the following package layout:

predictor/
  extractor/
  recognizer/
  analytical/
  models/
  training/
  serving/

tests/
scripts/

Keep module boundaries clean:
- extractor: parsing graphs, traces, metadata
- recognizer: classify kernel implementation bucket
- analytical: decomposition, scheduling, analytical features, baseline latency
- models: residual head, uncertainty head
- training: dataset builders, trainers, evaluators
- serving: predict kernel latency, aggregate predictions

## Functional requirements
The system must support:
- parsing kernel metadata
- recognizing kernel implementation buckets
- estimating analytical baseline latency
- generating pipeline-aware features
- predicting latency mean
- predicting uncertainty or p90
- exposing a simple CLI or Python entrypoint for demo inference

## Modeling requirements
Use this strategy:
1. identify kernel family and implementation bucket
2. compute analytical/task/pipeline features
3. estimate baseline latency analytically
4. apply a lightweight learned residual correction
5. optionally output uncertainty or p90

Do NOT start with one giant end-to-end neural network.

## Coding rules
- Python 3.10+
- Use type hints
- Use dataclasses where helpful
- Every public function must have a docstring
- No huge monolithic files
- Prefer small composable classes/functions
- Keep files under ~300 lines when practical

## Testing rules
Every phase must add or update tests.
At minimum:
- one unit test per new module
- one integration test for the phase
- one runnable demo script

## Documentation rules
Whenever a phase is completed:
- update README.md
- add a short “Implemented in this phase” section
- add a short “Remaining work” section

## Phase order
Phase 1: scaffold repository and interfaces
Phase 2: implement GEMM/BMM analytical baseline
Phase 3: add GEMM/BMM learned residual head
Phase 4: extend to Attention and Vector kernels
Phase 5: add uncertainty prediction and aggregation

## Definition of done
A phase is complete only if:
- code runs
- tests pass
- demo script runs
- README is updated
- a short summary of implemented vs remaining work is produced