# TASKS.md

## Phase 1
Scaffold the repository for the 7-module GPU operator latency prediction system.
Create:
- package directories
- interface files
- placeholder implementations
- test skeletons
- top-level README structure

Do not implement full logic yet.

## Phase 2
Implement GEMM/BMM analytical baseline only.
Required:
- kernel bucket recognition for GEMM/BMM
- task decomposition for GEMM/BMM
- wave estimation
- analytical baseline latency estimation
- tests
- one demo script

No learned model yet.

## Phase 3
Add a lightweight learned residual head for GEMM/BMM.
Required:
- dataset schema
- dataset builder
- training script
- evaluation script
- latency mean prediction
- tests
- one runnable demo

## Phase 4
Extend the same structure to:
- Attention
- Vector/Fused vector
Reuse public interfaces whenever possible.
Add tests and update README.

## Phase 5
Add:
- uncertainty or p90 prediction
- simple operator/model aggregation
- end-to-end demo on a toy kernel sequence

## General execution rule
For every phase:
1. read AGENTS.md and TASKS.md first
2. implement only the requested phase
3. run tests
4. run demo
5. update README
6. summarize implemented vs remaining work