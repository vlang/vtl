# `vtl.nn`

Neural-network building blocks are split into focused modules:

| Module | Purpose |
| --- | --- |
| [`layers`](layers/README.md) | Trainable and activation layers |
| [`models`](models/README.md) | Sequential model composition |
| [`loss`](loss/README.md) | Differentiable objective functions |
| [`optimizers`](optimizers/README.md) | Parameter update algorithms |
| [`gates`](gates/README.md) | Backward rules used by layers and losses |
| [`types`](types/README.md) | Type-erased layer/loss/optimizer adapters |
| [`internal`](internal/README.md) | Implementation helpers, not a stable application API |

The high-level learning path is the [neural-network tutorial](../docs/TUTORIAL_NEURAL_NETWORKS.md),
followed by the [optimizer tutorial](../docs/TUTORIAL_OPTIMIZERS.md). Accelerator
support is experimental and depends on the V build flags and host toolchain.
