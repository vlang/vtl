# Tutorial

This is the recommended learning path for VTL. It starts with tensors, then
builds toward autograd, neural networks, optimizers, and GPU-backed examples.

## Tensor fundamentals

1. [First Steps](./TUTORIAL_FIRST_STEPS.md) — creation, indexing, shapes.
2. [Slicing](./TUTORIAL_SLICING.md) — views and sub-tensors.
3. [Broadcasting](./TUTORIAL_BROADCASTING.md) — shape-compatible operations.
4. [Map and Reduce](./TUTORIAL_MAP_REDUCE.md) — element-wise transforms and reductions.
5. [Reductions](./TUTORIAL_REDUCTIONS.md) — argmax, argmin, cumulative operations.

## Linear algebra

6. [Matrix and Vector operations](./TUTORIAL_LINEAR_ALGEBRA.md) — VSL-backed LA.
7. [Advanced Linear Algebra](./TUTORIAL_ADVANCED_LA.md) — QR, LU, Cholesky, pinv.
8. [Fourier transforms](./TUTORIAL_FFT.md) — one-dimensional real FFTs.

## Machine learning

9. [Automatic Differentiation](./TUTORIAL_AUTOGRAD.md) — `Variable`, gates, backprop.
10. [Neural Networks](./TUTORIAL_NEURAL_NETWORKS.md) — layers, losses, `Sequential`.
11. [Optimizers](./TUTORIAL_OPTIMIZERS.md) — SGD, Adam, AdamW, schedulers.

## Next

- [Examples catalog](../examples/README.md)
- [Datasets](../datasets/README.md)
- [Device memory and GPU paths](./DEVICE_MEMORY.md)
- [Safe local development commands](./DEV_LIGHTWEIGHT.md)
