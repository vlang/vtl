# VTL Documentation

<p align="center">
  <a href="https://github.com/vlang/vtl">VTL</a> · <a href="https://github.com/vlang/vsl">VSL</a>
</p>

---

## VTL — V Tensor Library

VTL is a pure-V tensor library for numerical computing and machine learning.
It provides n-dimensional arrays, automatic differentiation (autograd), and a full
neural network module.

## Start Here

| Goal | Read |
|------|------|
| New to VTL | [Tutorial overview](./TUTORIAL.md) |
| Tensor creation, indexing, slicing | [First steps](./TUTORIAL_FIRST_STEPS.md), [Slicing](./TUTORIAL_SLICING.md) |
| Unique values and occurrence counts | [Unique values](./TUTORIAL_UNIQUE.md) |
| Gather and indexed updates | [Indexing and scatter](./TUTORIAL_INDEXING.md) |
| Broadcasting, maps, reductions | [Broadcasting](./TUTORIAL_BROADCASTING.md), [Map/reduce](./TUTORIAL_MAP_REDUCE.md), [Reductions](./TUTORIAL_REDUCTIONS.md) |
| Random tensors and reproducibility | [Random numbers](./TUTORIAL_RANDOM.md) |
| Fourier transforms | [FFT and normalization modes](./TUTORIAL_FFT.md) |
| NumPy file interchange | [`.npy` input/output](./TUTORIAL_NUMPY_IO.md) |
| Linear algebra and norms | [Linear algebra](./TUTORIAL_LINEAR_ALGEBRA.md), [Advanced LA](./TUTORIAL_ADVANCED_LA.md) |
| Autograd | [Autograd](./TUTORIAL_AUTOGRAD.md) |
| Neural networks | [Neural networks](./TUTORIAL_NEURAL_NETWORKS.md), [Optimizers](./TUTORIAL_OPTIMIZERS.md) |
| Datasets and examples | [Datasets](../datasets/README.md), [Examples catalog](../examples/README.md) |
| GPU/dev workflow | [Device memory](./DEVICE_MEMORY.md), [Lightweight development](./DEV_LIGHTWEIGHT.md) |
| Release status | [ML Roadmap](./ML_ROADMAP.md), [Project roadmap](../ROADMAP.md) |

## Learning Path

1. [First Steps](./TUTORIAL_FIRST_STEPS.md)
2. [Slicing](./TUTORIAL_SLICING.md)
3. [Broadcasting](./TUTORIAL_BROADCASTING.md)
4. [Map and Reduce](./TUTORIAL_MAP_REDUCE.md)
5. [Reductions](./TUTORIAL_REDUCTIONS.md)
6. [Fourier Transforms](./TUTORIAL_FFT.md)
7. [Matrix and Vector Operations](./TUTORIAL_LINEAR_ALGEBRA.md)
8. [Advanced Linear Algebra](./TUTORIAL_ADVANCED_LA.md)
9. [Automatic Differentiation](./TUTORIAL_AUTOGRAD.md)
10. [Neural Networks](./TUTORIAL_NEURAL_NETWORKS.md)
11. [Optimizers](./TUTORIAL_OPTIMIZERS.md)

## ML Release Docs

| Document | Description |
|----------|-------------|
| [ML_ROADMAP.md](./ML_ROADMAP.md) | Current ML release status and open items |
| [DEVICE_MEMORY.md](./DEVICE_MEMORY.md) | CUDA/Vulkan memory and sync model |
| [DEV_LIGHTWEIGHT.md](./DEV_LIGHTWEIGHT.md) | Safe commands for local work and CI |
| [VSL_VTL_CUDA.md](./VSL_VTL_CUDA.md) | CUDA integration notes between VSL and VTL |

## VSL Relationship

VSL is the scientific and GPU compute foundation for VTL. VTL automatically uses
VSL for linear algebra and backend kernels where available. Install/import VSL
directly when you need standalone scientific computing; use VTL when you need
tensors, autograd, datasets, layers, losses, optimizers, and training loops.

| Resource | Link |
|----------|------|
| VSL README | [vlang/vsl](https://github.com/vlang/vsl) |
| VSL docs | [vlang.github.io/vsl](https://vlang.github.io/vsl) |

## Module References

| Area | Reference |
|------|-----------|
| Autograd | [API overview](../autograd/README.md) |
| Linear algebra | [API overview](../la/README.md) |
| Statistics | [API overview](../stats/README.md) |
| Storage backends | [API overview](../storage/README.md) |
| ML metrics | [API overview](../ml/README.md) |
| Neural networks | [Overview](../nn/README.md), [layers](../nn/layers/README.md), [models](../nn/models/README.md), [losses](../nn/loss/README.md), [optimizers](../nn/optimizers/README.md) |
| NN implementation | [gates](../nn/gates/README.md), [types](../nn/types/README.md), [internal kernels](../nn/internal/README.md) |
