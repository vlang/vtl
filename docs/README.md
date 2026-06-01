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
| Broadcasting, maps, reductions | [Broadcasting](./TUTORIAL_BROADCASTING.md), [Map/reduce](./TUTORIAL_MAP_REDUCE.md), [Reductions](./TUTORIAL_REDUCTIONS.md) |
| Linear algebra | [Linear algebra](./TUTORIAL_LINEAR_ALGEBRA.md), [Advanced LA](./TUTORIAL_ADVANCED_LA.md) |
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
6. [Matrix and Vector Operations](./TUTORIAL_LINEAR_ALGEBRA.md)
7. [Advanced Linear Algebra](./TUTORIAL_ADVANCED_LA.md)
8. [Automatic Differentiation](./TUTORIAL_AUTOGRAD.md)
9. [Neural Networks](./TUTORIAL_NEURAL_NETWORKS.md)
10. [Optimizers](./TUTORIAL_OPTIMIZERS.md)

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
