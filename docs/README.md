# Documentation

<p align="center">
  <a href="https://github.com/vlang/vtl">VTL</a> · <a href="https://github.com/vlang/vsl">VSL</a>
</p>

---

## VTL — V Tensor Library

VTL is a pure-V tensor library for numerical computing and machine learning.
It provides n-dimensional arrays, automatic differentiation (autograd), and a full
neural network module.

| Document | Description |
|----------|-------------|
| [TUTORIAL.md](./TUTORIAL.md) | General overview and getting started |
| [TUTORIAL_FIRST_STEPS.md](./TUTORIAL_FIRST_STEPS.md) | Tensor creation, indexing, slicing |
| [TUTORIAL_MAP_REDUCE.md](./TUTORIAL_MAP_REDUCE.md) | Element-wise `map` / `nmap` and reductions |
| [TUTORIAL_AUTOGRAD.md](./TUTORIAL_AUTOGRAD.md) | Autograd: `Variable`, gates, backprop |
| [TUTORIAL_NEURAL_NETWORKS.md](./TUTORIAL_NEURAL_NETWORKS.md) | Layers, losses, optimizers, `Sequential` |
| [TUTORIAL_LINEAR_ALGEBRA.md](./TUTORIAL_LINEAR_ALGEBRA.md) | LA basics via VSL |
| [TUTORIAL_BROADCASTING.md](./TUTORIAL_BROADCASTING.md) | Broadcasting rules |
| [TUTORIAL_SLICING.md](./TUTORIAL_SLICING.md) | Slicing and views |
| [TUTORIAL_REDUCTIONS.md](./TUTORIAL_REDUCTIONS.md) | argmax/argmin/cumsum/cumprod |
| [TUTORIAL_ADVANCED_LA.md](./TUTORIAL_ADVANCED_LA.md) | trace/norm/outer/qr/lu/cholesky/pinv |
| [TUTORIAL_OPTIMIZERS.md](./TUTORIAL_OPTIMIZERS.md) | Adam/AdamW/RMSProp/AdaGrad/SGD + schedulers |

## VSL — V Standard Library

VSL is the pure-V linear algebra companion of VTL. It wraps LAPACK via `go-vlang/vsl`
and provides matrix operations, decompositions, and random number generation.

> **VTL automatically uses VSL as its LA backend.** There is usually no need to
> install or import VSL directly when working with VTL.

| Document | Description |
|----------|-------------|
| VSL README | [vlang/vsl](https://github.com/vlang/vsl) |
| VSL docs | [vlang.github.io/vsl](https://vlang.github.io/vsl) |
