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
| [TUTORIAL_GPU_BACKENDS.md](./TUTORIAL_GPU_BACKENDS.md) | Vulkan/OpenCL (VCL) backend usage and benchmarking |

## VSL — V Scientific Library

VSL is a comprehensive scientific computing library for V.  It provides linear
algebra (BLAS/LAPACK), statistics, polynomials, derivatives, noise generators,
plotting (`vsl.plot`), and more.

> **VTL uses VSL as its LA and plotting backend.** You generally don't need to
> import VSL directly when using VTL's tensor or neural network modules, but
> you will import `vsl.plot` for visualization.

| Document | Description |
|----------|-------------|
| VSL README | [vlang/vsl](https://github.com/vlang/vsl) |
| VSL docs | [vlang.github.io/vsl](https://vlang.github.io/vsl) |
