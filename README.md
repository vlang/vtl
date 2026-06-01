<div align="center">
  <p>
    <img
      style="width: 200px"
      width="200"
      src="https://github.com/vlang/vtl/blob/main/static/vtl-logo.png?sanitize=true&raw=true"
    >
  </p>
  <h1>The V Tensor Library</h1>

[![Mentioned in Awesome V][awesomevbadge]][awesomevurl]
[![CI][workflowbadge]][workflowurl]
[![Docs][docsbadge]][docsurl]
[![Full ML][fullmlbadge]][fullmlurl]
[![Benchmarks][benchmarksbadge]][benchmarksurl]
[![License: MIT][licensebadge]][licenseurl]
![VSL Backed][vslbackedbadge]
![CUDA Optional][cudaoptionalbadge]
![Vulkan f32][vulkanbadge]

**VTL** is a pure-[V](https://vlang.io) tensor library for numerical computing
and machine learning — n-dimensional arrays, autograd, linear algebra via
[VSL](https://github.com/vlang/vsl), and a full neural network module.

Train small neural networks, experiment with autograd, and use VSL-backed CPU,
CUDA, and Vulkan compute paths from one V-native API.

[vlang.io](https://vlang.io) |
[Docs](https://vlang.github.io/vtl) |
[Tutorials](docs/TUTORIAL.md) |
[ML Roadmap](docs/ML_ROADMAP.md) |
[Contributing](CONTRIBUTING.md) |
[VSL](https://github.com/vlang/vsl)

</div>

```v ignore
import vtl
t := vtl.from_array([1.0, 2, 3, 4], [2, 2])!
t.get([1, 1])
// 4.0
```

## Features

- **Tensors** — create, slice, reshape, transpose, broadcast, map/reduce
- **Autograd** — reverse-mode AD; arbitrary computational graphs
- **Neural networks** — `Sequential` API; Linear, Conv2D, LSTM, Attention, …
- **Losses & optimizers** — MSE, BCE, CrossEntropy, Huber; Adam, AdamW, SGD, …
- **Linear algebra** — VSL-backed matmul, solve, QR, LU, Cholesky, SVD, pinv
- **Hardware** — zero-copy `Tensor.data` for C libs; optional CUDA and Vulkan training paths

## ML Release Highlights

The ML beta scope is the high-level VTL API: tensors, autograd, layers, losses,
optimizers, datasets, and CPU training. CUDA and Vulkan paths are available for
opt-in validation and early adopters, but remain experimental backend
accelerators rather than stable user contracts.

| Area | Status |
|------|--------|
| f32 training | `Sequential` + MSE + Adam smoke tests |
| CUDA | Experimental opt-in Linear/Conv2D forward, CUDA backward, activation chain, Adam slots |
| Vulkan | Experimental opt-in f32 Linear, Conv2D same-padding, ReLU/Sigmoid, fused Adam shader |
| Datasets | MNIST, IMDB, CIFAR-10 loaders plus CI-safe synthetic examples |
| Benchmarks | VTL vs NumPy/PyTorch scripts and PR benchmark workflow |

For memory-safe local commands, see [DEV_LIGHTWEIGHT.md](docs/DEV_LIGHTWEIGHT.md).

## Quick start

```v ignore
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers

mut ctx := autograd.ctx[f32]()
mut model := models.sequential_from_ctx[f32](ctx)
model.input([784])
model.linear(256)
model.linear(10)
model.mse_loss()

input_tensor := vtl.zeros[f32]([64, 784])
mut x := ctx.variable(input_tensor)
y_pred := model.forward(x)!

target := vtl.zeros[f32]([64, 10])
mut loss_val := model.loss(y_pred, target)!
loss_val.backprop()!

mut opt := optimizers.adam_optimizer[f32](optimizers.AdamOptimizerConfig{
	learning_rate: 0.001
})
opt.build_params(model.info.layers)
opt.update()!
```

## Module overview

| Module | Purpose |
|--------|---------|
| `vtl` | Core `Tensor[T]`; creation, slicing, broadcasting |
| `vtl.autograd` | `Context`, `Variable`, gates, `backprop()` |
| `vtl.la` | Linear algebra (wraps VSL) |
| `vtl.nn` | Layers, losses, optimizers |
| `vtl.nn.models` | `Sequential` model API |
| `vtl.nn.internal` | Weight init (Kaiming, Xavier) |
| `vtl.nn.gates` | Autograd gate implementations |

## Installation

VTL uses [VSL](https://github.com/vlang/vsl) for linear algebra. The core `vtl`
module works without optional system BLAS/LAPACK, but LA features need VSL.

Follow [VSL install instructions](https://github.com/vlang/vsl#install-vsl-locally),
then:

```sh
v install vtl
```

## Testing

```sh
v test ~/.vmodules/vtl
```

See [DEV_LIGHTWEIGHT.md](docs/DEV_LIGHTWEIGHT.md) for memory-safe subsets in CI.

## Documentation

### Start Here

| Goal | Read |
|------|------|
| Learn tensors | [First steps](docs/TUTORIAL_FIRST_STEPS.md) |
| Learn autograd | [Autograd tutorial](docs/TUTORIAL_AUTOGRAD.md) |
| Build neural networks | [Neural networks](docs/TUTORIAL_NEURAL_NETWORKS.md) |
| Pick optimizers | [Optimizers](docs/TUTORIAL_OPTIMIZERS.md) |
| Run examples | [Examples catalog](examples/README.md) |
| Use datasets | [Datasets](datasets/README.md) |
| Use GPU paths safely | [DEV_LIGHTWEIGHT.md](docs/DEV_LIGHTWEIGHT.md), [DEVICE_MEMORY.md](docs/DEVICE_MEMORY.md) |

| Tutorial | Topic |
|----------|-------|
| [TUTORIAL_FIRST_STEPS.md](docs/TUTORIAL_FIRST_STEPS.md) | Tensor creation, indexing, slicing |
| [TUTORIAL_MAP_REDUCE.md](docs/TUTORIAL_MAP_REDUCE.md) | `map` / `nmap` and reductions |
| [TUTORIAL_AUTOGRAD.md](docs/TUTORIAL_AUTOGRAD.md) | `Variable`, gates, backprop |
| [TUTORIAL_REDUCTIONS.md](docs/TUTORIAL_REDUCTIONS.md) | argmax / argmin / cumsum |
| [TUTORIAL_NEURAL_NETWORKS.md](docs/TUTORIAL_NEURAL_NETWORKS.md) | Layers, losses, `Sequential` |
| [TUTORIAL_OPTIMIZERS.md](docs/TUTORIAL_OPTIMIZERS.md) | Adam, AdamW, RMSProp, schedulers |
| [TUTORIAL_LINEAR_ALGEBRA.md](docs/TUTORIAL_LINEAR_ALGEBRA.md) | LA basics via VSL |
| [TUTORIAL_ADVANCED_LA.md](docs/TUTORIAL_ADVANCED_LA.md) | QR, LU, Cholesky, pinv |
| [TUTORIAL_BROADCASTING.md](docs/TUTORIAL_BROADCASTING.md) | Broadcasting rules |
| [TUTORIAL_SLICING.md](docs/TUTORIAL_SLICING.md) | Slicing and views |

Full index: [`docs/README.md`](docs/README.md).

## Contributors

> Originally based on work by
> [christopherzimmerman](https://github.com/christopherzimmerman).
> The core was reimplemented while keeping that lineage and inspiration.

<a href="https://github.com/vlang/vtl/contributors">
  <img src="https://contrib.rocks/image?repo=vlang/vtl"/>
</a>

Made with [contributors-img](https://contrib.rocks).

## License

[MIT](LICENSE)

[awesomevbadge]: https://awesome.re/mentioned-badge.svg
[workflowbadge]: https://github.com/vlang/vtl/actions/workflows/ci.yml/badge.svg
[docsbadge]: https://github.com/vlang/vtl/actions/workflows/deploy-docs.yml/badge.svg
[fullmlbadge]: https://github.com/vlang/vtl/actions/workflows/ci-full-ml.yml/badge.svg
[benchmarksbadge]: https://github.com/vlang/vtl/actions/workflows/benchmark-pr-comment.yml/badge.svg
[licensebadge]: https://img.shields.io/badge/License-MIT-blue.svg
[vslbackedbadge]: https://img.shields.io/badge/VSL-backed-027d9c?logo=v
[cudaoptionalbadge]: https://img.shields.io/badge/CUDA-optional-76b900?logo=nvidia
[vulkanbadge]: https://img.shields.io/badge/Vulkan-f32-ac162c?logo=vulkan
[awesomevurl]: https://github.com/vlang/awesome-v/blob/master/README.md#scientific-computing
[workflowurl]: https://github.com/vlang/vtl/actions/workflows/ci.yml
[docsurl]: https://github.com/vlang/vtl/actions/workflows/deploy-docs.yml
[fullmlurl]: https://github.com/vlang/vtl/actions/workflows/ci-full-ml.yml
[benchmarksurl]: https://github.com/vlang/vtl/actions/workflows/benchmark-pr-comment.yml
[licenseurl]: https://github.com/vlang/vtl/blob/main/LICENSE
