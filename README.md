# The V Tensor Library

<div align="center">

[![Mentioned in Awesome V][awesomevbadge]][awesomevurl]
[![CI][workflowbadge]][workflowurl]
[![License: MIT][licensebadge]][licenseurl]

**VTL** is a pure-[V](https://vlang.io) tensor library for numerical computing
and machine learning. It provides n-dimensional arrays with automatic
differentiation, linear algebra via [VSL](https://github.com/vlang/vsl),
and a full neural network module.

[Documentation](docs/README.md) ·
[Tutorials](docs/TUTORIAL.md) ·
[VSL Backend](https://github.com/vlang/vsl)

</div>

## Features

- **Tensor operations** — create, slice, reshape, transpose, broadcast, map/reduce
- **Autograd** — reverse-mode automatic differentiation; build arbitrary computational graphs
- **Neural networks** — `Sequential` model API; layers (Linear, Conv2D, LSTM, Attention, …);
  losses (MSE, BCE, CrossEntropy, Huber, …); optimizers (Adam, AdamW, RMSProp, AdaGrad, SGD)
- **Linear algebra** — VSL-backed: matmul, solve, lstsq, qr, lu, cholesky, pinv, trace, norm, SVD
- **Hardware acceleration** — zero-copy sharing with C libraries via `vtl.Tensor.data`
  plus Vulkan (`-d vulkan`) and OpenCL/VCL (`-d vcl`) backend paths

## Quick start

```v ignore
import vtl
import vtl.autograd as ag
import vtl.nn.layers
import vtl.nn.loss
import vtl.nn.optimizers

// 1. Build a two-layer network
mut ctx := ag.ctx[f64]()
lin1 := layers.linear_layer[f64](ctx, 784, 256)
lin2 := layers.linear_layer[f64](ctx, 256, 10)
model := [layers.Layer[f64](lin1), layers.Layer[f64](lin2)]

// 2. Forward pass
input_tensor := vtl.zeros[f64]([64, 784])
mut x := ctx.variable(input_tensor)
for layer in model {
	x = layer.forward(x)!
}

// 3. Loss
y_pred := ctx.variable(vtl.zeros[f64]([64, 10]))
target := vtl.zeros[f64]([64, 10])
mut l := loss.mse_loss[f64]()
mut loss_val := l.loss(y_pred, target)!

// 4. Backward + update
loss_val.backward()!
mut opt := optimizers.adam_optimizer[f64](learning_rate: 0.001)
opt.build_params(model)
opt.update()!
```

## Module overview

| Module | Purpose |
|--------|---------|
| `vtl` | Core `Tensor[T]` type; creation, slicing, broadcasting |
| `vtl.autograd` | `Context`, `Variable`, gates, `backprop()` |
| `vtl.la` | Linear algebra (wraps VSL) |
| `vtl.nn` | Neural network layers, losses, optimizers |
| `vtl.nn.models` | `Sequential` model API |
| `vtl.nn.internal` | Weight initialisation (Kaiming, Xavier) |
| `vtl.nn.gates` | Autograd gate implementations |

## Installation

VTL depends on [VSL](https://github.com/vlang/vsl). Follow the
[VSL install instructions](https://github.com/vlang/vsl#install-vsl-locally)
before installing VTL.

```sh
v install vtl
```

## Testing

```sh
v test ~/.vmodules/vtl
```

## Documentation

All tutorials live in [`docs/`](docs/README.md):

| Tutorial | Topic |
|----------|-------|
| [TUTORIAL_FIRST_STEPS.md](docs/TUTORIAL_FIRST_STEPS.md) | Tensor creation, indexing, slicing |
| [TUTORIAL_MAP_REDUCE.md](docs/TUTORIAL_MAP_REDUCE.md) | `map` / `nmap` and reductions |
| [TUTORIAL_AUTOGRAD.md](docs/TUTORIAL_AUTOGRAD.md) | Autograd: `Variable`, gates, backprop |
| [TUTORIAL_REDUCTIONS.md](docs/TUTORIAL_REDUCTIONS.md) | argmax / argmin / cumsum / cumprod |
| [TUTORIAL_NEURAL_NETWORKS.md](docs/TUTORIAL_NEURAL_NETWORKS.md) | Layers, losses, optimizers, `Sequential` |
| [TUTORIAL_OPTIMIZERS.md](docs/TUTORIAL_OPTIMIZERS.md) | Adam / AdamW / RMSProp / AdaGrad / SGD + schedulers |
| [TUTORIAL_LINEAR_ALGEBRA.md](docs/TUTORIAL_LINEAR_ALGEBRA.md) | LA basics via VSL |
| [TUTORIAL_ADVANCED_LA.md](docs/TUTORIAL_ADVANCED_LA.md) | trace / norm / qr / lu / cholesky / pinv |
| [TUTORIAL_BROADCASTING.md](docs/TUTORIAL_BROADCASTING.md) | Broadcasting rules |
| [TUTORIAL_SLICING.md](docs/TUTORIAL_SLICING.md) | Slicing and views |
| [TUTORIAL_GPU_BACKENDS.md](docs/TUTORIAL_GPU_BACKENDS.md) | Vulkan + OpenCL/VCL backend guide |

## Benchmarking

Backend comparison benchmark is available in [`benchmark/`](benchmark/README.md).

```sh
v run benchmark/main.v
v -d vcl run benchmark/main.v
v -d vulkan run benchmark/main.v
v -d vulkan -d vcl run benchmark/main.v
```

## License

[MIT](LICENSE)

[awesomevbadge]: https://awesome.re/mentioned-badge.svg
[workflowbadge]: https://github.com/vlang/vtl/actions/workflows/ci.yml/badge.svg
[licensebadge]: https://img.shields.io/badge/License-MIT-blue.svg
[awesomevurl]: https://github.com/vlang/awesome-v/blob/master/README.md#scientific-computing
[workflowurl]: https://github.com/vlang/vtl/actions/workflows/ci.yml
[licenseurl]: https://github.com/vlang/vtl/blob/main/LICENSE
