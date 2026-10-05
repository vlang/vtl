# VTL Examples

Run examples from `~/.vmodules` unless a README says otherwise:

```bash
v run vtl/examples/nn_xor/main.v
```

Some examples download datasets or require GPU build flags. Use the synthetic
CIFAR examples for CI-safe checks.

## Tensor and autograd basics

| Example | What it shows | Command |
|---------|---------------|---------|
| [tensor_axis_manipulation](./tensor_axis_manipulation) | NumPy-style `moveaxis` and `rollaxis` views | `v run vtl/examples/tensor_axis_manipulation/main.v` |
| [vtl_basic_usage](./vtl_basic_usage) | Tensor creation and basic operations | `v run vtl/examples/vtl_basic_usage/main.v` |
| [vtl_vandermont](./vtl_vandermont) | Matrix construction and LA utilities | `v run vtl/examples/vtl_vandermont/main.v` |
| [autograd_backprop](./autograd_backprop) | Manual autograd/backprop flow | `v run vtl/examples/autograd_backprop/main.v` |
| [npy_round_trip](./npy_round_trip) | Read and write NumPy `.npy` arrays | `v run vtl/examples/npy_round_trip/main.v` |
| [stats_variance](./stats_variance) | Stable population/sample variance and standard deviation | `v run vtl/examples/stats_variance/main.v` |

## Neural networks

| Example | What it shows | Notes |
|---------|---------------|-------|
| [nn_xor](./nn_xor) | Small XOR classifier | Fast CPU smoke |
| [nn_simple_two_layer](./nn_simple_two_layer) | Basic MLP | Fast CPU smoke |
| [nn_regression_sine](./nn_regression_sine) | Regression with synthetic data | CPU |
| [nn_multiclass_iris](./nn_multiclass_iris) | Multiclass classifier | CPU |
| [nn_autoencoder_simple](./nn_autoencoder_simple) | Simple autoencoder | CPU |
| [nn_conv1d](./nn_conv1d) | Conv1D sequence forward and backward | CPU |
| [nn_gru](./nn_gru) | GRU sequence forward and autograd backward | CPU |
| [nn_mnist](./nn_mnist) | MNIST training path | Dataset download/cache |

## CIFAR-10 release examples

| Example | Purpose | Recommended use |
|---------|---------|-----------------|
| [nn_cifar10_tiny_synth](./nn_cifar10_tiny_synth) | Synthetic f64 CIFAR-shaped smoke | CI/default |
| [nn_cifar10_f32_tiny_synth](./nn_cifar10_f32_tiny_synth) | Synthetic f32 training smoke | CI/default |
| [nn_cifar10_safe](./nn_cifar10_safe) | Safer real-data config | Local |
| [nn_cifar10_tiny](./nn_cifar10_tiny) | Tiny real CIFAR subset | Local |
| [nn_cifar10](./nn_cifar10) | Full CIFAR path with checkpoints | Local/high RAM |

## GPU examples

| Example | Backend | Command |
|---------|---------|---------|
| [nn_cifar10_cuda](./nn_cifar10_cuda) | CUDA/cuBLAS/cuDNN via VSL | `VTL_USE_CUDA=1 v -d cuda run vtl/examples/nn_cifar10_cuda/main.v` |
| [nn_cifar10_vulkan](./nn_cifar10_vulkan) | Vulkan f32 Linear/Conv2D/ReLU/Adam via VSL | `VTL_USE_VULKAN=1 v -prod -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v` |
| [nn_cifar10_f32_vulkan_tiny_synth](./nn_cifar10_f32_vulkan_tiny_synth) | f32 Vulkan-shaped tiny smoke | `VTL_USE_VULKAN=1 v -prod -d vulkan run vtl/examples/nn_cifar10_f32_vulkan_tiny_synth/main.v` |
| [vtl_opencl_vcl_support](./vtl_opencl_vcl_support) | OpenCL/VCL support notes | See example README |

## Datasets and plotting

| Example | What it shows |
|---------|---------------|
| [datasets_mnist](./datasets_mnist) | MNIST loader shape smoke |
| [datasets_imdb](./datasets_imdb) | IMDB loader shape smoke |
| [vtl_plot_scatter_colorscale](./vtl_plot_scatter_colorscale) | VTL tensor data feeding VSL plot |
| [stats_quantile](./stats_quantile) | Linearly interpolated quantiles |

## Safe validation

Prefer scoped commands:

```bash
VJOBS=1 v test vtl/nn/f32_training_smoke_test.v
VTL_USE_VULKAN=1 VJOBS=1 v -prod -d vulkan test vtl/nn/f32_vulkan_training_smoke_d_vulkan_test.v
```

See [DEV_LIGHTWEIGHT.md](../docs/DEV_LIGHTWEIGHT.md) for the full safe command
matrix.
