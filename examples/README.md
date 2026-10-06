# VTL Examples

Run examples from `~/.vmodules` unless a README says otherwise:

```bash
v run ./vtl/examples/nn_xor/main.v
```

Some examples download datasets or require GPU build flags. Use the synthetic
CIFAR examples for CI-safe checks.

## Tensor and autograd basics

| Example | What it shows | Command |
|---------|---------------|---------|
| [tensor_axis_manipulation](./tensor_axis_manipulation) | NumPy-style `moveaxis` and `rollaxis` views | `v run ./vtl/examples/tensor_axis_manipulation/main.v` |
| [vtl_basic_usage](./vtl_basic_usage) | Tensor creation and basic operations | `v run ./vtl/examples/vtl_basic_usage/main.v` |
| [vtl_vandermont](./vtl_vandermont) | Matrix construction and LA utilities | `v run ./vtl/examples/vtl_vandermont/main.v` |
| [autograd_backprop](./autograd_backprop) | Manual autograd/backprop flow | `v run ./vtl/examples/autograd_backprop/main.v` |
| [npy_round_trip](./npy_round_trip) | Read and write NumPy `.npy` arrays | `v run ./vtl/examples/npy_round_trip/main.v` |
| [csv_round_trip](./csv_round_trip) | Read and write numeric CSV tensors with headers | `v run ./vtl/examples/csv_round_trip/main.v` |
| [npz_round_trip](./npz_round_trip) | Write and read named arrays in `.npz` archives | `v run ./vtl/examples/npz_round_trip/main.v` |
| [npz_read_compressed](./npz_read_compressed) | Read an f64 member from a NumPy-generated compressed `.npz` archive | `v run ./vtl/examples/npz_read_compressed/main.v ./vtl/npz/testdata/numpy_compressed.npz` |
| [stats_variance](./stats_variance) | Stable population/sample variance and standard deviation | `v run ./vtl/examples/stats_variance/main.v` |
| [stats_nan_reductions](./stats_nan_reductions) | NaN-aware sums, products, extrema, and axis reductions | `v run ./vtl/examples/stats_nan_reductions/main.v` |
| [random_seed](./random_seed) | Repeat a random tensor sequence with an explicit seed | `v run ./vtl/examples/random_seed/main.v` |
| [scatter](./scatter) | Add or assign values at indexed tensor positions | `v run ./vtl/examples/scatter/main.v` |
| [argwhere](./argwhere) | Find non-zero coordinates in a tensor | `v run ./vtl/examples/argwhere/main.v` |
| [meshgrid_n](./meshgrid_n) | Build N-dimensional coordinate grids with `xy` or `ij` indexing | `v run ./vtl/examples/meshgrid_n/main.v` |
| [diff](./diff) | Compute discrete differences in one-dimensional and multidimensional data | `v run ./vtl/examples/diff/main.v` |
| [trapezoid](./trapezoid) | Integrate sampled data with the composite trapezoidal rule | `v run ./vtl/examples/trapezoid/main.v` |
| [broadcast_mask](./broadcast_mask) | Select and fill values with broadcastable boolean masks | `v run ./vtl/examples/broadcast_mask/main.v` |
| [vector_norm](./vector_norm) | Compute stable p-norms globally and along an axis | `v run ./vtl/examples/vector_norm/main.v` |
| [fft_frequency](./fft_frequency) | Create FFT frequency bins and center a spectrum | `v run ./vtl/examples/fft_frequency/main.v` |
| [covariance_correlation](./covariance_correlation) | Compute sample covariance and Pearson correlation matrices | `v run ./vtl/examples/covariance_correlation/main.v` |
| [histogram](./histogram) | Count samples and compute weighted density with custom bin edges | `v run ./vtl/examples/histogram/main.v` |
| [bincount](./bincount) | Count integer labels and sum per-label weights | `v run ./vtl/examples/bincount/main.v` |
| [unique](./unique) | Find sorted unique values and count occurrences | `v run ./vtl/examples/unique/main.v` |

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
| [vtl_opencl_vcl_support](./vtl_opencl_vcl_support) | OpenCL VTL tensor transfer and VCL compute smoke | `v -d vcl run vtl/examples/vtl_opencl_vcl_support/main.v` |

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
VJOBS=1 v test ./vtl/nn/f32_training_smoke_test.v
VTL_USE_VULKAN=1 VJOBS=1 v -prod -d vulkan test vtl/nn/f32_vulkan_training_smoke_d_vulkan_test.v
```

See [DEV_LIGHTWEIGHT.md](../docs/DEV_LIGHTWEIGHT.md) for the full safe command
matrix.
