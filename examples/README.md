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
| [pad](./pad) | Constant, edge, wrap, reflect, and symmetric tensor padding | `v run ./vtl/examples/pad/main.v` |
| [diag](./diag) | Construct and extract offset diagonals, including N-D diagonal views | `v run ./vtl/examples/diag/main.v` |
| [kron](./kron) | Build NumPy-style Kronecker products from tensors of any rank | `v run ./vtl/examples/kron/main.v` |
| [vander](./vander) | Build polynomial features from powers of a vector | `v run ./vtl/examples/vander/main.v` |
| [complex_tensors](./complex_tensors) | Create `complex128` tensors and use elementwise arithmetic | `v run ./vtl/examples/complex_tensors/main.v` |
| [einsum_complex](./einsum_complex) | Contract complex tensors with Einstein summation | `v run ./vtl/examples/einsum_complex/main.v` |
| [tensor_axis_manipulation](./tensor_axis_manipulation) | `moveaxis` and `rollaxis` views | `v run ./vtl/examples/tensor_axis_manipulation/main.v` |
| [tensor_sorting](./tensor_sorting) | Sort, partially partition, and return local indices along tensor axes | `v run ./vtl/examples/tensor_sorting/main.v` |
| [vtl_basic_usage](./vtl_basic_usage) | Tensor creation and basic operations | `v run ./vtl/examples/vtl_basic_usage/main.v` |
| [vtl_vandermont](./vtl_vandermont) | Matrix construction and LA utilities | `v run ./vtl/examples/vtl_vandermont/main.v` |
| [autograd_backprop](./autograd_backprop) | Manual autograd/backprop flow | `v run ./vtl/examples/autograd_backprop/main.v` |
| [autograd_gather](./autograd_gather) | Gather gradients with repeated-index accumulation | `v run ./vtl/examples/autograd_gather/main.v` |
| [autograd_scatter](./autograd_scatter) | Propagate gradients through indexed additions | `v run ./vtl/examples/autograd_scatter/main.v` |
| [autograd_put](./autograd_put) | Propagate gradients through indexed replacement | `v run ./vtl/examples/autograd_put/main.v` |
| [autograd_slice](./autograd_slice) | Route gradients through slice views | `v run ./vtl/examples/autograd_slice/main.v` |
| [autograd_scalar_loss](./autograd_scalar_loss) | Scalar sum/mean loss and backward pass | `v run ./vtl/examples/autograd_scalar_loss/main.v` |
| [npy_round_trip](./npy_round_trip) | Read and write NumPy `.npy` arrays | `v run ./vtl/examples/npy_round_trip/main.v` |
| [npy_complex128](./npy_complex128) | Read and write NumPy complex128 `.npy` arrays | `v run ./vtl/examples/npy_complex128/main.v` |
| [csv_round_trip](./csv_round_trip) | Read and write numeric CSV tensors with headers | `v run ./vtl/examples/csv_round_trip/main.v` |
| [npz_round_trip](./npz_round_trip) | Write and read named arrays in `.npz` archives | `v run ./vtl/examples/npz_round_trip/main.v` |
| [NPZ fixture](./npz_read_compressed) | Compressed NumPy fixture | `v run vtl/examples/npz_read_compressed/main.v vtl/npz/testdata/numpy_compressed.npz` |
| [stats_variance](./stats_variance) | Stable population/sample variance and standard deviation | `v run ./vtl/examples/stats_variance/main.v` |
| [stats_nan_reductions](./stats_nan_reductions) | NaN-aware sums, products, extrema, and axis reductions | `v run ./vtl/examples/stats_nan_reductions/main.v` |
| [is_nan](./is_nan) | NaN, infinity, and finite-value predicates on tensors and views | `v run ./vtl/examples/is_nan/main.v` |
| [logical_reductions](./logical_reductions) | Axis-wise logical all/any reductions with keepdims | `v run ./vtl/examples/logical_reductions/main.v` |
| [logical_ops](./logical_ops) | Elementwise AND/OR/XOR/NOT with broadcasting | `v run ./vtl/examples/logical_ops/main.v` |
| [multi_axis_extrema](./multi_axis_extrema) | Multi-axis min/max reductions with NumPy-style output shapes | `v run ./vtl/examples/multi_axis_extrema/main.v` |
| [nan_multi_axis_reductions](./nan_multi_axis_reductions) | NaN-aware multi-axis reductions | `v run ./vtl/examples/nan_multi_axis_reductions/main.v` |
| [random_seed](./random_seed) | Repeat a random tensor sequence with an explicit seed | `v run ./vtl/examples/random_seed/main.v` |
| [random_dirichlet](./random_dirichlet) | Seeded Dirichlet probability vectors | `v run ./vtl/examples/random_dirichlet/main.v` |
| [random_generator](./random_generator) | Independent seeded streams and common probability distributions | `v run ./vtl/examples/random_generator/main.v` |
| [scatter](./scatter) | Add or assign values at indexed tensor positions | `v run ./vtl/examples/scatter/main.v` |
| [argwhere](./argwhere) | Find non-zero coordinates in a tensor | `v run ./vtl/examples/argwhere/main.v` |
| [meshgrid_n](./meshgrid_n) | Build N-dimensional coordinate grids with `xy` or `ij` indexing | `v run ./vtl/examples/meshgrid_n/main.v` |
| [indices](./indices) | Build a dense tensor of N-dimensional integer coordinates | `v run ./vtl/examples/indices/main.v` |
| [indices_sparse](./indices_sparse) | Build broadcastable coordinate tensors without a dense grid | `v run ./vtl/examples/indices_sparse/main.v` |
| [triangular_matrices](./triangular_matrices) | Extract lower and upper triangles from batched matrices | `v run ./vtl/examples/triangular_matrices/main.v` |
| [matrix_norm](./matrix_norm) | Compute batched Frobenius, nuclear, and spectral matrix norms | `v run ./vtl/examples/matrix_norm/main.v` |
| [trace_axes](./trace_axes) | Sum diagonals across selected tensor axes | `v run ./vtl/examples/trace_axes/main.v` |
| [svdvals](./svdvals) | Compute descending singular values for batched matrices | `v run ./vtl/examples/svdvals/main.v` |
| [svd](./svd) | Decompose rectangular matrices into U, singular values, and V transpose | `v run ./vtl/examples/svd/main.v` |
| [slogdet](./slogdet) | Compute determinant signs and stable log absolute determinants | `v run ./vtl/examples/slogdet/main.v` |
| [matrix_power](./matrix_power) | Raise each matrix in a batch to an integer exponent | `v run ./vtl/examples/matrix_power/main.v` |
| [condition_number](./condition_number) | Compute matrix condition numbers for stacked matrices | `v run ./vtl/examples/condition_number/main.v` |
| [batched_inverse](./batched_inverse) | Compute determinants and inverses of stacked matrices | `v run ./vtl/examples/batched_inverse/main.v` |
| [symmetric_eigen](./symmetric_eigen) | Compute symmetric eigenvalues and eigenvectors for batches | `v run ./vtl/examples/symmetric_eigen/main.v` |
| [matrix_rank](./matrix_rank) | Estimate matrix ranks with dtype-aware tolerances | `v run ./vtl/examples/matrix_rank/main.v` |
| [batched_solve](./batched_solve) | Solve stacked linear systems with vector or matrix right-hand sides | `v run ./vtl/examples/batched_solve/main.v` |
| [diff](./diff) | Compute discrete differences in one-dimensional and multidimensional data | `v run ./vtl/examples/diff/main.v` |
| [trapezoid](./trapezoid) | Integrate sampled data with the composite trapezoidal rule | `v run ./vtl/examples/trapezoid/main.v` |
| [gradient](./gradient) | Estimate numerical derivatives along a tensor axis | `v run ./vtl/examples/gradient/main.v` |
| [broadcast_mask](./broadcast_mask) | Select and fill values with broadcastable boolean masks | `v run ./vtl/examples/broadcast_mask/main.v` |
| [compress](./compress) | Select flattened values or tensor slices with a one-dimensional condition | `v run ./vtl/examples/compress/main.v` |
| [array_choose](./array_choose) | Select and broadcast tensor values by integer choice indices | `v run ./vtl/examples/array_choose/main.v` |
| [vector_norm](./vector_norm) | Compute stable p-norms globally and along an axis | `v run ./vtl/examples/vector_norm/main.v` |
| [fft_frequency](./fft_frequency) | Create FFT frequency bins and center a spectrum | `v run ./vtl/examples/fft_frequency/main.v` |
| [fft_axis](./fft_axis) | Transform real or complex tensors along one selected axis | `v run ./vtl/examples/fft_axis/main.v` |
| [covariance_correlation](./covariance_correlation) | Sample covariance and Pearson correlation | `v run ./vtl/examples/covariance_correlation/main.v` |
| [histogram](./histogram) | Count samples and compute weighted density with custom bin edges | `v run ./vtl/examples/histogram/main.v` |
| [bincount](./bincount) | Count integer labels and sum per-label weights | `v run ./vtl/examples/bincount/main.v` |
| [weighted_average](./weighted_average) | Compute weighted means globally and along an axis | `v run ./vtl/examples/weighted_average/main.v` |
| [unique](./unique) | Find sorted unique values and count occurrences | `v run ./vtl/examples/unique/main.v` |
| [set_membership](./set_membership) | Build a boolean mask for values contained in a set | `v run ./vtl/examples/set_membership/main.v` |
| [digitize](./digitize) | Assign values to monotonic bins and find insertion positions | `v run ./vtl/examples/digitize/main.v` |
| [clip](./clip) | Clamp a tensor with broadcastable per-element bounds in one pass | `v run ./vtl/examples/clip/main.v` |
| [bitwise](./bitwise) | Broadcast integer masks and shift flag bits | `v run ./vtl/examples/bitwise/main.v` |
| [take_nd](./take_nd) | Gather along an axis while preserving a multidimensional index shape | `v run ./vtl/examples/take_nd/main.v` |
| [mixed_index](./mixed_index) | Combine coordinate arrays, scalar indices, and Python-style slices | `v run ./vtl/examples/mixed_index/main.v` |
| [masked_array](./masked_array) | Broadcast masks, fill or compress values, reduce valid entries | `v run ./vtl/examples/masked_array/main.v` |

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
| [Vulkan CIFAR-10](./nn_cifar10_vulkan) | f32 Linear/Conv2D/ReLU/Adam | `VTL_USE_VULKAN=1 v -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v` |
| [synth](./nn_cifar10_f32_vulkan_tiny_synth) | Vulkan f32 smoke | `VTL_USE_VULKAN=1 v -d vulkan run vtl/examples/nn_cifar10_f32_vulkan_tiny_synth/main.v` |
| [OpenCL smoke](./vtl_opencl_vcl_support) | OpenCL tensor transfer and VCL compute | `v -d vcl run vtl/examples/vtl_opencl_vcl_support/main.v` |

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
