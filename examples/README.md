# VTL Examples

This directory contains runnable examples for tensor ops, datasets, autograd, and neural networks.

## Run pattern

```sh
v run examples/<example_folder>/main.v
```

Some examples use alternate file names (for example, `main_not_ci.v`) when excluded from CI.

## GPU-related examples

- `vtl_opencl_vcl_support` — OpenCL/VCL conversion and execution flow
- `nn_mnist` — MNIST training example (can be built with `-d vcl` / `-d vulkan`)

## Full list

- `autograd_backprop`
- `datasets_imdb`
- `datasets_mnist`
- `nn_autoencoder_simple`
- `nn_mnist`
- `nn_multiclass_iris`
- `nn_regression_sine`
- `nn_simple_two_layer`
- `nn_xor`
- `vtl_broadcast_basics`
- `vtl_basic_usage`
- `vtl_opencl_vcl_support`
- `vtl_plot_scatter_colorscale`
- `vtl_vandermont`
