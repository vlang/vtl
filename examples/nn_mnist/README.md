# MNIST Neural Network Training

This example trains a simple feed-forward network on MNIST.

Architecture:

- `784 -> 128 -> ReLU -> 64 -> ReLU -> 10`
- Loss: MSE with one-hot targets
- Optimizer: SGD

## Run

```sh
# CPU
v run examples/nn_mnist/main.v

# OpenCL/VCL-enabled build
v -d vcl run examples/nn_mnist/main.v

# Vulkan-enabled build
v -d vulkan run examples/nn_mnist/main.v
```

## GPU backend notes

- Backend flags enable GPU code paths where supported by the current operation set.
- End-to-end speedup depends on the exact model operations and backend coverage.
- Use `benchmark/main.v` for controlled backend comparisons on matmul workloads.

## Dataset

- MNIST is downloaded automatically on first run.
- Input normalization: `[0, 255] -> [0.0, 1.0]`.

## Practical tips

- Start with fewer batches/epochs while validating environment setup.
- Check OpenCL/Vulkan runtime availability before large runs.
