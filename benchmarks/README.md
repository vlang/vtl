# VTL benchmarks

Lightweight performance harness comparing VTL to external baselines (NumPy / PyTorch).

## Run locally

```bash
cd ~/.vmodules
v run vtl/benchmarks/vs_numpy/matmul_bench.v
v run vtl/benchmarks/vs_numpy/conv2d_bench.v
```

CUDA/Vulkan paths are opt-in:

```bash
VTL_USE_CUDA=1 v -d cuda run vtl/benchmarks/vs_numpy/matmul_bench.v
VTL_USE_VULKAN=1 v -prod -d vulkan run vtl/benchmarks/vs_numpy/matmul_bench.v
```

## NumPy / PyTorch reference

See [vs_numpy/README.md](vs_numpy/README.md) for Python reference commands
and result reporting guidance.

## CI

The benchmark comment workflow from [#88](https://github.com/vlang/vtl/issues/88)
is available for PR evidence. Keep local runs scoped; full benchmark suites are
hardware-sensitive and should not be part of default development loops.
