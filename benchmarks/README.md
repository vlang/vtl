# VTL benchmarks

Lightweight performance harness comparing VTL to external baselines (NumPy / PyTorch).

## Run locally

```bash
cd ~/.vmodules
v run vtl/benchmarks/vs_numpy/matmul_bench.v
v run vtl/benchmarks/vs_numpy/conv2d_bench.v
```

CUDA matmul (when wired):

```bash
VTL_USE_CUDA=1 v -d cuda run vtl/benchmarks/vs_numpy/matmul_bench.v
```

## NumPy reference

See [vs_numpy/README.md](vs_numpy/README.md) for copy-paste Python `%timeit` commands.

## CI

Full GH Actions benchmark comment workflow is tracked in [#88](https://github.com/vlang/vtl/issues/88)
(Phase D). This directory provides the runnable scripts and GFLOPS reporting.
