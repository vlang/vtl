# VTL benchmarks

Lightweight performance harness comparing VTL to external baselines (NumPy / PyTorch).

## Run locally

```bash
cd ~/.vmodules
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 v -prod run ./vtl/benchmarks/vs_numpy/matmul_bench.v
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 v -prod run ./vtl/benchmarks/vs_numpy/conv2d_bench.v
```

CUDA/Vulkan paths are opt-in:

```bash
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 VTL_USE_CUDA=1 v -d cuda run ./vtl/benchmarks/vs_numpy/matmul_bench.v
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 VTL_USE_VULKAN=1 v -prod -d vulkan run ./vtl/benchmarks/vs_numpy/matmul_bench.v
```

## NumPy / PyTorch reference

See [vs_numpy/README.md](vs_numpy/README.md) for Python reference commands
and result reporting guidance.

## Elementwise comparison allocation

`allclose_bench.v` compares `allclose` with the equivalent
`isclose(...).all()` path on 10,000 values. The direct reduction avoids
allocating an intermediate boolean tensor, uses a direct path for contiguous
inputs, and can stop at the first mismatch:

```bash
cd ~/.vmodules
systemd-run --user --scope --quiet -p MemoryMax=2G -p MemorySwapMax=0 --setenv=VJOBS=2 -- \
	v -prod run ./vtl/benchmarks/allclose_bench.v
```

Run it on the same host and build settings when comparing changes. It reports
equal inputs and an input that differs at the first element; results are
microbenchmark evidence, not a general NumPy performance comparison.

## Embedding kernels

`embedding_bench.v` compares contiguous `Embedding` forward and backward with
the coordinate-indexed implementation used as a reference. It verifies exact
output equality before reporting average times:

```bash
cd ~/.vmodules
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 --setenv=VJOBS=2 -- \
	v -prod run ./vtl/benchmarks/embedding_bench.v
```

One local run on an AMD Ryzen 9 5900X, V 0.5.2 (`2b15bbc`), using 8,192
tokens, embedding width 64, and 10 measured iterations reported:

| Operation | Contiguous path | Coordinate reference | Relative time |
| --- | ---: | ---: | ---: |
| Forward | 3.34 ms | 17.87 ms | 5.4× faster |
| Backward | 3.50 ms | 25.51 ms | 7.3× faster |

These are local CPU microbenchmarks against the prior coordinate-indexed VTL
loop, not comparisons with NumPy or other libraries. Repeat on the target host
with the same V build and settings before drawing performance conclusions.

## Linear bias gradient reduction

`linear_bias_reduction_bench.v` compares summing the batch axis with the
previous one-row matrix multiplication used by the linear layer backward pass.
It checks that both produce close results before timing them. Keep the GEMM
path unless measurements on supported build configurations show the reduction
is faster. Run it from `~/.vmodules` under a memory-limited systemd scope:

```bash
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 --setenv=VJOBS=2 -- \
	v run ./vtl/benchmarks/linear_bias_reduction_bench.v
```

## CI

The benchmark comment workflow from [#88](https://github.com/vlang/vtl/issues/88)
is available for PR evidence. Keep local runs scoped; full benchmark suites are
hardware-sensitive and should not be part of default development loops.
