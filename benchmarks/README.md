# VTL benchmarks

Lightweight performance harness comparing VTL to external baselines (NumPy / PyTorch).

## Run locally

Compile benchmark programs with `-prod`, then execute the produced binary in a
separate memory-limited scope. This keeps compilation out of the timed run and
avoids `v run` for benchmark measurements. V's production build enables its
production optimizations; use the same C compiler and CPU-target flags for VTL
and any VTL backend variant being compared. Add `-cflags "-march=native"` only
for machine-local results, and record that choice because it makes the binary
CPU-specific.

```bash
cd ~/.vmodules
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod -o /tmp/vtl-matmul-bench ./vtl/benchmarks/vs_numpy/matmul_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 /tmp/vtl-matmul-bench
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod -o /tmp/vtl-conv2d-bench ./vtl/benchmarks/vs_numpy/conv2d_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 /tmp/vtl-conv2d-bench
```

CUDA/Vulkan paths are opt-in:

```bash
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod -d cuda -o /tmp/vtl-matmul-cuda ./vtl/benchmarks/vs_numpy/matmul_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 VTL_USE_CUDA=1 /tmp/vtl-matmul-cuda
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod -d vulkan -o /tmp/vtl-matmul-vulkan ./vtl/benchmarks/vs_numpy/matmul_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 VTL_USE_VULKAN=1 /tmp/vtl-matmul-vulkan
```

## NumPy / PyTorch reference

See [vs_numpy/README.md](vs_numpy/README.md) for Python reference commands
and result reporting guidance. Compare VTL to NumPy for array and numerical
operations, and to PyTorch for tensor/autograd/NN workloads. Select VTL's best
available backend for the workload (for example, optimized CBLAS for CPU GEMM,
CUDA/cuBLAS on supported NVIDIA systems, or a supported Vulkan path), and
compare it with the corresponding NumPy BLAS or PyTorch device backend on the
same hardware. Report both end-to-end timings, including transfers when users
would incur them, and kernel-only timings separately when useful. Never compare
a VTL CPU path with a GPU reference and label it a library-wide result.

## Elementwise comparison allocation

`allclose_bench.v` compares `allclose` with the equivalent
`isclose(...).all()` path on 10,000 values. The direct reduction avoids
allocating an intermediate boolean tensor, uses a direct path for contiguous
inputs, and can stop at the first mismatch:

```bash
cd ~/.vmodules
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod -o /tmp/vtl-allclose-bench ./vtl/benchmarks/allclose_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 /tmp/vtl-allclose-bench
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
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod -o /tmp/vtl-embedding-bench ./vtl/benchmarks/embedding_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 /tmp/vtl-embedding-bench
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
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod -o /tmp/vtl-linear-bias-reduction-bench ./vtl/benchmarks/linear_bias_reduction_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 /tmp/vtl-linear-bias-reduction-bench
```

## CI

The benchmark comment workflow from [#88](https://github.com/vlang/vtl/issues/88)
is available for PR evidence. Keep local runs scoped; full benchmark suites are
hardware-sensitive and should not be part of default development loops.
