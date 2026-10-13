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

## ReLU autograd backward

`relu_backward_bench.v` compares the contiguous ReLU backward path with the
previous coordinate-indexed `nmap` implementation. It checks exact output
equality and reports local VTL CPU microbenchmark timings. This is an internal
kernel comparison, not a PyTorch performance claim. The same contiguous path
selection is used by sigmoid, tanh, and leaky ReLU backward operations; this
benchmark measures ReLU only:

```bash
cd ~/.vmodules
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -no-parallel -cc gcc -prod -cflags "-march=native" \
	-o /tmp/vtl-relu-backward-bench ./vtl/benchmarks/relu_backward_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 /tmp/vtl-relu-backward-bench
```

## GELU and Softplus autograd backward

`activation_backward_bench.v` compares the contiguous GELU and Softplus
backward paths with their previous coordinate-indexed `nmap` implementations.
It alternates operation order and checks every element of each final timed
output. These are local VTL CPU microbenchmarks, not PyTorch comparisons.

```bash
cd ~/.vmodules
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -no-parallel -cc gcc -prod -cflags "-march=native" \
	-o /tmp/vtl-activation-backward-bench ./vtl/benchmarks/activation_backward_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 \
	/tmp/vtl-activation-backward-bench
```

Three-process medians on an AMD Ryzen 9 5900X with V 0.5.2 and GCC 16.2.1:

| Elements | GELU nmap (ms) | GELU contiguous (ms) | Speedup |
|---------:|---------------:|--------------------:|--------:|
| 1,024 | 3.3896 | 0.0154 | 220.1x |
| 4,096 | 50.0460 | 0.0724 | 691.2x |
| 16,384 | 795.7903 | 0.2800 | 2,842.1x |

| Elements | Softplus nmap (ms) | Softplus contiguous (ms) | Speedup |
|---------:|------------------:|-----------------------:|--------:|
| 1,024 | 3.3954 | 0.0167 | 203.3x |
| 4,096 | 50.7017 | 0.0686 | 739.8x |
| 16,384 | 794.4893 | 0.2341 | 3,394.7x |

Each process used one warmup and five alternating timed calls. The comparison
is against VTL's coordinate-indexed `nmap` loop, whose per-element index work
dominates at larger sizes; these numbers are not comparisons with NumPy or
PyTorch and should be remeasured on other systems.

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

## Autograd binary gate fusion

`autograd_gate_backward_bench.v` compares `SubtractGate.backward` with
`multiply_scalar(-1)` and compares the separate tensor passes used by multiply
and divide gates with their fused backward paths on same-shaped, contiguous f64
tensors. It alternates subtract timing order, compares every element of the
final timed subtract outputs, and checks output checksums for multiply and divide
before reporting mean latency for 1,024, 65,536, and 1,048,576 elements. This
isolates gate backward work; it does not establish end-to-end training or
PyTorch performance parity.

```bash
cd ~/.vmodules
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -no-parallel -cc gcc -prod -cflags "-march=native" \
	-o /tmp/vtl-autograd-gates ./vtl/benchmarks/autograd_gate_backward_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 \
	/tmp/vtl-autograd-gates
```

Reference run on 2026-10-13: AMD Ryzen 9 5900X, V 0.5.2 (`37a3b74`), GCC
16.2.1, compiled with `-no-parallel -cc gcc -prod` and C flags
`-O3 -march=native`. Each table value is the median of three separate process
runs; each process reports the mean of 10 timed iterations after 3 warmups.

Subtract results from the three-process run:

| Elements | Scalar (ms) | Fused (ms) | Speedup |
|---------:|------------:|-----------:|--------:|
| 1,024 | 0.0082 | 0.0051 | 1.61x |
| 65,536 | 0.3225 | 0.0918 | 3.51x |
| 1,048,576 | 5.7572 | 3.1064 | 1.85x |

Earlier multiply/divide reference run:

| Elements | Multiply separate (ms) | Multiply fused (ms) | Speedup | Divide separate (ms) | Divide fused (ms) | Speedup |
|---------:|-----------------------:|-------------------:|--------:|---------------------:|-----------------:|--------:|
| 1,024 | 0.0135 | 0.0052 | 2.60x | 0.0283 | 0.0054 | 5.24x |
| 65,536 | 0.2915 | 0.0566 | 5.15x | 0.7586 | 0.1405 | 5.40x |
| 1,048,576 | 5.7262 | 4.9416 | 1.16x | 15.6397 | 5.1660 | 3.03x |

The subtract columns are medians of three processes on the same AMD Ryzen 9
5900X, V 0.5.2, GCC 16.2.1 configuration on 2026-10-13. Timings varied between
processes, so treat them as local reference data. The earlier multiply and
divide values retain their original run. The measured benefit depends on
operation and size; the earlier 1M-element multiply case improved only 1.16x.
This isolated f64 gate-backward microbenchmark does not establish end-to-end
training performance and does not close issue #40.

## Autograd clamp gate

`autograd_clamp_gate_backward_bench.v` compares the contiguous scalar clamp
kernel with the SIMD `ClampGate.backward` path for f32 and f64. It verifies
matching output checksums before timing and consumes and compares the outputs
from the timed iterations. Each process reports the mean of nine timed
iterations after three warmups.
Compile with `-prod` first and run the resulting executable separately:

```bash
cd ~/.vmodules
mkdir -p "$HOME/.cache/v/tmp_codex"
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	VTMP="$HOME/.cache/v/tmp_codex" TMPDIR="$HOME/.cache/v/tmp_codex" \
	VTOOLS_CACHE_DIR="$HOME/.cache/vtools-cache-codex" \
	v -no-parallel -cc gcc -prod -cflags "-march=native" \
	-o "$HOME/.cache/v/tmp_codex/vtl-autograd-clamp-gates" \
	./vtl/benchmarks/autograd_clamp_gate_backward_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 \
	"$HOME/.cache/v/tmp_codex/vtl-autograd-clamp-gates"
```

Reference run on 2026-10-13: AMD Ryzen 9 5900X, V 0.5.2 (`37a3b74`), GCC
16.2.1. Values below are medians of three separate process runs; times are in
milliseconds.

| Elements | f64 scalar | f64 SIMD | Speedup | f32 scalar | f32 SIMD | Speedup |
|---------:|-----------:|---------:|--------:|-----------:|---------:|--------:|
| 65,536 | 0.1479 | 0.0402 | 3.68x | 0.1173 | 0.0214 | 5.48x |
| 1,048,576 | 3.1617 | 1.8852 | 1.68x | 2.3850 | 0.6481 | 3.68x |

The 1,024-element timings are dominated by fixed overhead and were omitted;
the larger sizes are more representative of kernel throughput.

This microbenchmark measures clamp backward only; it does not establish
end-to-end training performance or NumPy/PyTorch parity.

Record CPU, V/compiler version, C compiler, production flags, and results when
publishing measurements.

## CI

The benchmark comment workflow from [#88](https://github.com/vlang/vtl/issues/88)
is available for PR evidence. Keep local runs scoped; full benchmark suites are
hardware-sensitive and should not be part of default development loops.
