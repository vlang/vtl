# VTL vs NumPy baselines

Run both sides from `~/.vmodules` under a memory-limited systemd scope. The VTL
benchmark calls `vtl.la.matmul` end to end, including tensor conversion and
result allocation. NumPy uses identical matrix values and sizes.

## Contiguous sum, mean, and variance

The VTL `stats_bench.v` measures contiguous sum and mean. The separate
`variance_bench.v` measures population variance. The NumPy baseline uses the
same `f64` values and calls `numpy.var` with its default `ddof=0`. Run both
from `~/.vmodules`:

```bash
cp ./vtl/benchmarks/vs_numpy/stats_bench.v /tmp/vtl_stats_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod run /tmp/vtl_stats_bench.v
cp ./vtl/benchmarks/vs_numpy/variance_bench.v /tmp/vtl_variance_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod run /tmp/vtl_variance_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	OPENBLAS_NUM_THREADS=2 uv run --with numpy python \
	./vtl/benchmarks/vs_numpy/numpy_stats_baseline.py
```

These timings cover reductions on already allocated tensors/arrays. They do not
include input construction; record the compiler, NumPy build, CPU, and thread
settings when comparing results.

On an AMD Ryzen 9 5900X with V 0.5.2 `-prod` and NumPy 2.5.3, one matched-input
run measured VTL contiguous sum at 0.0092 ms for 100k values and 0.0940 ms for
1M; NumPy measured 0.0201 ms and 0.1534 ms. VTL population variance measured
0.0204 ms and 0.2108 ms; NumPy measured 0.0736 ms and 0.5227 ms. These
single-host samples show VTL faster for these cases, not a general performance
guarantee. Rerun the committed benchmarks on the target machine before using
them for release claims.

## Vector p=2 norm

The VTL and NumPy cases use the same 500,000-element `f64` input, warm up
three times, then report the mean of seven timed reductions. Input allocation
is outside the timed region. Run both from `~/.vmodules`:

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod run ./vtl/benchmarks/vs_numpy/vector_norm_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	OPENBLAS_NUM_THREADS=2 uv run --with numpy python \
	./vtl/benchmarks/vs_numpy/numpy_vector_norm_baseline.py
```

Record V version, NumPy version, CPU, and BLAS build when comparing results.
On an AMD Ryzen 9 5900X with V 0.5.2 `-prod`, this VTL path measured 0.545 ms;
NumPy 2.5.3 measured 0.172 ms with `OPENBLAS_NUM_THREADS=2`. VTL is still
about 3.2x slower for this case, so this is a tracked optimization target rather
than a claim of parity or superiority.

## Matmul

For the 768 MiB workstation cap, run the f64 and f32 cases as separate
programs. Copy each source to `/tmp` so V compiles only that benchmark module;
all V commands still run from `~/.vmodules`:

```bash
cd ~/.vmodules
cp ./vtl/benchmarks/vs_numpy/f64/main/main.v /tmp/vtl_f64_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod run /tmp/vtl_f64_bench.v
cp ./vtl/benchmarks/vs_numpy/f32/main/main.v /tmp/vtl_f32_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod run /tmp/vtl_f32_bench.v
```

Both programs use the same deterministic inputs as the NumPy baselines, three
warmups, ten timed calls, and an output sanity check. The combined benchmark
below remains useful on machines with more compiler memory.

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod run ./vtl/benchmarks/vs_numpy/matmul_bench.v
```

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	OPENBLAS_NUM_THREADS=2 uv run --with numpy python \
	./vtl/benchmarks/vs_numpy/numpy_matmul_baseline.py
```

The same VTL benchmark measures single-precision matmul on the pure-V path
for sizes 128, 256, and 512. With CBLAS enabled, it measures sizes through
2048. Compare its `f32` rows against the matching NumPy end-to-end baseline;
both allocate a fresh result on each timed call and use the same inputs:

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	OPENBLAS_NUM_THREADS=2 v -d vsl_blas_generic_cblas -prod run ./vtl/benchmarks/vs_numpy/matmul_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	OPENBLAS_NUM_THREADS=2 uv run --with numpy python \
	./vtl/benchmarks/vs_numpy/numpy_matmul_baseline.py
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	OPENBLAS_NUM_THREADS=2 uv run --with numpy python \
	./vtl/benchmarks/vs_numpy/numpy_matmul_f32_end_to_end.py
```

For a system CBLAS GEMM path on Linux, add `-d vsl_blas_generic_cblas` to the
V command. For OpenBLAS, use the existing `-d vsl_blas_cblas` flag when its
development package is installed. Keep the compiler mode and thread settings
in the report; results depend on both the V backend and NumPy's BLAS build.
With either CBLAS flag, VTL `f32` matrix multiplication dispatches to
single-precision CBLAS `sgemm`; without those flags, it uses VSL's pure-V
`sgemm` implementation. `f64` continues to use `dgemm`. The benchmark reports
both `f64` and `f32` results for either backend.

## Local CPU sample

Matched-input local runs on a Ryzen 9 5900X with V 0.5.2 `-prod` and the
pure-V BLAS backend measured dense 512×512 `f64` GEMM at 16.84 ms (15.9
GFLOPS). NumPy 2.5.3 with scipy-openblas 0.3.34.106.0 and two BLAS threads
measured 2.12 ms (126.5 GFLOPS), making the current pure-V path about 7.9×
slower on this host. The VTL `f32` pure-V benchmark measured 6.02 ms at the
same size; its NumPy baseline measured 1.16 ms. The system CBLAS path backed by
the host's generic `libblas` previously measured 36.2 ms; this machine does
not have OpenBLAS installed for VSL's optimized `-d vsl_blas_cblas` backend.
Each matched benchmark uses 3 warmups and 10 timed calls. Rerun on the target
host before using these numbers for a release comparison. The remaining gaps
are optimization targets, not evidence of NumPy performance parity.

## Resident-buffer Vulkan f32 GEMM

This benchmark uploads both inputs once, reuses a GPU output buffer, and times
the VTL Vulkan GEMM operation without host readback in the timed section. Run
the NumPy reference with the same f32 inputs and a preallocated output:

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	VTL_USE_VULKAN=1 v -prod -d vulkan run \
	./vtl/benchmarks/vs_numpy/vulkan_matmul_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	OPENBLAS_NUM_THREADS=2 uv run --with numpy python \
	./vtl/benchmarks/vs_numpy/numpy_matmul_f32_baseline.py
```

The GPU result is read back only after timing to validate a checksum. Compare
these resident-buffer timings as accelerator kernel measurements; they exclude
CPU↔GPU transfer costs.

On the Ryzen 9 5900X with an RTX 3060, the tiled 32×32 Vulkan kernel measured
1.34 ms for 512×512, 4.39 ms for 1024×1024, and 42.95 ms for 2048×2048.
Earlier NumPy runs on the same host with two CPU BLAS threads measured 1.0 ms,
8.0 ms, and 62.2 ms respectively. This Vulkan kernel is about 1.8× faster at
1024×1024 and 1.4× faster at 2048×2048, while it remains slower at 512×512.
These resident-buffer kernel measurements exclude CPU↔GPU transfers, so they
do not establish end-to-end superiority over NumPy.

## Conv2D (CPU path)

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod run ./vtl/benchmarks/vs_numpy/conv2d_bench.v
```

## Autograd (3-layer MLP backprop)

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod run ./vtl/benchmarks/vs_numpy/autograd_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	python3 ./vtl/benchmarks/vs_numpy/pytorch_baseline.py autograd
```

## Real FFT

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod run ./vtl/benchmarks/vs_numpy/fft_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	uv run --with numpy python ./vsl/benchmarks/fft_numpy_baseline.py
```

The FFT benchmark covers real f64 and real f32 inputs with the same sizes and
iteration counts in VTL and NumPy. For real f32, VTL's dedicated API returns
complex f32 output while NumPy's `np.fft.rfft` promotes to complex128. Install
NumPy in an isolated environment if needed, for example with
`uv run --with numpy python3 ./vsl/benchmarks/fft_numpy_baseline.py`.

## Complex f32 FFT

The complex f32 benchmark uses deterministic complex64 input, matching sizes,
three warmups, and identical iteration counts. It measures the public VTL FFT
path, including plan creation and output allocation, against NumPy's public
`fft` call. NumPy currently promotes complex64 inputs to complex128 output, so
the timings compare user-facing behavior rather than identical internal
precision. Run both commands from `~/.vmodules`:

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	v -prod run ./vtl/benchmarks/vs_numpy/fft_complex_f32_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -p MemorySwapMax=0 -- env VJOBS=2 \
	uv run --with numpy python ./vtl/benchmarks/vs_numpy/numpy_fft_complex_f32_baseline.py
```

This benchmark reports measurements but does not claim a speedup before a
matched run on the target machine.

## Notes

- Use the same matrix sizes when comparing manually.
- Report GFLOPS from the VTL script output for PR comments.
- CUDA paths require `VTL_USE_CUDA=1` and `-d cuda` where applicable.
- Vulkan paths require `VTL_USE_VULKAN=1`, `-d vulkan`, and usually `v -prod`.
- Autograd comparisons use PyTorch as the reference; record CPU/GPU model and flags.

## Suggested report fields

| Field | Example |
|-------|---------|
| Operation | `matmul`, `conv2d`, `autograd` |
| Backend | CPU, CUDA, Vulkan |
| Shape | Matrix size or model dimensions |
| V flags | `-d cuda`, `-d vulkan`, `-prod` |
| Baseline | NumPy or PyTorch timing |
