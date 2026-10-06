# VTL vs NumPy baselines

Run both sides from `~/.vmodules` under a memory-limited systemd scope. The VTL
benchmark calls `vtl.la.matmul` end to end, including tensor conversion and
result allocation. NumPy uses identical matrix values and sizes.

## Matmul

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -- env VJOBS=2 \
	v -prod run ./vtl/benchmarks/vs_numpy/matmul_bench.v
```

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -- env VJOBS=2 \
	OPENBLAS_NUM_THREADS=2 uv run --with numpy python \
	./vtl/benchmarks/vs_numpy/numpy_matmul_baseline.py
```

For a system CBLAS GEMM path on Linux, add `-d vsl_blas_generic_cblas` to the
V command. For OpenBLAS, use the existing `-d vsl_blas_cblas` flag when its
development package is installed. Keep the compiler mode and thread settings
in the report; results depend on both the V backend and NumPy's BLAS build.

## Local CPU sample

One local run on a Ryzen 9 5900X with V `-prod` and the pure-V BLAS backend
measured 512×512 `f64` GEMM at about 30 ms (8.9 GFLOPS). NumPy 2.5.3 with
OpenBLAS 0.3.34 and two BLAS threads measured about 2.2 ms (121.6 GFLOPS).
VTL is currently about 14× slower for this operation on this setup. The
benchmark uses 3 warmups and 10 timed calls; rerun it on the target host before
using the numbers for a release comparison. This gap is an optimization target,
not evidence of NumPy performance parity.

## Conv2D (CPU path)

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -- env VJOBS=2 \
	v -prod run ./vtl/benchmarks/vs_numpy/conv2d_bench.v
```

## Autograd (3-layer MLP backprop)

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -- env VJOBS=2 \
	v -prod run ./vtl/benchmarks/vs_numpy/autograd_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -- env VJOBS=2 \
	python3 ./vtl/benchmarks/vs_numpy/pytorch_baseline.py autograd
```

## Real FFT

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -- env VJOBS=2 \
	v -prod run ./vtl/benchmarks/vs_numpy/fft_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -- env VJOBS=2 \
	uv run --with numpy python ./vsl/benchmarks/fft_numpy_baseline.py
```

The FFT benchmark uses the same input sizes and iteration counts in VTL and
NumPy. Install NumPy in an isolated environment if needed, for example with
`uv run --with numpy python3 vsl/benchmarks/fft_numpy_baseline.py`.

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
