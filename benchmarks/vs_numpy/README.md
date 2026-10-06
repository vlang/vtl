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

Matched-input local runs on a Ryzen 9 5900X with V `-prod` and the pure-V BLAS
backend measured dense 512×512 `f64` GEMM at about 37.0 ms (7.3 GFLOPS).
NumPy 2.5.3 with OpenBLAS 0.3.34 and two BLAS threads measured about 2.3 ms
(115 GFLOPS). VTL is currently about 16× slower for this operation on this
setup. Each benchmark uses 3 warmups and 10 timed calls; rerun on the target
host before using these numbers for a release comparison. This gap is an
optimization target, not evidence of NumPy performance parity.

## Resident-buffer Vulkan f32 GEMM

This benchmark uploads both inputs once, reuses a GPU output buffer, and times
the VTL Vulkan GEMM operation without host readback in the timed section. Run
the NumPy reference with the same f32 inputs and a preallocated output:

```bash
systemd-run --user --scope --quiet -p MemoryMax=768M -- env VJOBS=2 \
	VTL_USE_VULKAN=1 v -prod -d vulkan run \
	./vtl/benchmarks/vs_numpy/vulkan_matmul_bench.v
systemd-run --user --scope --quiet -p MemoryMax=768M -- env VJOBS=2 \
	OPENBLAS_NUM_THREADS=2 uv run --with numpy python \
	./vtl/benchmarks/vs_numpy/numpy_matmul_f32_baseline.py
```

The GPU result is read back only after timing to validate a checksum. Compare
these resident-buffer timings as accelerator kernel measurements; they exclude
CPU↔GPU transfer costs.

On the same machine, one local run measured VTL Vulkan at 1.7 ms for 512×512,
13.1 ms for 1024×1024, and 159.2 ms for 2048×2048. NumPy with two CPU BLAS
threads measured 1.0 ms, 8.0 ms, and 62.2 ms respectively. The current Vulkan
GEMM kernel is slower in this comparison and needs more work.

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
