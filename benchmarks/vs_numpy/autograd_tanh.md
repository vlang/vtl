# Tanh autograd backward benchmark

This paired microbenchmark computes `gradient * (1 - cached_tanh * cached_tanh)`
for 1,048,576 contiguous `f64` values. Both implementations use three warmups
and nine timed calls, report the median, and check the same checksum. The V
benchmark exercises the public `autograd.TanhGate.backward` API; the Python
baseline uses NumPy array operations with output allocation. Compile V with
`-prod` before running its binary, from `~/.vmodules`:

```bash
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -no-parallel -cc gcc -prod -cflags "-O3 -march=native" \
	-o /tmp/vtl-tanh-backward-bench ./vtl/benchmarks/vs_numpy/autograd_tanh_backward_bench.v
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- env VJOBS=2 \
	/tmp/vtl-tanh-backward-bench
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 -- env VJOBS=2 \
	uv run --offline --no-project --with numpy python \
	./vtl/benchmarks/vs_numpy/numpy_autograd_tanh_backward_baseline.py
```

On the local AMD Ryzen 9 5900X workstation, V `-prod -march=native` measured
6.672 ms and `-prod -O3 -march=native` measured 5.339 ms; NumPy 2.5.4 measured
1.437 ms. Checksums matched at `983040`. NumPy was 3.72x faster than the
`-O3` VTL build in this one-operation comparison. These numbers are specific
to this CPU and input size and do not establish general framework performance.
