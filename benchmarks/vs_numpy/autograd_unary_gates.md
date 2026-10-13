# Unary autograd backward benchmarks

The paired benchmark covers contiguous `f64` backward passes for `tanh`, `abs`,
`sqrt`, and `clamp`, each with 1,048,576 elements. Each process performs three
warmups and nine timed calls per operation. Both implementations allocate the
output inside each call and report a checksum. V uses `simd.F64x4` for `tanh`,
`abs`, and `sqrt` on the C backend, then handles remaining elements with a
scalar tail. `clamp` uses a contiguous flat loop. Strided and broadcast tensors
continue through the general paths.

Compile V with `-prod` before executing the binary. Run all commands from
`~/.vmodules`:

```bash
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v -no-parallel -cc gcc -prod -cflags "-O3 -march=native" \
	-o /tmp/vtl-autograd-unary-bench ./vtl/benchmarks/vs_numpy/autograd_unary_gates_backward_bench.v
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 -- env VJOBS=2 \
	/tmp/vtl-autograd-unary-bench
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 -- env VJOBS=2 \
	uv run --offline --no-project --with numpy python \
	./vtl/benchmarks/vs_numpy/numpy_autograd_unary_gates_baseline.py
```

On an AMD Ryzen 9 5900X with V 0.5.2 `dc16e46`, GCC 16.2.1, and NumPy 2.5.4,
interleaved runs measured these medians across six independent process medians
per side. The clamp row uses three process medians per side after keeping its
flat scalar loop; the earlier SIMD select path was slower on this workload.

| Gate | VTL median (ms) | NumPy median (ms) | Faster in this workload |
| --- | ---: | ---: | --- |
| `tanh` | 1.459 | 1.510 | VTL, 1.03x |
| `abs` | 2.112 | 1.837 | NumPy, 1.15x |
| `sqrt` | 3.327 | 1.952 | NumPy, 1.70x |
| `clamp` | 1.985 | 0.822 | NumPy, 2.42x |

All corresponding checksums matched: `983040`, `1310720`, `327680`, and
`1310720`. These measurements cover one CPU, one shape, and one operation per
call. They show a small `tanh` win and useful fusion over VTL's previous
multi-pass formulas, while NumPy remains faster for the other measured gates.
They do not establish overall NumPy or PyTorch performance parity.

V's [`simd` module documentation](https://github.com/vlang/v/blob/master/vlib/simd/README.md)
describes the portable vector API. GCC and Clang lower its operations to
available target instructions; other C compilers use scalar implementations.
