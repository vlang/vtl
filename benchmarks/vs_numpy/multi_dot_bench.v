// Compare optimal matrix-chain ordering with left-associated multiplication.
// Run from ~/.vmodules with: v -prod run ./vtl/benchmarks/vs_numpy/multi_dot_bench.v
module main

import time
import vtl
import vtl.la
import vtl.stats

fn main() {
	a := vtl.ones[f64]([400, 40])
	b := vtl.ones[f64]([40, 4000])
	c := vtl.ones[f64]([4000, 40])
	iterations := 10
	println('VTL matrix-chain benchmark (400x40 · 40x4000 · 4000x40)')
	println('Ordering | Mean ms | Checksum')
	println('---------+---------+---------')
	for optimized in [true, false] {
		for _ in 0 .. 2 {
			_ = multiply_chain(a, b, c, optimized)!
		}
		mut elapsed_ns := i64(0)
		mut checksum := 0.0
		for _ in 0 .. iterations {
			started := time.sys_mono_now()
			result := multiply_chain(a, b, c, optimized)!
			elapsed_ns += time.sys_mono_now() - started
			checksum += stats.sum[f64](result)
		}
		name := if optimized { 'multi_dot' } else { 'left-associated' }
		mean_ms := f64(elapsed_ns) / f64(iterations) / 1_000_000.0
		println('${name} | ${mean_ms:.4f} | ${checksum:.1f}')
	}
}

fn multiply_chain(a &vtl.Tensor[f64], b &vtl.Tensor[f64], c &vtl.Tensor[f64], optimized bool) !&vtl.Tensor[f64] {
	if optimized {
		return la.multi_dot[f64]([a, b, c])
	}
	ab := la.matmul[f64](a, b)!
	return la.matmul[f64](ab, c)
}
