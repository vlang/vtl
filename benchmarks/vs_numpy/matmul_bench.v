// VTL matmul benchmark — reports avg ms and GFLOPS per matrix size.
// Times vsl.la GEMM only (matrices pre-built; no per-iter tensor copies).
// Run: v run vtl/benchmarks/vs_numpy/matmul_bench.v
module main

import time
import vsl.la as vsl_la
import vtl.benchmarks.util as bu

fn main() {
	bu.print_header('VTL matmul benchmark (vsl.la GEMM, f64)')
	config := bu.BenchConfig{
		sizes:       [128, 256, 512]
		iterations:  5
		warmup_runs: 2
	}
	bu.print_table_header()
	for n in config.sizes {
		bench_matmul(n, config)
	}
	println('\nCompare with NumPy: see benchmarks/vs_numpy/README.md')
}

fn bench_matmul(n int, config bu.BenchConfig) {
	mut a := vsl_la.Matrix.new[f64](n, n)
	mut b := vsl_la.Matrix.new[f64](n, n)
	mut c := vsl_la.Matrix.new[f64](n, n)
	for i in 0 .. n {
		for j in 0 .. n {
			a.set(i, j, f64((i + j) % 7) * 0.01)
			b.set(i, j, f64((i * j) % 5) * 0.02)
		}
	}

	for _ in 0 .. config.warmup_runs {
		vsl_la.matrix_matrix_mul(mut c, 1.0, a, b)
	}

	mut samples := []f64{len: config.iterations}
	for i in 0 .. config.iterations {
		t0 := time.ticks()
		vsl_la.matrix_matrix_mul(mut c, 1.0, a, b)
		samples[i] = f64(time.ticks() - t0)
	}
	avg := bu.mean_time_ms(mut samples)
	gflops := bu.gflops_gemm(n, n, n, avg)
	bu.print_row('gemm', '${n}x${n}', avg, '${gflops}')
}
