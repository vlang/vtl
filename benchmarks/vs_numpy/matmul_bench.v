// End-to-end VTL matmul benchmark. Run from ~/.vmodules with `v -prod run`.
module main

import time
import vtl
import vtl.la as vtl_la
import vtl.benchmarks.util as bu

fn main() {
	bu.print_header('VTL matmul benchmark (vtl.la API, f64)')
	config := bu.BenchConfig{
		sizes:       [128, 256, 512]
		iterations:  10
		warmup_runs: 3
	}
	bu.print_table_header()
	for n in config.sizes {
		bench_matmul(n, config)!
	}
	println('\nNumPy baseline: run numpy_matmul_baseline.py with the same BLAS thread count')
}

fn bench_matmul(n int, config bu.BenchConfig) ! {
	mut a_values := []f64{len: n * n}
	mut b_values := []f64{len: n * n}
	for i in 0 .. n {
		for j in 0 .. n {
			a_values[i * n + j] = f64((i + j) % 7) * 0.01
			b_values[i * n + j] = f64((i * j) % 5) * 0.02
		}
	}
	a := vtl.from_array(a_values, [n, n])!
	b := vtl.from_array(b_values, [n, n])!
	for _ in 0 .. config.warmup_runs {
		_ := vtl_la.matmul[f64](a, b)!
	}
	mut samples := []f64{len: config.iterations}
	mut checksum := 0.0
	for i in 0 .. config.iterations {
		started := time.sys_mono_now()
		result := vtl_la.matmul[f64](a, b)!
		elapsed_ns := time.sys_mono_now() - started
		samples[i] = f64(elapsed_ns) / 1_000_000.0
		checksum += result.get_nth[f64](n * n / 2 + n / 2)
	}
	avg := bu.mean_time_ms(mut samples)
	gflops := bu.gflops_gemm(n, n, n, avg)
	bu.print_row('gemm', '${n}x${n}', avg, '${gflops}')
	println('checksum: ${checksum}')
}
