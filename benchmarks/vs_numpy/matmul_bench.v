// End-to-end VTL matmul benchmark. Run from ~/.vmodules with `v -prod run`.
module main

import time
import vtl
import vtl.la as vtl_la
import vtl.benchmarks.util as bu

fn main() {
	bu.print_header('VTL matmul benchmark (f64 + optional f32 CBLAS)')
	config := bu.BenchConfig{
		sizes:       [128, 256, 512]
		iterations:  10
		warmup_runs: 3
	}
	bu.print_table_header()
	for n in config.sizes {
		bench_matmul(n, config)!
	}
	$if vsl_blas_cblas || vsl_blas_generic_cblas {
		println('\nVTL f32 matmul (CBLAS sgemm)')
		for n in [128, 256, 512, 1024, 2048] {
			bench_matmul_f32(n, config)!
		}
	} $else {
		println('\nVTL f32 matmul (pure-V sgemm)')
		for n in [128, 256, 512] {
			bench_matmul_f32(n, config)!
		}
	}
	println('\nNumPy baselines: numpy_matmul_baseline.py and numpy_matmul_f32_end_to_end.py')
}

fn bench_matmul(n int, config bu.BenchConfig) ! {
	mut a_values := []f64{len: n * n}
	mut b_values := []f64{len: n * n}
	for i in 0 .. n {
		for j in 0 .. n {
			a_values[i * n + j] = f64((i * 17 + j * 13) % 997 + 1) / 997.0
			b_values[i * n + j] = f64((i * 7 + j * 19) % 991 + 1) / 991.0
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

fn bench_matmul_f32(n int, config bu.BenchConfig) ! {
	mut a_values := []f32{len: n * n}
	mut b_values := []f32{len: n * n}
	for i in 0 .. n {
		for j in 0 .. n {
			a_values[i * n + j] = f32((i * 17 + j * 13) % 997 + 1) / 997.0
			b_values[i * n + j] = f32((i * 7 + j * 19) % 991 + 1) / 991.0
		}
	}
	a := vtl.from_array(a_values, [n, n])!
	b := vtl.from_array(b_values, [n, n])!
	for _ in 0 .. config.warmup_runs {
		_ := vtl_la.matmul[f32](a, b)!
	}
	mut samples := []f64{len: config.iterations}
	mut checksum := 0.0
	for i in 0 .. config.iterations {
		started := time.sys_mono_now()
		result := vtl_la.matmul[f32](a, b)!
		elapsed_ns := time.sys_mono_now() - started
		samples[i] = f64(elapsed_ns) / 1_000_000.0
		checksum += f64(result.get_nth[f32](n * n / 2 + n / 2))
	}
	avg := bu.mean_time_ms(mut samples)
	gflops := bu.gflops_gemm(n, n, n, avg)
	bu.print_row('gemm_f32', '${n}x${n}', avg, '${gflops}')
	println('checksum: ${checksum}')
}
