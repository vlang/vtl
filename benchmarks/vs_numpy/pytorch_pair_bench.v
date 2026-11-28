// End-to-end VTL baselines matched to pytorch_pair_baseline.py.
// Compile with -prod from ~/.vmodules, then run the binary.
module main

import os
import time
import vtl
import vtl.stats
import vtl.la as vtl_la
import vtl.benchmarks.util as bu

const warmup_runs = 3

const iterations = 10

fn main() {
	bu.print_header('VTL / PyTorch matched CPU benchmarks (f64)')
	backend := $if vsl_blas_cblas || vsl_blas_generic_cblas { 'CBLAS' } $else { 'pure-V' }
	println('vtl | device=cpu | dtype=float64 | backend=${backend} | VJOBS=${env_value('VJOBS')} | warmup=${warmup_runs} | iterations=${iterations}')
	println('source | op | size | median_ms | stddev_ms | checksum')
	println('------+----+------+-----------+-----------+---------')
	for n in [100_000, 1_000_000] {
		tensor := vtl.from_array(vector_values(n), [n])!
		bench_stat('sum', n, tensor, false)
		bench_stat('mean', n, tensor, true)
		bench_tanh(n, tensor)
	}
	for n in [128, 256, 512] {
		bench_gemm(n)!
	}
	println('\nPyTorch reference: python3 pytorch_pair_baseline.py all')
}

fn env_value(name string) string {
	return os.getenv_opt(name) or { 'unset' }
}

fn vector_values(n int) []f64 {
	mut values := []f64{len: n}
	for i in 0 .. n {
		values[i] = f64(i % 1000 + 1) / 1000.0
	}
	return values
}

// median_time_ms returns the middle value of a copy of the samples.
fn median_time_ms(samples []f64) f64 {
	mut sorted := samples.clone()
	sorted.sort()
	mid := sorted.len / 2
	if sorted.len % 2 == 0 {
		return (sorted[mid - 1] + sorted[mid]) / 2.0
	}
	return sorted[mid]
}

fn bench_stat(op string, n int, tensor &vtl.Tensor[f64], use_mean bool) {
	for _ in 0 .. warmup_runs {
		if use_mean {
			_ = stats.mean(tensor)
		} else {
			_ = stats.sum(tensor)
		}
	}
	mut samples := []f64{len: iterations}
	mut checksum := 0.0
	for i in 0 .. iterations {
		started := time.sys_mono_now()
		result := if use_mean { stats.mean(tensor) } else { stats.sum(tensor) }
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
		checksum += result
	}
	median := median_time_ms(samples)
	stddev := bu.stddev_time_ms(samples, bu.mean_time_ms(mut samples))
	println('vtl | ${op} | ${n} | ${median:.4f} | ${stddev:.4f} | ${checksum:.4f}')
}

fn bench_tanh(n int, tensor &vtl.Tensor[f64]) {
	for _ in 0 .. warmup_runs {
		_ = vtl.tanh_promoted[f64, f64](tensor) or { panic(err) }
	}
	mut samples := []f64{len: iterations}
	mut checksum := 0.0
	for i in 0 .. iterations {
		started := time.sys_mono_now()
		result := vtl.tanh_promoted[f64, f64](tensor) or { panic(err) }
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
		checksum += result.get_nth[f64](n / 2)
	}
	median := median_time_ms(samples)
	stddev := bu.stddev_time_ms(samples, bu.mean_time_ms(mut samples))
	println('vtl | tanh | ${n} | ${median:.4f} | ${stddev:.4f} | ${checksum:.4f}')
}

fn bench_gemm(n int) ! {
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
	for _ in 0 .. warmup_runs {
		_ = vtl_la.matmul[f64](a, b)!
	}
	mut samples := []f64{len: iterations}
	mut checksum := 0.0
	for i in 0 .. iterations {
		started := time.sys_mono_now()
		result := vtl_la.matmul[f64](a, b)!
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
		checksum += result.get_nth[f64](n * n / 2 + n / 2)
	}
	median := median_time_ms(samples)
	stddev := bu.stddev_time_ms(samples, bu.mean_time_ms(mut samples))
	println('vtl | gemm | ${n}x${n} | ${median:.4f} | ${stddev:.4f} | ${checksum:.6f}')
}
