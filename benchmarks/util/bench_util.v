module util

import math

// BenchConfig defines a public data structure for this module.
pub struct BenchConfig {
pub:
	sizes       []int
	iterations  int = 5
	warmup_runs int = 2
}

// print_header exposes this operation as part of the public API.
pub fn print_header(title string) {
	println('\n${'='.repeat(72)}')
	println('  ${title}')
	println('${'='.repeat(72)}')
}

// print_table_header exposes this operation as part of the public API.
pub fn print_table_header() {
	println('Benchmark                | Size         | Avg (ms)     | GFLOPS')
	println('-------------------------+--------------+--------------+----------')
}

// gflops_gemm exposes this operation as part of the public API.
pub fn gflops_gemm(m int, n int, k int, time_ms f64) f64 {
	if time_ms <= 0.0 {
		return 0.0
	}
	ops := 2.0 * f64(m) * f64(n) * f64(k)
	sec := time_ms / 1000.0
	return ops / sec / 1_000_000_000.0
}

// format_ms exposes this operation as part of the public API.
pub fn format_ms(time_ms f64) string {
	return '${time_ms}'
}

// print_row exposes this operation as part of the public API.
pub fn print_row(name string, size string, time_ms f64, extra string) {
	println('${name} | ${size} | ${format_ms(time_ms)} | ${extra}')
}

// mean_time_ms exposes this operation as part of the public API.
pub fn mean_time_ms(mut samples []f64) f64 {
	if samples.len == 0 {
		return 0.0
	}
	mut sum := 0.0
	for t in samples {
		sum += t
	}
	return sum / f64(samples.len)
}

// stddev_time_ms exposes this operation as part of the public API.
pub fn stddev_time_ms(samples []f64, mean f64) f64 {
	if samples.len < 2 {
		return 0.0
	}
	mut acc := 0.0
	for t in samples {
		d := t - mean
		acc += d * d
	}
	return math.sqrt(acc / f64(samples.len - 1))
}
