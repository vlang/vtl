module util

import math

pub struct BenchConfig {
pub:
	sizes       []int
	iterations  int = 5
	warmup_runs int = 2
}

pub fn print_header(title string) {
	println('\n${'='.repeat(72)}')
	println('  ${title}')
	println('${'='.repeat(72)}')
}

pub fn print_table_header() {
	println('Benchmark                | Size         | Avg (ms)     | GFLOPS')
	println('-------------------------+--------------+--------------+----------')
}

pub fn gflops_gemm(m int, n int, k int, time_ms f64) f64 {
	if time_ms <= 0.0 {
		return 0.0
	}
	ops := f64(2 * m * n * k)
	sec := time_ms / 1000.0
	return ops / sec / 1_000_000_000.0
}

pub fn format_ms(time_ms f64) string {
	return '${time_ms}'
}

pub fn print_row(name string, size string, time_ms f64, extra string) {
	println('${name} | ${size} | ${format_ms(time_ms)} | ${extra}')
}

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
