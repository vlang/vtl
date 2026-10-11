// Compile with -prod from ~/.vmodules, then run the resulting binary.
module main

import time
import vtl

const cumulative_size = 1_048_576
const cumulative_iterations = 7

fn main() {
	mut sum_values := []f64{len: cumulative_size}
	mut product_values := []f64{len: cumulative_size}
	for i in 0 .. cumulative_size {
		sum_values[i] = f64(i % 1000) / 1000.0
		product_values[i] = 1.0 - f64(i % 16) * 1e-8
	}
	sum_input := vtl.from_1d[f64](sum_values) or { panic(err) }
	product_input := vtl.from_1d[f64](product_values) or { panic(err) }
	bench_cumsum(sum_input)
	bench_cumprod(product_input)
}

fn bench_cumsum(input &vtl.Tensor[f64]) {
	for _ in 0 .. 2 {
		_ = input.cumsum[f64](0) or { panic(err) }
	}
	mut result := input.cumsum[f64](0) or { panic(err) }
	mut samples := []f64{len: cumulative_iterations}
	for i in 0 .. cumulative_iterations {
		started := time.sys_mono_now()
		result = input.cumsum[f64](0) or { panic(err) }
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
	}
	println('vtl cumsum ${cumulative_size} f64 | ${mean_time(samples):.3} ms | checksum ${checksum(result):.6f}')
}

fn bench_cumprod(input &vtl.Tensor[f64]) {
	for _ in 0 .. 2 {
		_ = input.cumprod[f64](0) or { panic(err) }
	}
	mut result := input.cumprod[f64](0) or { panic(err) }
	mut samples := []f64{len: cumulative_iterations}
	for i in 0 .. cumulative_iterations {
		started := time.sys_mono_now()
		result = input.cumprod[f64](0) or { panic(err) }
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
	}
	println('vtl cumprod ${cumulative_size} f64 | ${mean_time(samples):.3} ms | checksum ${checksum(result):.6f}')
}

fn mean_time(samples []f64) f64 {
	mut total := f64(0)
	for sample in samples {
		total += sample
	}
	return total / samples.len
}

fn checksum(tensor &vtl.Tensor[f64]) f64 {
	mut total := f64(0)
	for value in tensor.data.data {
		total += value
	}
	return total
}
