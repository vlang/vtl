// Compare the contiguous ReLU backward fast path with its prior nmap reference.
// Compile with -prod from ~/.vmodules, then execute the resulting binary.
module main

import math
import time
import vtl
import vtl.nn.internal

const sizes = [1024, 4096, 16384]
const warmup_runs = 1
const iterations = 3

fn main() {
	println('VTL ReLU backward (f64, CPU); microbenchmark vs coordinate-indexed nmap reference')
	println('Elements | nmap reference (ms) | contiguous fast path (ms) | speedup')
	for size in sizes {
		reference_ms, fast_ms := benchmark(size)!
		println('${size} | ${reference_ms:.4f} | ${fast_ms:.4f} | ${reference_ms / fast_ms:.2f}x')
	}
}

fn benchmark(size int) !(f64, f64) {
	mut gradient_values := []f64{len: size}
	mut cached_values := []f64{len: size}
	for i in 0 .. size {
		gradient_values[i] = f64((i % 19) - 9) / 10.0
		cached_values[i] = f64((i % 23) - 11) / 10.0
	}
	gradient := vtl.from_1d(gradient_values)!
	cached := vtl.from_1d(cached_values)!
	mut reference := vtl.zeros[f64]([size])
	mut fast := vtl.zeros[f64]([size])
	for _ in 0 .. warmup_runs {
		reference = relu_backward_nmap_reference(gradient, cached)!
		fast = internal.deriv_relu[f64](gradient, cached)!
	}
	assert reference.allclose(fast, rtol: 0.0, atol: 0.0)!
	reference_ms := time_repeated(gradient, cached, false)
	fast_ms := time_repeated(gradient, cached, true)
	assert math.abs(sum_values(reference) - sum_values(fast)) < 1e-9
	return reference_ms, fast_ms
}

fn sum_values(values &vtl.Tensor[f64]) f64 {
	mut total := 0.0
	for value in values.data.data {
		total += value
	}
	return total
}

fn relu_backward_nmap_reference(gradient &vtl.Tensor[f64], cached &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	return gradient.nmap([cached], fn (values []f64, _ []int) f64 {
		return if values[1] < 0.0 { 0.0 } else { values[0] }
	})
}

fn time_repeated(gradient &vtl.Tensor[f64], cached &vtl.Tensor[f64], fast bool) f64 {
	mut times := []f64{len: iterations}
	for i in 0 .. iterations {
		start := time.sys_mono_now()
		if fast {
			_ := internal.deriv_relu[f64](gradient, cached) or { panic(err) }
		} else {
			_ := relu_backward_nmap_reference(gradient, cached) or { panic(err) }
		}
		times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
	}
	return mean(mut times)
}

fn mean(mut values []f64) f64 {
	mut total := 0.0
	for value in values {
		total += value
	}
	return total / values.len
}
