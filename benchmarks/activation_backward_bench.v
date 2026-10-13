// Compare contiguous GELU and Softplus backward kernels with their nmap references.
// Compile with -prod from ~/.vmodules, then execute the resulting binary.
module main

import math
import time
import vtl
import vtl.nn.internal

const sizes = [1024, 4096, 16384]
const warmup_runs = 1
const iterations = 5

fn main() {
	println('VTL activation backward (f64, CPU); contiguous path vs nmap reference')
	println('Elements | GELU nmap (ms) | GELU fast (ms) | speedup | Softplus nmap (ms) | Softplus fast (ms) | speedup')
	for size in sizes {
		gelu_reference, gelu_fast, softplus_reference, softplus_fast := benchmark(size)!
		println('${size} | ${gelu_reference:.4f} | ${gelu_fast:.4f} | ${gelu_reference / gelu_fast:.2f}x | ${softplus_reference:.4f} | ${softplus_fast:.4f} | ${softplus_reference / softplus_fast:.2f}x')
	}
}

fn benchmark(size int) !(f64, f64, f64, f64) {
	mut gradient_values := []f64{len: size}
	mut cached_values := []f64{len: size}
	for i in 0 .. size {
		gradient_values[i] = 0.25 + f64(i % 17) / 8.0
		cached_values[i] = f64(i % 101 - 50) / 10.0
	}
	gradient := vtl.from_1d(gradient_values)!
	cached := vtl.from_1d(cached_values)!
	mut gelu_reference := vtl.zeros[f64]([size])
	mut gelu_fast := vtl.zeros[f64]([size])
	mut softplus_reference := vtl.zeros[f64]([size])
	mut softplus_fast := vtl.zeros[f64]([size])
	for _ in 0 .. warmup_runs {
		gelu_reference = gelu_nmap_reference(gradient, cached)!
		gelu_fast = internal.deriv_gelu[f64](gradient, cached)!
		softplus_reference = softplus_nmap_reference(gradient, cached)!
		softplus_fast = internal.deriv_softplus[f64](gradient, cached)!
	}
	assert_outputs_equal(gelu_reference, gelu_fast)
	assert_outputs_equal(softplus_reference, softplus_fast)
	gelu_reference_ms, gelu_fast_ms, timed_gelu_reference, timed_gelu_fast := time_gelu(gradient, cached)
	softplus_reference_ms, softplus_fast_ms, timed_softplus_reference, timed_softplus_fast := time_softplus(gradient,
		cached)
	assert_outputs_equal(timed_gelu_reference, timed_gelu_fast)
	assert_outputs_equal(timed_softplus_reference, timed_softplus_fast)
	return gelu_reference_ms, gelu_fast_ms, softplus_reference_ms, softplus_fast_ms
}

fn time_gelu(gradient &vtl.Tensor[f64], cached &vtl.Tensor[f64]) (f64, f64, &vtl.Tensor[f64], &vtl.Tensor[f64]) {
	mut reference_times := []f64{len: iterations}
	mut fast_times := []f64{len: iterations}
	mut reference_output := gelu_nmap_reference(gradient, cached) or { panic(err) }
	mut fast_output := internal.deriv_gelu[f64](gradient, cached) or { panic(err) }
	for i in 0 .. iterations {
		if i % 2 == 0 {
			mut start := time.sys_mono_now()
			reference_output = gelu_nmap_reference(gradient, cached) or { panic(err) }
			reference_times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
			start = time.sys_mono_now()
			fast_output = internal.deriv_gelu[f64](gradient, cached) or { panic(err) }
			fast_times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
		} else {
			mut start := time.sys_mono_now()
			fast_output = internal.deriv_gelu[f64](gradient, cached) or { panic(err) }
			fast_times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
			start = time.sys_mono_now()
			reference_output = gelu_nmap_reference(gradient, cached) or { panic(err) }
			reference_times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
		}
	}
	return mean(mut reference_times), mean(mut fast_times), reference_output, fast_output
}

fn time_softplus(gradient &vtl.Tensor[f64], cached &vtl.Tensor[f64]) (f64, f64, &vtl.Tensor[f64], &vtl.Tensor[f64]) {
	mut reference_times := []f64{len: iterations}
	mut fast_times := []f64{len: iterations}
	mut reference_output := softplus_nmap_reference(gradient, cached) or { panic(err) }
	mut fast_output := internal.deriv_softplus[f64](gradient, cached) or { panic(err) }
	for i in 0 .. iterations {
		if i % 2 == 0 {
			mut start := time.sys_mono_now()
			reference_output = softplus_nmap_reference(gradient, cached) or { panic(err) }
			reference_times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
			start = time.sys_mono_now()
			fast_output = internal.deriv_softplus[f64](gradient, cached) or { panic(err) }
			fast_times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
		} else {
			mut start := time.sys_mono_now()
			fast_output = internal.deriv_softplus[f64](gradient, cached) or { panic(err) }
			fast_times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
			start = time.sys_mono_now()
			reference_output = softplus_nmap_reference(gradient, cached) or { panic(err) }
			reference_times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
		}
	}
	return mean(mut reference_times), mean(mut fast_times), reference_output, fast_output
}

fn gelu_nmap_reference(gradient &vtl.Tensor[f64], cached &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	return gradient.nmap([cached], fn (values []f64, _ []int) f64 {
		x := values[1]
		z := 0.7978845608028654 * (x + 0.044715 * x * x * x)
		tanh_z := math.tanh(z)
		dz_dx := 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
		derivative := 0.5 * (1.0 + tanh_z + x * (1.0 - tanh_z * tanh_z) * dz_dx)
		return values[0] * derivative
	})
}

fn softplus_nmap_reference(gradient &vtl.Tensor[f64], cached &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	return gradient.nmap([cached], fn (values []f64, _ []int) f64 {
		x := values[1]
		sigmoid := if x >= 0 {
			1.0 / (1.0 + math.exp(-x))
		} else {
			exp_x := math.exp(x)
			exp_x / (1.0 + exp_x)
		}
		return values[0] * sigmoid
	})
}

fn assert_outputs_equal(reference &vtl.Tensor[f64], optimized &vtl.Tensor[f64]) {
	assert reference.shape == optimized.shape
	for i in 0 .. reference.size {
		assert reference.data.data[i] == optimized.data.data[i]
	}
}

fn mean(mut values []f64) f64 {
	mut total := 0.0
	for value in values {
		total += value
	}
	return total / f64(values.len)
}
