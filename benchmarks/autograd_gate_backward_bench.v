// Compare one-pass binary autograd gate gradients with separate tensor passes.
// Compile with -prod from ~/.vmodules, then execute the resulting binary.
module main

import math
import time
import vtl
import vtl.autograd

const sizes = [1024, 65536, 1048576]
const warmup_runs = 3
const iterations = 10

fn main() {
	println('VTL binary autograd gate backward (f64, CPU)')
	println('Elements | Multiply separate (ms) | fused (ms) | speedup | Divide separate (ms) | fused (ms) | speedup')
	for size in sizes {
		multiply_separate, multiply_fused := benchmark_multiply(size)!
		divide_separate, divide_fused := benchmark_divide(size)!
		println('${size} | ${multiply_separate:.4f} | ${multiply_fused:.4f} | ${multiply_separate / multiply_fused:.2f}x | ${divide_separate:.4f} | ${divide_fused:.4f} | ${divide_separate / divide_fused:.2f}x')
	}
}

fn benchmark_multiply(size int) !(f64, f64) {
	left_values, right_values, gradient_values := inputs(size)
	left := vtl.from_array[f64](left_values, [size])!
	right := vtl.from_array[f64](right_values, [size])!
	gradient := vtl.from_array[f64](gradient_values, [size])!
	ctx := autograd.ctx[f64]()
	left_variable := ctx.variable(left)
	right_variable := ctx.variable(right)
	mut result := ctx.variable(vtl.zeros[f64]([size]))
	result.grad = gradient
	payload := autograd.payload(result)
	gate := autograd.multiply_gate[f64](left_variable, right_variable)
	mut separate := []&vtl.Tensor[f64]{}
	mut fused := []&vtl.Tensor[f64]{}
	for _ in 0 .. warmup_runs {
		separate = separate_multiply_backward(gradient, left, right)!
		fused = gate.backward(payload)!
	}
	separate_ms := time_separate_multiply(gradient, left, right)
	fused_ms := time_fused_multiply(gate, payload)
	assert math.abs(pair_checksum(separate) - pair_checksum(fused)) < 1e-9
	return separate_ms, fused_ms
}

fn benchmark_divide(size int) !(f64, f64) {
	left_values, right_values, gradient_values := inputs(size)
	left := vtl.from_array[f64](left_values, [size])!
	right := vtl.from_array[f64](right_values, [size])!
	gradient := vtl.from_array[f64](gradient_values, [size])!
	ctx := autograd.ctx[f64]()
	left_variable := ctx.variable(left)
	right_variable := ctx.variable(right)
	mut result := ctx.variable(vtl.zeros[f64]([size]))
	result.grad = gradient
	payload := autograd.payload(result)
	gate := autograd.divide_gate[f64](left_variable, right_variable)
	mut separate := []&vtl.Tensor[f64]{}
	mut fused := []&vtl.Tensor[f64]{}
	for _ in 0 .. warmup_runs {
		separate = separate_divide_backward(gradient, left, right)!
		fused = gate.backward(payload)!
	}
	separate_ms := time_separate_divide(gradient, left, right)
	fused_ms := time_fused_divide(gate, payload)
	assert math.abs(pair_checksum(separate) - pair_checksum(fused)) < 1e-8
	return separate_ms, fused_ms
}

fn separate_multiply_backward(gradient &vtl.Tensor[f64], left &vtl.Tensor[f64], right &vtl.Tensor[f64]) ![]&vtl.Tensor[f64] {
	return [gradient.multiply[f64](right)!, gradient.multiply[f64](left)!]
}

fn separate_divide_backward(gradient &vtl.Tensor[f64], left &vtl.Tensor[f64], right &vtl.Tensor[f64]) ![]&vtl.Tensor[f64] {
	left_gradient := gradient.divide[f64](right)!
	right_squared := right.multiply[f64](right)!
	negative_gradient := gradient.multiply_scalar[f64](-1.0)!
	right_numerator := negative_gradient.multiply[f64](left)!
	right_gradient := right_numerator.divide[f64](right_squared)!
	return [left_gradient, right_gradient]
}

fn time_separate_multiply(gradient &vtl.Tensor[f64], left &vtl.Tensor[f64], right &vtl.Tensor[f64]) f64 {
	mut times := []f64{len: iterations}
	for i in 0 .. iterations {
		start := time.sys_mono_now()
		_ := separate_multiply_backward(gradient, left, right) or { panic(err) }
		times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
	}
	return mean(mut times)
}

fn time_fused_multiply(gate &autograd.MultiplyGate[f64], payload &autograd.Payload[f64]) f64 {
	mut times := []f64{len: iterations}
	for i in 0 .. iterations {
		start := time.sys_mono_now()
		_ := gate.backward(payload) or { panic(err) }
		times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
	}
	return mean(mut times)
}

fn time_separate_divide(gradient &vtl.Tensor[f64], left &vtl.Tensor[f64], right &vtl.Tensor[f64]) f64 {
	mut times := []f64{len: iterations}
	for i in 0 .. iterations {
		start := time.sys_mono_now()
		_ := separate_divide_backward(gradient, left, right) or { panic(err) }
		times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
	}
	return mean(mut times)
}

fn time_fused_divide(gate &autograd.DivideGate[f64], payload &autograd.Payload[f64]) f64 {
	mut times := []f64{len: iterations}
	for i in 0 .. iterations {
		start := time.sys_mono_now()
		_ := gate.backward(payload) or { panic(err) }
		times[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
	}
	return mean(mut times)
}

fn pair_checksum(pair []&vtl.Tensor[f64]) f64 {
	mut total := 0.0
	for i in 0 .. pair[0].size {
		total += pair[0].data.data[i] + pair[1].data.data[i]
	}
	return total
}

fn inputs(size int) ([]f64, []f64, []f64) {
	mut left := []f64{len: size}
	mut right := []f64{len: size}
	mut gradient := []f64{len: size}
	for i in 0 .. size {
		left[i] = 0.25 + f64(i % 11) / 10.0
		right[i] = 0.5 + f64(i % 31) / 100.0
		gradient[i] = 0.1 + f64(i % 7) / 100.0
	}
	return left, right, gradient
}

fn mean(mut values []f64) f64 {
	mut total := 0.0
	for value in values {
		total += value
	}
	return total / f64(values.len)
}
