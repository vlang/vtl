// Compare the contiguous clamp gate with an equivalent scalar tensor pass.
// Compile with -prod from ~/.vmodules, then execute the binary separately.
module main

import math
import time
import vtl
import vtl.autograd

const sizes = [1024, 65536, 1048576]
const warmup_runs = 3
const iterations = 9

fn main() {
	println('VTL clamp autograd gate backward (CPU)')
	println('dtype | elements | scalar (ms) | SIMD gate (ms) | speedup')
	for size in sizes {
		scalar_f64, gate_f64 := benchmark_f64(size)!
		println('f64 | ${size} | ${scalar_f64:.4f} | ${gate_f64:.4f} | ${scalar_f64 / gate_f64:.2f}x')
		scalar_f32, gate_f32 := benchmark_f32(size)!
		println('f32 | ${size} | ${scalar_f32:.4f} | ${gate_f32:.4f} | ${scalar_f32 / gate_f32:.2f}x')
	}
}

fn benchmark_f64(size int) !(f64, f64) {
	input := vtl.from_array[f64](input_values_f64(size), [size])!
	gradient := vtl.from_array[f64](gradient_values_f64(size), [size])!
	ctx := autograd.ctx[f64]()
	mut output := ctx.variable(vtl.zeros[f64]([size]))
	output.grad = gradient
	payload := autograd.payload(output)
	gate := autograd.clamp_gate[f64](-1.0, 1.0, input)
	mut scalar_result := vtl.tensor_like[f64](gradient)
	mut gate_result := vtl.tensor_like[f64](gradient)
	for _ in 0 .. warmup_runs {
		scalar_result = scalar_clamp_backward[f64](gradient, input, -1.0, 1.0)
		gate_result = gate.backward(payload)![0]
	}
	assert math.abs(checksum_f64(scalar_result) - checksum_f64(gate_result)) < 1e-9
	scalar_ms, scalar_checksum := time_scalar_f64(gradient, input)
	gate_ms, gate_checksum := time_gate_f64(gate, payload)
	assert math.abs(scalar_checksum - gate_checksum) < 1e-9
	return scalar_ms, gate_ms
}

fn benchmark_f32(size int) !(f64, f64) {
	input := vtl.from_array[f32](input_values_f32(size), [size])!
	gradient := vtl.from_array[f32](gradient_values_f32(size), [size])!
	ctx := autograd.ctx[f32]()
	mut output := ctx.variable(vtl.zeros[f32]([size]))
	output.grad = gradient
	payload := autograd.payload(output)
	gate := autograd.clamp_gate[f32](-1.0, 1.0, input)
	mut scalar_result := vtl.tensor_like[f32](gradient)
	mut gate_result := vtl.tensor_like[f32](gradient)
	for _ in 0 .. warmup_runs {
		scalar_result = scalar_clamp_backward[f32](gradient, input, -1.0, 1.0)
		gate_result = gate.backward(payload)![0]
	}
	assert math.abs(checksum_f32(scalar_result) - checksum_f32(gate_result)) < 1e-4
	scalar_ms, scalar_checksum := time_scalar_f32(gradient, input)
	gate_ms, gate_checksum := time_gate_f32(gate, payload)
	assert math.abs(scalar_checksum - gate_checksum) < 1e-4
	return scalar_ms, gate_ms
}

@[direct_array_access]
fn scalar_clamp_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], min_val T,
	max_val T) &vtl.Tensor[T] {
	mut result := vtl.tensor_like[T](gradient)
	for i in 0 .. gradient.size {
		value := input.data.data[i]
		result.data.data[i] = if value >= min_val && value <= max_val {
			gradient.data.data[i]
		} else {
			vtl.cast[T](0)
		}
	}
	return result
}

fn time_scalar_f64(gradient &vtl.Tensor[f64], input &vtl.Tensor[f64]) (f64, f64) {
	mut samples := []f64{len: iterations}
	mut checksum := 0.0
	for i in 0 .. iterations {
		start := time.sys_mono_now()
		result := scalar_clamp_backward[f64](gradient, input, -1.0, 1.0)
		end := time.sys_mono_now()
		samples[i] = f64(end - start) / 1_000_000.0
		checksum += checksum_f64(result)
	}
	return mean(mut samples), checksum
}

fn time_gate_f64(gate &autograd.ClampGate[f64], payload &autograd.Payload[f64]) (f64, f64) {
	mut samples := []f64{len: iterations}
	mut checksum := 0.0
	for i in 0 .. iterations {
		start := time.sys_mono_now()
		result := gate.backward(payload) or { panic(err) }
		end := time.sys_mono_now()
		samples[i] = f64(end - start) / 1_000_000.0
		checksum += checksum_f64(result[0])
	}
	return mean(mut samples), checksum
}

fn time_scalar_f32(gradient &vtl.Tensor[f32], input &vtl.Tensor[f32]) (f64, f64) {
	mut samples := []f64{len: iterations}
	mut checksum := 0.0
	for i in 0 .. iterations {
		start := time.sys_mono_now()
		result := scalar_clamp_backward[f32](gradient, input, -1.0, 1.0)
		end := time.sys_mono_now()
		samples[i] = f64(end - start) / 1_000_000.0
		checksum += checksum_f32(result)
	}
	return mean(mut samples), checksum
}

fn time_gate_f32(gate &autograd.ClampGate[f32], payload &autograd.Payload[f32]) (f64, f64) {
	mut samples := []f64{len: iterations}
	mut checksum := 0.0
	for i in 0 .. iterations {
		start := time.sys_mono_now()
		result := gate.backward(payload) or { panic(err) }
		end := time.sys_mono_now()
		samples[i] = f64(end - start) / 1_000_000.0
		checksum += checksum_f32(result[0])
	}
	return mean(mut samples), checksum
}

fn input_values_f64(size int) []f64 {
	mut values := []f64{len: size}
	for i in 0 .. size {
		values[i] = f64((i % 19) - 9) / 8.0
	}
	return values
}

fn input_values_f32(size int) []f32 {
	mut values := []f32{len: size}
	for i in 0 .. size {
		values[i] = f32((i % 19) - 9) / 8.0
	}
	return values
}

fn gradient_values_f64(size int) []f64 {
	mut values := []f64{len: size}
	for i in 0 .. size {
		values[i] = 0.25 + f64(i % 13) / 10.0
	}
	return values
}

fn gradient_values_f32(size int) []f32 {
	mut values := []f32{len: size}
	for i in 0 .. size {
		values[i] = 0.25 + f32(i % 13) / 10.0
	}
	return values
}

fn checksum_f64(tensor &vtl.Tensor[f64]) f64 {
	mut sum := 0.0
	for value in tensor.data.data {
		sum += value
	}
	return sum
}

fn checksum_f32(tensor &vtl.Tensor[f32]) f64 {
	mut sum := 0.0
	for value in tensor.data.data {
		sum += f64(value)
	}
	return sum
}

fn mean(mut values []f64) f64 {
	mut total := 0.0
	for value in values {
		total += value
	}
	return total / f64(values.len)
}
