// Contiguous unary-gate backward microbenchmarks. Compile with -prod first.
module main

import time
import vtl
import vtl.autograd

const element_count = 1_048_576
const warmup_count = 3
const iteration_count = 9

fn main() {
	ctx := autograd.ctx[f64]()
	gradient := vtl.full[f64]([element_count], 1.25)
	mut output := ctx.variable(vtl.zeros[f64]([element_count]))
	output.grad = gradient
	payload := autograd.payload(output)

	tanh := autograd.tanh_gate[f64](vtl.full[f64]([element_count], 0.5))
	mut tanh_samples := []f64{len: iteration_count}
	mut tanh_checksum := 0.0
	for _ in 0 .. warmup_count {
		tanh.backward(payload) or { panic(err) }
	}
	for i in 0 .. iteration_count {
		start := time.sys_mono_now()
		result := tanh.backward(payload) or { panic(err) }
		tanh_samples[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
		tanh_checksum = result[0].get_nth(0) * f64(element_count)
	}
	print_bench('tanh', mut tanh_samples, tanh_checksum)

	abs_input := ctx.variable(vtl.full[f64]([element_count], 0.5))
	abs := autograd.abs_gate[f64](abs_input)
	mut abs_samples := []f64{len: iteration_count}
	mut abs_checksum := 0.0
	for _ in 0 .. warmup_count {
		abs.backward(payload) or { panic(err) }
	}
	for i in 0 .. iteration_count {
		start := time.sys_mono_now()
		result := abs.backward(payload) or { panic(err) }
		abs_samples[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
		abs_checksum = result[0].get_nth(0) * f64(element_count)
	}
	print_bench('abs', mut abs_samples, abs_checksum)

	sqrt_input := ctx.variable(vtl.full[f64]([element_count], 4.0))
	sqrt := autograd.sqrt_gate[f64](sqrt_input)
	mut sqrt_samples := []f64{len: iteration_count}
	mut sqrt_checksum := 0.0
	for _ in 0 .. warmup_count {
		sqrt.backward(payload) or { panic(err) }
	}
	for i in 0 .. iteration_count {
		start := time.sys_mono_now()
		result := sqrt.backward(payload) or { panic(err) }
		sqrt_samples[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
		sqrt_checksum = result[0].get_nth(0) * f64(element_count)
	}
	print_bench('sqrt', mut sqrt_samples, sqrt_checksum)

	clamp := autograd.clamp_gate[f64](-1.0, 1.0, vtl.zeros[f64]([element_count]))
	mut clamp_samples := []f64{len: iteration_count}
	mut clamp_checksum := 0.0
	for _ in 0 .. warmup_count {
		clamp.backward(payload) or { panic(err) }
	}
	for i in 0 .. iteration_count {
		start := time.sys_mono_now()
		result := clamp.backward(payload) or { panic(err) }
		clamp_samples[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
		clamp_checksum = result[0].get_nth(0) * f64(element_count)
	}
	print_bench('clamp', mut clamp_samples, clamp_checksum)
}

fn print_bench(name string, mut samples []f64, checksum f64) {
	samples.sort()
	println('op=${name} elements=${element_count} median_ms=${samples[samples.len / 2]:.3f} checksum=${checksum:.3f}')
}
