// Contiguous TanhGate backward microbenchmark. Compile with -prod before running.
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
	cached := vtl.full[f64]([element_count], 0.5)
	mut output := ctx.variable(vtl.zeros[f64]([element_count]))
	output.grad = gradient
	gate := autograd.tanh_gate[f64](cached)
	payload := autograd.payload(output)
	for _ in 0 .. warmup_count {
		gate.backward(payload) or { panic(err) }
	}
	mut samples := []f64{len: iteration_count}
	mut checksum := 0.0
	for i in 0 .. iteration_count {
		start := time.sys_mono_now()
		result := gate.backward(payload) or { panic(err) }
		samples[i] = f64(time.sys_mono_now() - start) / 1_000_000.0
		checksum = result[0].get_nth(0) * f64(element_count)
	}
	samples.sort()
	println('elements=${element_count} median_ms=${samples[samples.len / 2]:.3f} checksum=${checksum:.3f}')
}
