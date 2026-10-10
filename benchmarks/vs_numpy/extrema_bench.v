// Compile with -prod from ~/.vmodules, then run the binary.
module main

import time
import vtl

const extrema_size = 1_048_576
const extrema_iterations = 7

fn main() {
	mut left_values := []f64{len: extrema_size}
	mut right_values := []f64{len: extrema_size}
	for i in 0 .. extrema_size {
		left_values[i] = f64(i % 1000 - 500) * 0.25
		right_values[i] = f64(i % 997 - 498) * 0.25
	}
	left := vtl.from_1d[f64](left_values) or { panic(err) }
	right := vtl.from_1d[f64](right_values) or { panic(err) }
	for _ in 0 .. 2 {
		_ = left.max(right) or { panic(err) }
	}
	mut result := left.max(right) or { panic(err) }
	mut samples := []f64{len: extrema_iterations}
	for i in 0 .. extrema_iterations {
		started := time.sys_mono_now()
		result = left.max(right) or { panic(err) }
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
	}
	mut checksum := i64(0)
	for value in result.data.data {
		checksum += i64(value * 4.0)
	}
	mut total := f64(0)
	for sample in samples {
		total += sample
	}
	println('vtl maximum ${extrema_size} f64 | ${total / extrema_iterations:.3} ms')
	println('checksum: ${checksum}')
}
