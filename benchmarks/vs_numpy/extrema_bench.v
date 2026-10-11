// Compile with -prod from ~/.vmodules, then run the binary.
module main

import time
import vtl

const extrema_size = 1_048_576
const extrema_iterations = 7

fn main() {
	mut left_f32 := []f32{len: extrema_size}
	mut right_f32 := []f32{len: extrema_size}
	mut left_f64 := []f64{len: extrema_size}
	mut right_f64 := []f64{len: extrema_size}
	for i in 0 .. extrema_size {
		left := i % 1000 - 500
		right := i % 997 - 498
		left_f32[i] = f32(left) * 0.25
		right_f32[i] = f32(right) * 0.25
		left_f64[i] = f64(left) * 0.25
		right_f64[i] = f64(right) * 0.25
	}
	benchmark_maximum[f32]('f32', left_f32, right_f32) or { panic(err) }
	benchmark_maximum[f64]('f64', left_f64, right_f64) or { panic(err) }
}

fn benchmark_maximum[T](dtype string, left_values []T, right_values []T) ! {
	left := vtl.from_1d[T](left_values)!
	right := vtl.from_1d[T](right_values)!
	for _ in 0 .. 2 {
		_ = left.max(right)!
	}
	mut result := left.max(right)!
	mut samples := []f64{len: extrema_iterations}
	for i in 0 .. extrema_iterations {
		started := time.sys_mono_now()
		result = left.max(right)!
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
	}
	mut checksum := i64(0)
	for value in result.data.data {
		checksum += i64(f64(value) * 4.0)
	}
	mut total := f64(0)
	for sample in samples {
		total += sample
	}
	println('vtl maximum ${extrema_size} ${dtype} | ${total / extrema_iterations:.3} ms')
	println('checksum: ${checksum}')
}
