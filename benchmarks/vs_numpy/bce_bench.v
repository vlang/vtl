// Compile with -prod from ~/.vmodules, then run the binary.
module main

import os
import time
import vtl
import vtl.nn.internal
import math

const bce_iterations = 3

fn previous_bce[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	one := vtl.cast[T](1)
	epsilon := vtl.cast[T](1e-7)
	mut clamped := []T{len: input.size()}
	for i in 0 .. input.size() {
		value := input.data.data[i]
		clamped[i] = if value <= epsilon {
			epsilon
		} else if value >= one - epsilon {
			one - epsilon
		} else {
			value
		}
	}
	mut sum := vtl.cast[T](0)
	for i in 0 .. input.size() {
		p := clamped[i]
		y := target.data.data[i]
		sum += -y * vtl.cast[T](math.log(vtl.cast[f64](p))) -
			(one - y) * vtl.cast[T](math.log(vtl.cast[f64](one - p)))
	}
	return vtl.from_1d([sum / vtl.cast[T](input.size())])
}

fn measure_bce(name string, input &vtl.Tensor[f64], target &vtl.Tensor[f64], previous bool) ! {
	for _ in 0 .. 1 {
		if previous {
			_ = previous_bce[f64](input, target)!
		} else {
			_ = internal.bce[f64](input, target)!
		}
	}
	mut samples := []f64{len: bce_iterations}
	mut checksum := f64(0)
	for i in 0 .. bce_iterations {
		started := time.sys_mono_now()
		result := if previous {
			previous_bce[f64](input, target)!
		} else {
			internal.bce[f64](input, target)!
		}
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
		checksum = result.get_nth[f64](0)
	}
	samples.sort()
	println('${name} | ${samples[samples.len / 2]:.3} ms')
	println('checksum: ${checksum:.12f}')
}

fn main() {
	requested_size := os.getenv('VTL_BCE_BENCH_SIZE')
	size := if requested_size == '' { 524_288 } else { requested_size.int() }
	if size <= 0 {
		eprintln('VTL_BCE_BENCH_SIZE must be a positive integer')
		exit(2)
	}
	mut probabilities := []f64{len: size}
	mut targets := []f64{len: size}
	for i in 0 .. size {
		probabilities[i] = 0.000001 + f64(i % 999_999) * (0.999998 / 999_999.0)
		targets[i] = if i % 2 == 0 { 1.0 } else { 0.0 }
	}
	input := vtl.from_array[f64](probabilities, [size])!
	target := vtl.from_array[f64](targets, [size])!
	println('binary cross entropy f64 | ${size} values | median of ${bce_iterations} calls after 1 warmup')
	measure_bce('VTL fused', input, target, false) or { panic(err) }
	measure_bce('VTL previous two-pass', input, target, true) or { panic(err) }
}
