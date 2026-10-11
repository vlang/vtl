// Compile with -prod from ~/.vmodules, then run the binary.
module main

import os
import time
import vtl
import vtl.nn.internal

const bce_iterations = 3

fn measure_bce(name string, input &vtl.Tensor[f64], target &vtl.Tensor[f64]) ! {
	for _ in 0 .. 1 {
		_ = internal.bce_with_logits[f64](input, target)!
	}
	mut samples := []f64{len: bce_iterations}
	mut checksum := f64(0)
	for i in 0 .. bce_iterations {
		started := time.sys_mono_now()
		result := internal.bce_with_logits[f64](input, target)!
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
	mut logits := []f64{len: size}
	mut targets := []f64{len: size}
	for i in 0 .. size {
		logits[i] = f64(i % 2001) / 100.0 - 10.0
		targets[i] = if i % 2 == 0 { 1.0 } else { 0.0 }
	}
	input := vtl.from_array[f64](logits, [size])!
	target := vtl.from_array[f64](targets, [size])!
	println('binary cross entropy with logits f64 | ${size} values | median of ${bce_iterations} calls after 1 warmup')
	measure_bce('VTL stable fused', input, target) or { panic(err) }
}
