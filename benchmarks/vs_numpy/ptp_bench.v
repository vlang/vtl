// Compile with -prod from ~/.vmodules, then run the binary.
module main

import os
import time
import vtl
import vtl.stats

const ptp_iterations = 7

fn main() {
	requested_size := os.getenv('VTL_PTP_BENCH_SIZE')
	size := if requested_size == '' { 1_048_576 } else { requested_size.int() }
	if size <= 0 {
		eprintln('VTL_PTP_BENCH_SIZE must be a positive integer')
		exit(2)
	}
	mut values := []f64{len: size}
	for i in 0 .. size {
		values[i] = f64(i % 2001) - 1000.0
	}
	tensor := vtl.from_1d[f64](values)!
	for _ in 0 .. 2 {
		_ = stats.ptp[f64](tensor)!
	}
	mut result := f64(0)
	mut samples := []f64{len: ptp_iterations}
	for i in 0 .. ptp_iterations {
		started := time.sys_mono_now()
		result = stats.ptp[f64](tensor)!
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
	}
	mut total := f64(0)
	for sample in samples {
		total += sample
	}
	println('vtl ptp ${size} f64 | ${total / ptp_iterations:.3} ms')
	println('checksum: ${result:.1}')
}
