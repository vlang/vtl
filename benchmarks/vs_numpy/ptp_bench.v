// Compile with -prod from ~/.vmodules, then run the binary.
module main

import time
import vtl
import vtl.stats

const ptp_size = 1_048_576
const ptp_iterations = 7

fn main() {
	mut values := []f64{len: ptp_size}
	for i in 0 .. ptp_size {
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
	println('vtl ptp ${ptp_size} f64 | ${total / ptp_iterations:.3} ms')
	println('checksum: ${result:.1}')
}
