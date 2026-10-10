// Compile with -prod from ~/.vmodules, then run the binary; see README.md.
module main

import time
import vtl
import vtl.stats
import vtl.benchmarks.util as bu

const putmask_size = 1_000_000
const putmask_iterations = 7

fn main() {
	mut initial := []f64{len: putmask_size}
	mut mask_data := []bool{len: putmask_size}
	for i in 0 .. putmask_size {
		initial[i] = f64(i % 97)
		mask_data[i] = i % 2 == 1
	}
	target := vtl.from_1d(initial)!
	mask := vtl.from_1d(mask_data)!
	updates := vtl.from_1d([101.0, 202.0, 303.0])!
	for _ in 0 .. 2 {
		mut warmup := target.copy(.row_major)
		warmup.putmask(mask, updates)!
	}
	mut samples := []f64{len: putmask_iterations}
	mut checksum := 0.0
	for i in 0 .. putmask_iterations {
		mut result := target.copy(.row_major)
		started := time.sys_mono_now()
		result.putmask(mask, updates)!
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
		checksum += stats.sum(result)
	}
	average := bu.mean_time_ms(mut samples)
	bu.print_header('VTL Tensor.putmask benchmark')
	println('vtl putmask ${putmask_size} f64, 50% mask, 3 updates | ${average:.3f} ms')
	println('checksum: ${checksum:.3f}')
}
