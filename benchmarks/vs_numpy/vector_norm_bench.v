// Run from ~/.vmodules with `v -prod run`.
module main

import time
import vtl
import vtl.la
import vtl.benchmarks.util as bu

const value_count = 500_000

fn main() {
	mut values := []f64{len: value_count}
	for i in 0 .. value_count {
		values[i] = f64(i % 97) / 13.0 - 3.0
	}
	tensor := vtl.from_array(values, [value_count])!
	for _ in 0 .. 3 {
		_ := la.vector_norm_axes(tensor, 2, [0], false)!
	}
	mut samples := []f64{len: 7}
	mut checksum := 0.0
	for i in 0 .. samples.len {
		started := time.sys_mono_now()
		result := la.vector_norm_axes(tensor, 2, [0], false)!
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
		checksum += result.get_nth(0)
	}
	average := bu.mean_time_ms(mut samples)
	bu.print_header('VTL vector p=2 norm benchmark')
	bu.print_table_header()
	bu.print_row('vector_norm_axes', '${value_count} f64', average, '7 samples')
	println('checksum: ${checksum}')
}
