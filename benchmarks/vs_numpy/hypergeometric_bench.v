// Compile with -prod from ~/.vmodules, then run the binary; see benchmarks/README.md.
module main

import time
import vtl
import vtl.benchmarks.util as bu

const output_size = 100_000

fn main() {
	mut generator := vtl.new_random_generator(42)
	for _ in 0 .. 2 {
		_ := generator.hypergeometric(50_000, 50_000, 50_000, [output_size])!
	}
	mut samples := []f64{len: 7}
	mut checksum := 0
	for i in 0 .. samples.len {
		started := time.sys_mono_now()
		values := generator.hypergeometric(50_000, 50_000, 50_000, [output_size])!
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
		checksum += values.get_nth(0)
	}
	average := bu.mean_time_ms(mut samples)
	bu.print_header('VTL hypergeometric benchmark')
	bu.print_table_header()
	bu.print_row('hypergeometric', '${output_size} samples (population 100,000; sample 50,000)', average,
		'7 samples')
	println('checksum: ${checksum}')
	generator.free()
}
