// VTL population-variance benchmark. Compile with -prod from ~/.vmodules, then run the binary.
module main

import time
import vtl
import vtl.stats

fn main() {
	println('VTL contiguous population variance benchmark (f64)')
	println('Elements | Mean (ms) | Checksum')
	println('---------+-----------+---------')
	for n in [100_000, 1_000_000] {
		mut values := []f64{len: n}
		for i in 0 .. n {
			values[i] = f64(i % 1000 + 1) / 1000.0
		}
		tensor := vtl.from_array(values, [n])!
		for _ in 0 .. 3 {
			_ = stats.population_variance(tensor)
		}
		mut total_ns := i64(0)
		mut checksum := 0.0
		iterations := 10
		for _ in 0 .. iterations {
			started := time.sys_mono_now()
			result := stats.population_variance(tensor)
			total_ns += time.sys_mono_now() - started
			checksum += result
		}
		mean_ms := f64(total_ns) / f64(iterations) / 1_000_000.0
		println('${n} | ${mean_ms:.4f} | ${checksum:.6f}')
	}
}
