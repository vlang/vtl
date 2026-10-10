// End-to-end VTL sum/mean benchmark. Compile with -prod from ~/.vmodules, then run the binary.
module main

import time
import vtl
import vtl.stats
import vtl.benchmarks.util as bu

fn main() {
	bu.print_header('VTL contiguous sum/mean benchmark (f64)')
	println('Benchmark | Size | Avg (ms)')
	println('----------+------+----------')
	for n in [100_000, 1_000_000] {
		mut values := []f64{len: n}
		for i in 0 .. n {
			values[i] = f64(i % 1000 + 1) / 1000.0
		}
		tensor := vtl.from_array(values, [n])!
		bench_stat('vtl_sum', tensor, false)
		bench_stat('vtl_mean', tensor, true)
	}
	println('\nNumPy reference: numpy_stats_baseline.py')
}

fn bench_stat(name string, tensor &vtl.Tensor[f64], use_mean bool) {
	for _ in 0 .. 3 {
		if use_mean {
			_ = stats.mean(tensor)
		} else {
			_ = stats.sum(tensor)
		}
	}
	mut samples := []f64{len: 10}
	mut checksum := 0.0
	for i in 0 .. samples.len {
		started := time.sys_mono_now()
		result := if use_mean { stats.mean(tensor) } else { stats.sum(tensor) }
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
		checksum += result
	}
	println('${name} | ${tensor.size} | ${bu.mean_time_ms(mut samples):.4f} | checksum=${checksum:.4f}')
}
